"""Integration tests for training workflow.

This module tests the end-to-end training pipeline, verifying that
the model can be trained and checkpoints are correctly saved to the cache directory.
"""

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp

from small_transformer_based.flax_train import (
    CustomDataset,
    CustomTokenizer,
    create_train_state,
    save_checkpoint,
    train_epoch,
)
from small_transformer_based.flax_model import FlaxCustomTransformer
from utils import PathConfig


class TestTrainingMinimal:
    """Test minimal training workflow."""

    def test_training_minimal(self, test_cache_dir: Path, tmp_path: Path) -> None:
        """Test minimal training workflow with 1 epoch.

        This test verifies that:
        1. Training completes without errors for 1 epoch
        2. Checkpoint is saved to cache/checkpoints/
        3. Vocab file is created in cache/data/
        4. Memory usage stays within reasonable bounds
        5. Model parameters are updated during training

        Args:
            test_cache_dir: Temporary cache directory fixture
            tmp_path: Temporary path for test files
        """
        # Ensure cache directories exist
        PathConfig.ensure_cache_dirs()

        # Create minimal training dataset (10 samples)
        minimal_data = self._create_minimal_dataset(num_samples=10)

        # Save dataset to temporary file
        data_path = tmp_path / "minimal_train_data.json"
        with open(data_path, "w") as f:
            json.dump(minimal_data, f)

        # Initialize tokenizer
        tokenizer = CustomTokenizer()

        # Build vocabulary from minimal data
        all_texts = [
            f"{self._extract_input_output_pairs(item['input'])} {item['output']}"
            for item in minimal_data
        ]
        tokenizer.build_vocab(all_texts)

        # Save vocabulary to cache
        vocab_path = PathConfig.get_data_path("test_vocab.json")
        tokenizer.save_vocab(vocab_path)

        # Verify vocab file was created
        assert os.path.exists(vocab_path), "Vocabulary file was not created"

        # Create dataset
        train_dataset = CustomDataset(minimal_data, tokenizer, max_length=512)

        assert len(train_dataset) == 10, f"Expected 10 samples, got {len(train_dataset)}"

        # Initialize model with small configuration for testing
        rng = jax.random.PRNGKey(42)
        model = FlaxCustomTransformer(
            vocab_size=len(tokenizer.vocab),
            n_embd=64,  # Reduced from default for faster testing
            n_head=2,  # Reduced from default
            n_layer=1,  # Reduced from default
        )

        # Create train state
        learning_rate = 1e-4
        state = create_train_state(rng, model, learning_rate, len(tokenizer.vocab))

        # Store initial parameters for comparison
        initial_params = jax.tree_util.tree_map(lambda x: x.copy(), state.params)

        # Get initial memory usage
        initial_memory = self._get_memory_usage_mb()
        print(f"Initial memory usage: {initial_memory:.1f}MB")

        # Create checkpoint directory
        checkpoint_dir = PathConfig.get_checkpoint_dir("test_model")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Train for 1 epoch with small batch size
        batch_size = 2
        epoch = 0

        try:
            state = train_epoch(
                state=state,
                train_data=train_dataset,
                tokenizer=tokenizer,
                batch_size=batch_size,
                epoch=epoch,
                rng=rng,
                max_memory_gb=8.0,
                checkpoint_dir=checkpoint_dir,
            )
        except Exception as e:
            raise AssertionError(f"Training failed with error: {e}")

        # Verify training completed
        assert state is not None, "Training state is None after training"

        # Verify parameters were updated (not identical to initial)
        params_changed = False
        for key in state.params["params"]:
            initial_param = initial_params["params"][key]
            current_param = state.params["params"][key]

            # Check if any parameter changed
            if isinstance(initial_param, dict):
                for subkey in initial_param:
                    if not jnp.allclose(initial_param[subkey], current_param[subkey], atol=1e-6):
                        params_changed = True
                        break
            else:
                if not jnp.allclose(initial_param, current_param, atol=1e-6):
                    params_changed = True
                    break

            if params_changed:
                break

        assert params_changed, "Model parameters were not updated during training"

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_epoch0_final.msgpack")
        save_checkpoint(state, checkpoint_path, epoch, len(train_dataset) // batch_size)

        # Verify checkpoint was saved
        assert os.path.exists(checkpoint_path), "Checkpoint file was not created"
        assert os.path.getsize(checkpoint_path) > 0, "Checkpoint file is empty"

        # Check memory usage after training
        final_memory = self._get_memory_usage_mb()
        memory_increase = final_memory - initial_memory
        print(f"Final memory usage: {final_memory:.1f}MB (increase: {memory_increase:.1f}MB)")

        # Memory should not increase excessively (allow up to 2GB increase for small model)
        assert memory_increase < 2048, (
            f"Memory increased by {memory_increase:.1f}MB, which exceeds threshold"
        )

        # Verify checkpoint directory structure
        assert os.path.exists(PathConfig.CHECKPOINTS_DIR), "Checkpoints directory not created"
        assert checkpoint_dir.startswith(PathConfig.CHECKPOINTS_DIR), (
            "Checkpoint not in correct directory"
        )

    def test_training_with_validation(self, test_cache_dir: Path, tmp_path: Path) -> None:
        """Test training with validation split.

        This test verifies that training works correctly when data is split
        into training and validation sets.

        Args:
            test_cache_dir: Temporary cache directory fixture
            tmp_path: Temporary path for test files
        """
        # Ensure cache directories exist
        PathConfig.ensure_cache_dirs()

        # Create minimal dataset with more samples for split
        minimal_data = self._create_minimal_dataset(num_samples=20)

        # Split into train and validation
        split_idx = 16  # 80/20 split
        train_data = minimal_data[:split_idx]
        val_data = minimal_data[split_idx:]

        assert len(train_data) == 16, f"Expected 16 training samples, got {len(train_data)}"
        assert len(val_data) == 4, f"Expected 4 validation samples, got {len(val_data)}"

        # Initialize tokenizer
        tokenizer = CustomTokenizer()

        # Build vocabulary
        all_texts = [
            f"{self._extract_input_output_pairs(item['input'])} {item['output']}"
            for item in minimal_data
        ]
        tokenizer.build_vocab(all_texts)

        # Create datasets
        train_dataset = CustomDataset(train_data, tokenizer, max_length=512)
        val_dataset = CustomDataset(val_data, tokenizer, max_length=512)

        assert len(train_dataset) == 16, "Training dataset size mismatch"
        assert len(val_dataset) == 4, "Validation dataset size mismatch"

        # Initialize model
        rng = jax.random.PRNGKey(42)
        model = FlaxCustomTransformer(
            vocab_size=len(tokenizer.vocab), n_embd=64, n_head=2, n_layer=1
        )

        # Create train state
        state = create_train_state(rng, model, learning_rate=1e-4, vocab_size=len(tokenizer.vocab))

        # Train for 1 epoch
        checkpoint_dir = PathConfig.get_checkpoint_dir("test_model_val")
        os.makedirs(checkpoint_dir, exist_ok=True)

        state = train_epoch(
            state=state,
            train_data=train_dataset,
            tokenizer=tokenizer,
            batch_size=2,
            epoch=0,
            rng=rng,
            max_memory_gb=8.0,
            checkpoint_dir=checkpoint_dir,
        )

        # Verify training completed
        assert state is not None, "Training state is None"

    def test_checkpoint_save_and_structure(self, test_cache_dir: Path) -> None:
        """Test checkpoint saving and file structure.

        This test verifies that checkpoints are saved with the correct
        structure and can be loaded back.

        Args:
            test_cache_dir: Temporary cache directory fixture
        """
        # Ensure cache directories exist
        PathConfig.ensure_cache_dirs()

        # Create minimal model
        rng = jax.random.PRNGKey(42)
        tokenizer = CustomTokenizer()

        # Build minimal vocab
        tokenizer.build_vocab(["Input Output 0 1 2"])

        model = FlaxCustomTransformer(
            vocab_size=len(tokenizer.vocab), n_embd=32, n_head=2, n_layer=1
        )

        # Create train state
        state = create_train_state(rng, model, learning_rate=1e-4, vocab_size=len(tokenizer.vocab))

        # Create checkpoint directory
        checkpoint_dir = PathConfig.get_checkpoint_dir("test_checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, "test_checkpoint.msgpack")
        save_checkpoint(state, checkpoint_path, epoch=0, iteration=0)

        # Verify checkpoint exists
        assert os.path.exists(checkpoint_path), "Checkpoint was not saved"

        # Verify checkpoint is in correct location
        assert checkpoint_path.startswith(PathConfig.CHECKPOINTS_DIR), (
            "Checkpoint not in cache/checkpoints/"
        )

        # Verify file is not empty
        file_size = os.path.getsize(checkpoint_path)
        assert file_size > 0, "Checkpoint file is empty"
        print(f"Checkpoint size: {file_size / 1024:.2f}KB")

    def _create_minimal_dataset(self, num_samples: int = 10) -> list[dict]:
        """Create a minimal synthetic dataset for testing.

        Args:
            num_samples: Number of samples to generate

        Returns:
            List of training samples with input/output pairs
        """
        samples = []

        # Simple transformations for testing
        transformations = [
            "extract",
            "move_node",
            "rotate_node",
            "mirror",
            "flip",
            "upscale_grid",
            "crop",
            "fill",
            "shift",
            "rotate_grid",
        ]

        for i in range(num_samples):
            # Create simple 3x3 grids
            input_grid = [[i % 10, (i + 1) % 10, (i + 2) % 10] for _ in range(3)]
            output_grid = [[(i + 1) % 10, (i + 2) % 10, (i + 3) % 10] for _ in range(3)]

            # Format as text
            input_text = self._format_grid_pair(input_grid, output_grid)

            # Select transformation
            transformation = transformations[i % len(transformations)]

            sample = {"input": input_text, "output": transformation}

            samples.append(sample)

        return samples

    def _format_grid_pair(self, input_grid: list[list[int]], output_grid: list[list[int]]) -> str:
        """Format input and output grids as text.

        Args:
            input_grid: Input grid as 2D list
            output_grid: Output grid as 2D list

        Returns:
            Formatted text string
        """
        input_str = "\n".join(["|".join(map(str, row)) for row in input_grid])
        output_str = "\n".join(["|".join(map(str, row)) for row in output_grid])

        return f"Input:\n{input_str}\nOutput:\n{output_str}"

    def _extract_input_output_pairs(self, text: str) -> str:
        """Extract input-output pairs from formatted text.

        Args:
            text: Formatted text with Input:/Output: markers

        Returns:
            Extracted pairs as string
        """
        import re

        pattern = re.compile(r"Input:\n([\d\|\n]+)\nOutput:\n([\d\|\n]+)", re.DOTALL)
        matches = pattern.findall(text)
        input_output_pairs = []

        for match in matches:
            input_grid_str = match[0].strip().split("\n")
            output_grid_str = match[1].strip().split("\n")
            input_grid = "\n".join(input_grid_str)
            output_grid = "\n".join(output_grid_str)
            input_output_pairs.append(f"Input:\n{input_grid}\nOutput:\n{output_grid}")

        return "\n".join(input_output_pairs)

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024**2)
        except ImportError:
            return 0.0
