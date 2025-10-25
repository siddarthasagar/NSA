"""Integration tests for evaluation workflow.

This module tests the end-to-end evaluation pipeline, verifying that
the model can evaluate tasks and generate predictions correctly.
"""

import json
import os
from pathlib import Path

import jax

from small_transformer_based.flax_eval import (
    batch_predict_transformations,
    format_grid_for_tokenizer,
)
from small_transformer_based.flax_model import FlaxCustomTransformer
from small_transformer_based.flax_train import (
    CustomTokenizer,
    create_train_state,
)
from utils import PathConfig


class TestEvaluationMinimal:
    """Test minimal evaluation workflow."""

    def test_evaluation_minimal(self, test_cache_dir: Path, tmp_path: Path) -> None:
        """Test minimal evaluation workflow with 2 tasks.

        This test verifies that:
        1. Evaluation completes without errors for 2 tasks
        2. Predictions are generated for each task
        3. Results file is created in cache/data/
        4. Predictions contain valid transformation names
        5. Model inference works correctly

        Args:
            test_cache_dir: Temporary cache directory fixture
            tmp_path: Temporary path for test files
        """
        # Ensure cache directories exist
        PathConfig.ensure_cache_dirs()

        # Initialize tokenizer and load vocabulary
        tokenizer = CustomTokenizer()

        # Create minimal vocabulary for testing
        minimal_vocab = self._create_minimal_vocab()
        vocab_path = PathConfig.get_data_path("test_eval_vocab.json")

        with open(vocab_path, "w") as f:
            json.dump(minimal_vocab, f)

        tokenizer.load_vocab(vocab_path)

        # Verify vocab was loaded
        assert len(tokenizer.vocab) > 0, "Vocabulary is empty"
        assert "<PAD>" in tokenizer.vocab, "Missing <PAD> token"

        # Initialize model with small configuration
        rng = jax.random.PRNGKey(42)
        model = FlaxCustomTransformer(
            vocab_size=len(tokenizer.vocab),
            n_embd=64,  # Small for testing
            n_head=2,
            n_layer=1,
        )

        # Create train state
        state = create_train_state(rng, model, learning_rate=1e-4, vocab_size=len(tokenizer.vocab))

        # Verify state was created
        assert state is not None, "Train state is None"
        assert state.params is not None, "Model parameters are None"

        # Create test tasks
        test_tasks = self._create_test_tasks(tmp_path)

        # Verify test tasks were created
        assert len(test_tasks) == 2, f"Expected 2 test tasks, got {len(test_tasks)}"

        # Prepare task data for batch prediction
        task_data_list = []
        for task_id, task_file, data_path in test_tasks:
            task_data_list.append((task_id, task_file, data_path))

        # Run batch prediction
        try:
            predictions = batch_predict_transformations(state, model, tokenizer, task_data_list)
        except Exception as e:
            raise AssertionError(f"Batch prediction failed with error: {e}")

        # Verify predictions were generated
        assert predictions is not None, "Predictions are None"
        assert isinstance(predictions, dict), "Predictions should be a dictionary"
        assert len(predictions) > 0, "No predictions were generated"

        # Verify each task has predictions
        for task_id, _, _ in test_tasks:
            assert task_id in predictions, f"Missing predictions for task {task_id}"
            task_predictions = predictions[task_id]

            # Predictions should be a list
            assert isinstance(task_predictions, list), f"Predictions for {task_id} should be a list"

            # Predictions can be empty or contain transformation names
            if len(task_predictions) > 0:
                # Verify predictions are strings
                for pred in task_predictions:
                    assert isinstance(pred, str), f"Prediction should be string, got {type(pred)}"
                    assert len(pred) > 0, "Prediction should not be empty string"

        # Save results to cache
        results_file = PathConfig.get_data_path("test_eval_results.json")
        results = {
            "test": {
                task_id: {"predictions": predictions.get(task_id, []), "solved": False}
                for task_id, _, _ in test_tasks
            }
        }

        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

        # Verify results file was created
        assert os.path.exists(results_file), "Results file was not created"
        assert os.path.getsize(results_file) > 0, "Results file is empty"

        # Verify results file is in correct location
        assert results_file.startswith(PathConfig.DATA_DIR), "Results file not in cache/data/"

        # Load and verify results structure
        with open(results_file) as f:
            loaded_results = json.load(f)

        assert "test" in loaded_results, "Missing 'test' key in results"
        assert len(loaded_results["test"]) == 2, "Results should contain 2 tasks"

        for task_id, _, _ in test_tasks:
            assert task_id in loaded_results["test"], f"Missing task {task_id} in results"
            task_result = loaded_results["test"][task_id]
            assert "predictions" in task_result, f"Missing predictions for {task_id}"
            assert "solved" in task_result, f"Missing solved status for {task_id}"

        print(f"Evaluation completed successfully. Results saved to {results_file}")

    def test_evaluation_with_checkpoint(self, test_cache_dir: Path, tmp_path: Path) -> None:
        """Test evaluation using a saved checkpoint.

        This test verifies that evaluation works when loading a model
        from a checkpoint file.

        Args:
            test_cache_dir: Temporary cache directory fixture
            tmp_path: Temporary path for test files
        """
        # Ensure cache directories exist
        PathConfig.ensure_cache_dirs()

        # Initialize tokenizer
        tokenizer = CustomTokenizer()
        minimal_vocab = self._create_minimal_vocab()
        vocab_path = PathConfig.get_data_path("test_eval_checkpoint_vocab.json")

        with open(vocab_path, "w") as f:
            json.dump(minimal_vocab, f)

        tokenizer.load_vocab(vocab_path)

        # Initialize model
        rng = jax.random.PRNGKey(42)
        model = FlaxCustomTransformer(
            vocab_size=len(tokenizer.vocab), n_embd=32, n_head=2, n_layer=1
        )

        # Create and save checkpoint
        state = create_train_state(rng, model, learning_rate=1e-4, vocab_size=len(tokenizer.vocab))

        checkpoint_dir = PathConfig.get_checkpoint_dir("test_eval_model")
        checkpoint_path = os.path.join(checkpoint_dir, "test_checkpoint.msgpack")

        # Save checkpoint using flax serialization
        from flax import serialization

        checkpoint_dict = {"model": state.params, "step": 0}

        with open(checkpoint_path, "wb") as f:
            f.write(serialization.msgpack_serialize(checkpoint_dict))

        # Verify checkpoint was saved
        assert os.path.exists(checkpoint_path), "Checkpoint was not saved"

        # Load checkpoint
        with open(checkpoint_path, "rb") as f:
            bytes_input = f.read()

        loaded_checkpoint = serialization.msgpack_restore(bytes_input)

        # Create new state with loaded parameters
        loaded_state = state.replace(params=loaded_checkpoint["model"])

        # Verify loaded state
        assert loaded_state is not None, "Loaded state is None"
        assert loaded_state.params is not None, "Loaded parameters are None"

        # Create test task
        test_tasks = self._create_test_tasks(tmp_path, num_tasks=1)

        # Run prediction with loaded model
        predictions = batch_predict_transformations(loaded_state, model, tokenizer, test_tasks)

        # Verify predictions
        assert predictions is not None, "Predictions are None"
        assert len(predictions) > 0, "No predictions generated"

    def test_evaluation_format_grid(self, test_cache_dir: Path) -> None:
        """Test grid formatting for evaluation.

        This test verifies that grid data is correctly formatted
        for model input during evaluation.

        Args:
            test_cache_dir: Temporary cache directory fixture
        """
        # Create sample grid data
        grid_data = [
            {
                "input": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                "output": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            }
        ]

        # Format grid
        formatted = format_grid_for_tokenizer(grid_data)

        # Verify formatting
        assert isinstance(formatted, str), "Formatted output should be a string"
        assert len(formatted) > 0, "Formatted output should not be empty"
        assert "Input:" in formatted, "Formatted output should contain 'Input:' marker"
        assert "Output:" in formatted, "Formatted output should contain 'Output:' marker"
        assert "|" in formatted, "Formatted output should contain pipe separators"

        # Verify grid values are present
        for row in grid_data[0]["input"]:
            for val in row:
                assert str(val) in formatted, f"Input value {val} not in formatted output"

    def test_evaluation_cache_location(self, test_cache_dir: Path) -> None:
        """Test that evaluation uses correct cache directory paths.

        This test verifies that PathConfig correctly directs evaluation
        outputs to the cache directory structure.

        Args:
            test_cache_dir: Temporary cache directory fixture
        """
        # Ensure cache directories exist
        PathConfig.ensure_cache_dirs()

        # Test results file path
        results_file = PathConfig.get_data_path("test_results.json")
        assert PathConfig.DATA_DIR in results_file, "Results file not in cache/data/"

        # Test TTA directory path
        task_id = "test_task_123"
        tta_dir = PathConfig.get_tta_dir(task_id)
        assert PathConfig.TTA_DIR in tta_dir, "TTA directory not in cache/tta/"
        assert task_id in tta_dir, "Task ID not in TTA directory path"

        # Test TTA task file path
        tta_file = PathConfig.get_tta_task_file(task_id)
        assert PathConfig.TTA_DIR in tta_file, "TTA file not in cache/tta/"
        assert task_id in tta_file, "Task ID not in TTA file path"
        assert tta_file.endswith(".json"), "TTA file should have .json extension"

    def _create_minimal_vocab(self) -> dict[str, int]:
        """Create a minimal vocabulary for testing.

        Returns:
            Dictionary mapping tokens to integer IDs
        """
        # Basic tokens
        vocab = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
        }

        # Add transformation names
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
            "no_trans",
        ]

        for idx, trans in enumerate(transformations, start=4):
            vocab[trans] = idx

        # Add grid tokens (0-9)
        for i in range(10):
            vocab[str(i)] = len(vocab)

        # Add special characters
        vocab["|"] = len(vocab)
        vocab["\n"] = len(vocab)
        vocab["Input:"] = len(vocab)
        vocab["Output:"] = len(vocab)

        return vocab

    def _create_test_tasks(self, tmp_path: Path, num_tasks: int = 2) -> list[tuple[str, str, str]]:
        """Create test task files for evaluation.

        Args:
            tmp_path: Temporary path for test files
            num_tasks: Number of test tasks to create

        Returns:
            List of (task_id, task_file, data_path) tuples
        """
        # Create test data directory
        test_data_dir = tmp_path / "test_tasks"
        test_data_dir.mkdir(exist_ok=True)

        tasks = []

        for i in range(num_tasks):
            task_id = f"test_task_{i:03d}"
            task_file = f"{task_id}.json"
            task_path = test_data_dir / task_file

            # Create simple task data
            task_data = {
                "train": [
                    {
                        "input": [[i, i + 1, i + 2], [i + 3, i + 4, i + 5], [i + 6, i + 7, i + 8]],
                        "output": [
                            [i + 1, i + 2, i + 3],
                            [i + 4, i + 5, i + 6],
                            [i + 7, i + 8, i + 9],
                        ],
                    }
                ],
                "test": [
                    {"input": [[i, i + 1, i + 2], [i + 3, i + 4, i + 5], [i + 6, i + 7, i + 8]]}
                ],
            }

            # Normalize values to 0-9 range
            for example in task_data["train"] + task_data["test"]:
                for key in ["input", "output"]:
                    if key in example:
                        example[key] = [[val % 10 for val in row] for row in example[key]]

            # Save task file
            with open(task_path, "w") as f:
                json.dump(task_data, f)

            tasks.append((task_id, task_file, str(test_data_dir)))

        return tasks
