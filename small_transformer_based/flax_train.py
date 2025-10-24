"""
Flax training pipeline for NSA transformer model.
Replaces PyTorch training with JAX/Flax/Optax.
"""

import argparse
import hashlib
import json
import os
import re
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training import train_state
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from small_transformer_based.flax_model import FlaxCustomTransformer, count_parameters


class TrainState(train_state.TrainState):
    """Extended train state with dropout RNG."""

    dropout_rng: jax.random.PRNGKey


def create_train_state(rng, model, learning_rate, vocab_size):
    """
    Initialize model parameters and optimizer state.

    Args:
        rng: JAX random key
        model: Flax model
        learning_rate: Learning rate for optimizer
        vocab_size: Size of vocabulary

    Returns:
        TrainState with initialized parameters and optimizer
    """
    # Initialize model parameters with dummy input
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    params = model.init(rng, dummy_input, deterministic=True)

    # Create optimizer with gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(learning_rate),
    )

    # Create train state
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, dropout_rng=rng)


def cross_entropy_loss(logits, labels, padding_idx):
    """
    Compute cross-entropy loss with padding mask.

    Args:
        logits: Model logits of shape [batch, num_cls_tokens, vocab_size]
        labels: Target labels of shape [batch, num_cls_tokens]
        padding_idx: Index to ignore in loss computation

    Returns:
        Scalar loss value
    """
    # Create mask for non-padding tokens
    mask = labels != padding_idx

    # Compute cross-entropy
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot_labels = jax.nn.one_hot(labels, logits.shape[-1])
    loss = -jnp.sum(one_hot_labels * log_probs, axis=-1)

    # Apply mask and compute mean
    masked_loss = loss * mask
    return jnp.sum(masked_loss) / jnp.maximum(jnp.sum(mask), 1)


@jax.jit
def train_step(state, batch, padding_idx):
    """
    Single training step (JIT-compiled).

    Args:
        state: Current train state
        batch: Tuple of (input_ids, output_ids)
        padding_idx: Padding token index

    Returns:
        Tuple of (new_state, loss)
    """
    input_ids, output_ids = batch

    # Split RNG for dropout
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def loss_fn(params):
        # Forward pass
        attention_mask = (input_ids != padding_idx).astype(jnp.float32)
        logits = state.apply_fn(
            params,
            input_ids,
            attention_mask=attention_mask,
            deterministic=False,
            rngs={"dropout": dropout_rng},
        )

        # Compute loss
        return cross_entropy_loss(logits, output_ids, padding_idx)

    # Compute gradients
    loss, grads = jax.value_and_grad(loss_fn)(state.params)

    # Update parameters
    state = state.apply_gradients(grads=grads)
    state = state.replace(dropout_rng=new_dropout_rng)

    return state, loss


def collate_fn_jax(batch, padding_value=0):
    """
    Collate function to convert batches to JAX arrays.

    Args:
        batch: List of (input_ids, output_ids) tuples
        padding_value: Value to use for padding

    Returns:
        Tuple of padded (input_ids, output_ids) as JAX arrays
    """
    input_ids_list = [item[0] for item in batch]
    output_ids_list = [item[1] for item in batch]

    # Pad sequences
    max_input_len = max(len(ids) for ids in input_ids_list)
    max_output_len = max(len(ids) for ids in output_ids_list)

    padded_inputs = []
    padded_outputs = []

    for input_ids, output_ids in zip(input_ids_list, output_ids_list):
        # Pad inputs
        padded_input = np.pad(
            input_ids, (0, max_input_len - len(input_ids)), constant_values=padding_value
        )
        padded_inputs.append(padded_input)

        # Pad outputs
        padded_output = np.pad(
            output_ids, (0, max_output_len - len(output_ids)), constant_values=padding_value
        )
        padded_outputs.append(padded_output)

    return jnp.array(padded_inputs), jnp.array(padded_outputs)


def train_epoch(state, train_data, tokenizer, batch_size, epoch, rng):
    """
    Train for one epoch.

    Args:
        state: Current train state
        train_data: Training dataset
        tokenizer: Tokenizer instance
        batch_size: Batch size
        epoch: Current epoch number
        rng: JAX random key

    Returns:
        Updated train state
    """
    # Shuffle data
    rng, shuffle_rng = jax.random.split(rng)
    indices = jax.random.permutation(shuffle_rng, len(train_data))

    # Create batches
    num_batches = len(train_data) // batch_size
    progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}")

    total_loss = 0.0
    padding_idx = tokenizer.vocab["<PAD>"]

    for batch_idx in progress_bar:
        # Get batch indices
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]

        # Prepare batch
        batch_data = [train_data[int(i)] for i in batch_indices]
        batch = collate_fn_jax(batch_data, padding_value=padding_idx)

        # Training step
        state, loss = train_step(state, batch, padding_idx)
        total_loss += float(loss)

        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    return state


def save_checkpoint(state, path, epoch, iteration):
    """
    Save model checkpoint using Flax serialization.

    Args:
        state: Train state to save
        path: Path to save checkpoint
        epoch: Current epoch
        iteration: Current iteration
    """
    checkpoint = {
        "model": state.params,
        "optimizer": state.opt_state,
        "step": state.step,
        "epoch": epoch,
        "iteration": iteration,
    }

    # Serialize and save
    bytes_output = serialization.to_bytes(checkpoint)
    with open(path, "wb") as f:
        f.write(bytes_output)

    print(f"Checkpoint saved at {path}")


def load_checkpoint(path, state):
    """
    Load model checkpoint.

    Args:
        path: Path to checkpoint file
        state: Train state to restore into

    Returns:
        Restored train state
    """
    with open(path, "rb") as f:
        bytes_input = f.read()

    checkpoint = serialization.from_bytes(state, bytes_input)
    return checkpoint


# Import tokenizer and dataset classes from PyTorch version
# These remain unchanged as they don't depend on PyTorch tensors
def extract_input_output_pairs(text, use_2dpe=False):
    """Extract input-output pairs from formatted text."""
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


class CustomTokenizer:
    """Custom tokenizer for ARC tasks (unchanged from PyTorch version)."""

    def __init__(self, special_tokens=None, transformations=None):
        if special_tokens is None:
            special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.special_tokens = special_tokens
        self.transformations = transformations or [
            "update_color",
            "move_node",
            "extend_node",
            "move_node_max",
            "rotate_node",
            "add_border",
            "fill_rectangle",
            "extract",
            "hollow_rectangle",
            "mirror",
            "flip",
            "remove_node",
            "insert",
            "duplicate",
            "upscale_grid",
            "crop",
            "fill",
            "magnet",
            "beam",
            "shift",
            "no_trans",
            "arbitrary_duplicate",
            "rotate_duplicate",
            "mirror_grid",
            "rotate_grid",
            "connect",
            "recolor",
            "truncate",
        ]
        self.vocab = {}
        self.inv_vocab = {}
        self.token_count = Counter()
        self.build_initial_vocab()

    def build_initial_vocab(self):
        """Build initial vocabulary with special tokens and predefined tokens."""
        idx = 0
        for token in self.special_tokens:
            self.vocab[token] = idx
            idx += 1
        predefined_tokens = (
            ["Input", "Output", "\n", "|"] + [str(i) for i in range(10)] + self.transformations
        )
        for token in predefined_tokens:
            self.vocab[token] = idx
            idx += 1

    def build_vocab(self, data, min_freq=1):
        """Build vocabulary from data."""
        print("...BUILD THE VOCABULARY")
        for text in tqdm(data):
            tokens = self.tokenize(text)
            self.token_count.update(tokens)
        for token, count in self.token_count.items():
            if token not in self.vocab and count >= min_freq:
                self.vocab[token] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        """Tokenize text."""
        tokens = []
        if isinstance(text, str):
            text = text.replace("Input:", " Input ").replace("Output:", " Output ")
        elif isinstance(text, list):
            text = str(text[0])
        else:
            raise ValueError("Wrong type")
        combined_keywords = sorted(self.transformations, key=len, reverse=True)
        pattern = r"\d+|Input|:|,|Output|\||\n|\{|\}|" + "|".join(map(re.escape, combined_keywords))
        grid_tokens = re.findall(pattern, text)
        for token in grid_tokens:
            if token in self.vocab:
                tokens.append(token)
            else:
                tokens.append("<UNK>")
        return tokens

    def encode(self, text, max_length=None):
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        token_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        token_ids = [self.vocab["<BOS>"]] + token_ids + [self.vocab["<EOS>"]]
        if max_length is not None:
            token_ids = token_ids[:max_length]
        return token_ids

    def decode(self, token_ids):
        """Decode token IDs to text."""
        tokens = [self.inv_vocab.get(token_id, "<UNK>") for token_id in token_ids]
        decoded_text = " ".join(tokens)
        return decoded_text.replace("<BOS>", "").replace("<EOS>", "").strip()

    def save_vocab(self, file_path):
        """Save vocabulary to file."""
        with open(file_path, "w") as f:
            json.dump(self.vocab, f)

    def load_vocab(self, file_path):
        """Load vocabulary from file."""
        with open(file_path) as f:
            self.vocab = json.load(f)
            self.inv_vocab = {v: k for k, v in self.vocab.items()}


class CustomDataset:
    """Custom dataset for training (adapted for JAX)."""

    def __init__(self, data, tokenizer, max_length=25600):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input"]
        output_text = str(item["output"])
        combined_input = extract_input_output_pairs(input_text)
        input_ids = self.tokenizer.encode(combined_input, max_length=self.max_length)
        output_ids = self.tokenizer.encode(output_text, max_length=self.max_length)
        return np.array(input_ids, dtype=np.int32), np.array(output_ids, dtype=np.int32)


def calculate_file_hash(file_path):
    """Calculate MD5 hash of file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def balance_transformations(data):
    """Balance dataset by transformation type."""
    required_transformations = [
        "update_color",
        "move_node",
        "extend_node",
        "move_node_max",
        "rotate_node",
        "add_border",
        "fill_rectangle",
        "extract",
        "hollow_rectangle",
        "mirror",
        "flip",
        "remove_node",
        "insert",
        "duplicate",
        "upscale_grid",
        "crop",
        "fill",
        "magnet",
        "beam",
        "shift",
        "arbitrary_duplicate",
        "rotate_duplicate",
        "rotate_grid",
        "connect",
        "recolor",
        "truncate",
    ]
    transformation_samples = {transformation: [] for transformation in required_transformations}
    less_than_two_no_trans_samples = []
    for item in data:
        output_text = item["output"].strip()
        transformations = output_text.split()
        no_trans_count = transformations.count("no_trans")
        if len(transformations) == 3 and no_trans_count == 2:
            first_transformation = transformations[0]
            if first_transformation in required_transformations:
                transformation_samples[first_transformation].append(item)
        elif no_trans_count < 2:
            less_than_two_no_trans_samples.append(item)
    non_zero_counts = [
        len(samples) for samples in transformation_samples.values() if len(samples) > 0
    ]
    if not non_zero_counts:
        print("No samples found with two 'no_trans' transformations to balance.")
        balanced_data = less_than_two_no_trans_samples.copy()
        return balanced_data
    min_count = min(non_zero_counts)
    print(f"Balancing to {min_count} samples per transformation.")
    balanced_data = []
    for transformation in required_transformations:
        samples = transformation_samples[transformation]
        if len(samples) >= min_count:
            balanced_subset = samples[:min_count]
            balanced_data.extend(balanced_subset)
    balanced_data.extend(less_than_two_no_trans_samples)
    print(f"Total balanced samples: {len(balanced_data)}")
    return balanced_data


def main(data_path, epochs=1, batch_size=32, save_iterations=100):
    """
    Main training function.

    Args:
        data_path: Path to training data JSON
        epochs: Number of training epochs
        batch_size: Batch size for training
        save_iterations: Save checkpoint every N iterations
    """
    # Initialize tokenizer
    tokenizer = CustomTokenizer()
    cache_file = "dataset_cache.txt"
    vocab_file = "vocab.json"

    # Check if dataset changed
    current_hash = calculate_file_hash(data_path)
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cached_hash = f.read().strip()
    else:
        cached_hash = None

    # Load data
    with open(data_path) as f:
        data = json.load(f)

    if not len(data):
        raise ValueError("No data loaded!")

    data = [x for x in data if len(x["input"]) < 12000]
    print("Balancing transformations in the dataset...")
    balanced_data = balance_transformations(data)
    print(f"Balanced dataset contains {len(balanced_data)} samples")

    # Build or load vocabulary
    if cached_hash != current_hash or not os.path.exists(vocab_file):
        print("Dataset changed or vocabulary does not exist yet. Creating a new vocabulary.")
        all_texts = [
            f"{extract_input_output_pairs(item['input'])} {item['output']}"
            for item in tqdm(balanced_data)
        ]
        tokenizer.build_vocab(all_texts)
        tokenizer.save_vocab(vocab_file)
        with open(cache_file, "w") as f:
            f.write(current_hash)
    else:
        print("Detected same dataset as before. Loading the vocabulary.")
        tokenizer.load_vocab(vocab_file)

    # Split data
    train_data, val_data = train_test_split(balanced_data, test_size=0.1, random_state=42)

    # Create datasets
    max_len = 25600
    train_dataset = CustomDataset(train_data, tokenizer, max_length=max_len)
    val_dataset = CustomDataset(val_data, tokenizer, max_length=max_len)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")

    # Initialize model
    rng = jax.random.PRNGKey(0)
    model = FlaxCustomTransformer(vocab_size=len(tokenizer.vocab))

    # Create train state
    state = create_train_state(rng, model, learning_rate=5e-5, vocab_size=len(tokenizer.vocab))

    # Count parameters
    total_params = count_parameters(state.params["params"])
    total_params_millions = total_params / 1_000_000
    print(f"Total number of parameters: {total_params_millions:.1f}M")

    # Create checkpoint directory
    plot_dir = f"small_transformer_based/results/{total_params_millions:.1f}M"
    os.makedirs(plot_dir, exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}/{epochs}")
        state = train_epoch(state, train_dataset, tokenizer, batch_size, epoch, rng)

        # Save checkpoint
        checkpoint_path = os.path.join(plot_dir, f"checkpoint_epoch{epoch}_final.msgpack")
        save_checkpoint(state, checkpoint_path, epoch, len(train_dataset) // batch_size)

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flax Transformer Model")
    parser.add_argument("--data_path", type=str, default="full_trans.json")
    parser.add_argument("--save_iterations", type=int, default=1)
    parser.add_argument("--print_iterations", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=25600)
    args = parser.parse_args()

    main(
        args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_iterations=args.save_iterations,
    )
