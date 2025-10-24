# Design Document: PyTorch to JAX/Flax Migration

## Overview

This design document outlines the architecture and implementation strategy for migrating the NSA transformer model from PyTorch to JAX/Flax/Optax. The migration focuses on reducing memory footprint while maintaining the paper's three-phase methodology and achieving equivalent model performance.

### Key Design Goals

1. **Memory Efficiency**: Reduce framework overhead by replacing PyTorch with JAX
2. **Functional Programming**: Leverage JAX's functional paradigm for cleaner state management
3. **Workflow Preservation**: Maintain pre-training → TTA → DSL search pipeline
4. **Performance Parity**: Achieve equivalent or better solve rates on ARC tasks
5. **Minimal Disruption**: Keep non-ML components (tokenizer, DSL, Task) unchanged

## Architecture

### High-Level Component Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    NSA System (Unchanged)                    │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Tokenizer  │  │ ARCGraph DSL │  │  Task.solve()    │   │
│  │ (Python)   │  │  (Python)    │  │   (Python)       │   │
│  └────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              ML Components (JAX/Flax Migration)              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Transformer Model (Flax nn.Module)                  │  │
│  │  - Embedding Layer                                   │  │
│  │  - Positional Encoding                               │  │
│  │  - 8x Transformer Encoder Layers                     │  │
│  │  - 3x Classification Heads                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Training Pipeline (JAX/Optax)                       │  │
│  │  - Data Loading → JAX Arrays                         │  │
│  │  - Loss Computation (CrossEntropy)                   │  │
│  │  - Gradient Computation (jax.grad)                   │  │
│  │  - Optimizer Updates (Optax AdamW)                   │  │
│  │  - Checkpoint Management                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Inference Pipeline (JAX)                            │  │
│  │  - Batch Prediction                                  │  │
│  │  - Test-Time Adaptation (TTA)                        │  │
│  │  - Top-K Transformation Selection                    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Flax Transformer Model

**File**: `small_transformer_based/flax_model.py`

**Design Decisions**:
- Use Flax `nn.Module` with `@nn.compact` decorator for concise layer definitions
- Implement custom `SinusoidalPositionalEncoding` as a Flax module
- Use `nn.SelfAttention` for transformer encoder layers
- Maintain exact parameter count (25.3M) through identical architecture

**Key Classes**:

```python
class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence data"""
    n_embd: int
    max_len: int = 6500
    
    @nn.compact
    def __call__(self, x):
        # Generate sinusoidal position embeddings
        # Add to input embeddings
        pass

class FlaxTransformerEncoder(nn.Module):
    """Single transformer encoder layer"""
    n_embd: int = 512
    n_head: int = 8
    n_inner: int = 2048
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        # Self-attention + LayerNorm + Feedforward + LayerNorm
        pass

class FlaxCustomTransformer(nn.Module):
    """Main transformer model with 3 classification tokens"""
    vocab_size: int
    n_embd: int = 512
    n_layer: int = 8
    n_head: int = 8
    dropout_rate: float = 0.0
    num_cls_tokens: int = 3
    
    @nn.compact
    def __call__(self, input_ids, attention_mask=None, deterministic=True):
        # Embedding → Positional Encoding → CLS Token Concat
        # → Transformer Layers → Classification Heads
        # Returns: [batch, num_cls_tokens, vocab_size]
        pass
```

**Interface with Existing Code**:
- Input: `input_ids` (JAX array of shape `[batch, seq_len]`)
- Output: `logits` (JAX array of shape `[batch, 3, vocab_size]`)
- Maintains same tokenizer interface (no changes to CustomTokenizer)

### 2. Training Pipeline

**File**: `small_transformer_based/flax_train.py`

**Design Decisions**:
- Use JAX's functional approach: separate model parameters from model definition
- Implement training step as a pure function with `@jax.jit` compilation
- Use Optax for optimizer state management
- Convert PyTorch DataLoader batches to JAX arrays on-the-fly

**Key Functions**:

```python
def create_train_state(rng, model, learning_rate):
    """Initialize model parameters and optimizer state"""
    # Returns: TrainState with params, opt_state, apply_fn
    pass

@jax.jit
def train_step(state, batch, dropout_rng):
    """Single training step (JIT-compiled)"""
    # Compute loss, gradients, update parameters
    # Returns: new_state, loss
    pass

def train_epoch(state, train_loader, epoch, rng):
    """Train for one epoch"""
    # Iterate batches, call train_step, track metrics
    pass

def save_checkpoint(state, path):
    """Save model checkpoint using Flax serialization"""
    pass

def load_checkpoint(path, model):
    """Load model checkpoint"""
    pass
```

**Data Loading Strategy**:
- Keep existing `CustomDataset` and `collate_fn` for tokenization
- Convert PyTorch tensors to JAX arrays: `jnp.array(tensor.numpy())`
- Use `jax.device_put()` for explicit device placement if needed

**Loss Computation**:
```python
def cross_entropy_loss(logits, labels, padding_idx):
    """Compute cross-entropy loss with padding mask"""
    # logits: [batch, num_cls_tokens, vocab_size]
    # labels: [batch, num_cls_tokens]
    # Returns: scalar loss
    pass
```

### 3. Inference Pipeline

**File**: `small_transformer_based/flax_eval.py`

**Design Decisions**:
- Implement batch inference with `@jax.jit` for speed
- Support both standard inference and TTA workflows
- Maintain compatibility with existing `evaluate_true()` interface

**Key Functions**:

```python
@jax.jit
def predict_batch(params, model, input_ids, attention_mask):
    """Batch inference (JIT-compiled)"""
    # Returns: logits [batch, num_cls_tokens, vocab_size]
    pass

def batch_predict_transformations(state, model, tokenizer, device, task_data_list):
    """Predict transformations for multiple tasks"""
    # Tokenize → Batch → Predict → Decode top-k
    pass

def evaluate_with_tta(state, model, tokenizer, task_id, task_file, tta_epochs=15):
    """Perform test-time adaptation for a single task"""
    # Generate synthetic data → Fine-tune → Predict
    pass

def evaluate_true(state, model, tokenizer, tta=True, num_workers=3):
    """Main evaluation function (maintains existing interface)"""
    # Iterate tasks → Predict → Solve with DSL
    pass
```

**TTA Implementation**:
- Clone model state for task-specific fine-tuning
- Generate 2500 synthetic samples using existing `generate_samples()`
- Fine-tune for 15 epochs with Optax optimizer
- Use fine-tuned model for prediction, then discard

### 4. Checkpoint Conversion Utility

**File**: `small_transformer_based/convert_checkpoint.py`

**Design Decisions**:
- Provide one-time conversion from PyTorch `.pth` to Flax checkpoint
- Map PyTorch state dict keys to Flax parameter tree structure
- Handle DataParallel wrapper if present

**Key Function**:

```python
def convert_pytorch_to_flax(pytorch_checkpoint_path, flax_checkpoint_path, model):
    """Convert PyTorch checkpoint to Flax format"""
    # Load PyTorch checkpoint
    # Map parameter names (e.g., 'embedding.weight' → 'Embed_0/embedding')
    # Initialize Flax model with converted parameters
    # Save using Flax serialization
    pass
```

**Parameter Mapping**:
```
PyTorch                          →  Flax
─────────────────────────────────────────────────────────
embedding.weight                 →  Embed_0/embedding
cls_tokens                       →  cls_tokens
transformer_blocks.0.self_attn   →  TransformerEncoder_0/SelfAttention_0
transformer_blocks.0.linear1     →  TransformerEncoder_0/Dense_0
fc_out.weight                    →  Dense_final/kernel
```

### 5. Replacing einops

**Design Decision**: Replace `einops.repeat` with JAX native operations

**Current Usage**:
```python
from einops import repeat
cls_tokens = repeat(self.cls_tokens, "1 n d -> b n d", b=b)
```

**JAX Replacement**:
```python
# Option 1: Using jnp.tile
cls_tokens = jnp.tile(self.cls_tokens, (b, 1, 1))

# Option 2: Using broadcasting + jnp.broadcast_to
cls_tokens = jnp.broadcast_to(self.cls_tokens, (b, self.num_cls_tokens, self.n_embd))
```

**Implementation**: Use `jnp.tile` for clarity and explicit repetition semantics.

## Data Models

### TrainState

```python
from flax.training import train_state

class TrainState(train_state.TrainState):
    """Extended train state with dropout RNG"""
    dropout_rng: jax.random.PRNGKey
```

**Fields**:
- `step`: Current training step
- `apply_fn`: Model's apply function
- `params`: Model parameters (PyTree)
- `tx`: Optax optimizer
- `opt_state`: Optimizer state
- `dropout_rng`: RNG key for dropout

### Checkpoint Format

**Flax Checkpoint Structure**:
```python
{
    'model': {
        'params': {...},  # Nested dict of parameters
    },
    'optimizer': {
        'state': {...},   # Optimizer state
    },
    'step': int,
    'epoch': int,
}
```

**Serialization**: Use `flax.serialization.msgpack_serialize()` for efficient storage.

## Error Handling

### Device Compatibility

**Strategy**: JAX automatically handles device placement, but provide explicit control for edge cases.

```python
# Check available devices
devices = jax.devices()
if jax.devices('gpu'):
    print("Using GPU")
elif jax.devices('cpu'):
    print("Using CPU")

# Explicit device placement if needed
params = jax.device_put(params, jax.devices('gpu')[0])
```

### Memory Management

**Strategy**: Use JAX's memory profiling and gradient checkpointing for large models.

```python
# Enable memory profiling
import jax.profiler
jax.profiler.start_trace("/tmp/jax-trace")
# ... training code ...
jax.profiler.stop_trace()

# Gradient checkpointing (if needed for memory)
from flax.training import checkpoints
# Use remat for activation checkpointing
```

### Numerical Stability

**Strategy**: Use same precision as PyTorch (float32) by default, with option for mixed precision.

```python
# Set default dtype
jax.config.update("jax_default_dtype_bits", "32")

# Optional: Mixed precision training
from jax.experimental import enable_x64
```

## Testing Strategy

### Unit Tests

**File**: `tests/test_flax_model.py`

1. **Model Architecture Test**: Verify parameter count matches 25.3M
2. **Forward Pass Test**: Compare output shapes with PyTorch version
3. **Gradient Test**: Verify gradients are computed correctly
4. **Checkpoint Test**: Test save/load functionality

### Integration Tests

**File**: `tests/test_flax_training.py`

1. **Training Loop Test**: Run 1 epoch on small dataset, verify loss decreases
2. **TTA Test**: Test task-specific fine-tuning workflow
3. **Inference Test**: Compare predictions with PyTorch model on same input

### End-to-End Test

**File**: `tests/test_e2e_workflow.py`

1. **Full Pipeline Test**: Pre-train → TTA → DSL solve on 1 ARC task
2. **Memory Test**: Measure peak memory usage vs PyTorch
3. **Performance Test**: Compare solve rate on 10 ARC tasks

### Validation Criteria

- ✅ Model parameter count: 25.3M ± 0.1M
- ✅ Training loss convergence: Within 5% of PyTorch
- ✅ Solve rate on ARC eval: ≥ 75/400 (same as paper)
- ✅ Memory footprint: < 80% of PyTorch implementation
- ✅ Inference speed: ≥ PyTorch speed

## Migration Strategy

### Phase 1: Model Implementation
1. Implement Flax model architecture
2. Verify parameter count and forward pass
3. Create checkpoint conversion utility

### Phase 2: Training Pipeline
1. Implement training loop with Optax
2. Test on small dataset (overfit single example)
3. Verify loss convergence

### Phase 3: Inference Pipeline
1. Implement batch inference
2. Implement TTA workflow
3. Test on ARC tasks

### Phase 4: Integration
1. Update `eval.py` to use Flax model
2. Update `train.py` to use Flax training
3. Update dependencies in `pyproject.toml`

### Phase 5: Validation
1. Run full training on synthetic dataset
2. Evaluate on ARC train/eval sets
3. Compare solve rates with PyTorch baseline

## Performance Considerations

### Memory Optimization

1. **Batch Size Tuning**: JAX may allow larger batch sizes due to lower overhead
2. **Gradient Accumulation**: Implement if needed for memory-constrained environments
3. **Model Sharding**: Use JAX's `pjit` for multi-device training (future enhancement)

### Speed Optimization

1. **JIT Compilation**: Use `@jax.jit` for all performance-critical functions
2. **XLA Optimization**: Let XLA optimize computation graphs automatically
3. **Batch Processing**: Maximize batch sizes for inference

### Deployment Considerations

1. **Model Size**: Flax checkpoints may be smaller than PyTorch (msgpack vs pickle)
2. **Startup Time**: JAX has faster import time than PyTorch
3. **Agent Integration**: Functional API makes it easier to integrate into agent workflows

## Dependencies Update

### New Dependencies (JAX Ecosystem)

**Add with uv**:
```bash
uv add "jax>=0.7.1"
uv add "flax>=0.11.1,<0.12.0"
uv add "optax>=0.2.5,<0.3.0"
```

**Exact Version Boundaries**:
- `jax>=0.7.1` - Core JAX library for numerical computing and autodiff
- `flax>=0.11.1,<0.12.0` - Neural network library (pin to 0.11.x for stability)
- `optax>=0.2.5,<0.3.0` - Gradient processing and optimization (pin to 0.2.x for stability)

### Dependencies to Remove

**Remove with uv**:
```bash
uv remove torch
uv remove einops
```

**Removed Libraries**:
- `torch` - Replaced by JAX/Flax
- `einops` - Replaced by JAX native operations (jnp.tile, jnp.broadcast_to)

### Dependencies to Keep

**Unchanged**:
- `numpy` - Array operations (JAX is numpy-compatible)
- `scikit-learn` - Train/test split utilities
- `tqdm` - Progress bars
- `pillow` - Image processing
- `networkx` - Graph operations for DSL
- `matplotlib` - Visualization
- `scipy` - Scientific computing

### Final pyproject.toml

```toml
[project]
name = "nsa"
version = "0.1.0"
description = "Neuro-Symbolic ARC Challenge"
readme = "README.md"
requires-python = ">=3.11,<3.12"
dependencies = [
  "jax>=0.7.1",
  "flax>=0.11.1,<0.12.0",
  "optax>=0.2.5,<0.3.0",
  "numpy",
  "scikit-learn",
  "tqdm",
  "pillow",
  "networkx",
  "matplotlib",
  "scipy",
]

[dependency-groups]
dev = [
    "pytest>=8.4.2",
    "pytest-cov>=7.0.0",
    "pyupgrade>=3.21.0",
    "ruff>=0.14.2",
]
```

## Development Workflow

The project uses **uv** for package management and provides Makefile targets for common tasks:

### Setup
```bash
make main          # Create venv and install dependencies
make dev           # Install with dev dependencies
```

### Code Quality
```bash
make format        # Run pyupgrade (py311), ruff format, and ruff fix
make lint          # Report Ruff diagnostics without fixing
```

### Data Generation
```bash
make generate-small  # Generate 100 samples for quick testing
make generate-large  # Generate 10k samples for production
```

### Training
```bash
make train-quick    # Quick training (5 epochs, 4 batch size)
make train          # Full training
```

### Evaluation
```bash
make eval-quick     # Evaluate first 5 tasks
make eval           # Full evaluation
```

### Testing Migration

After implementing JAX/Flax changes:

1. **Format code**: `make format`
2. **Generate test data**: `make generate-small`
3. **Quick training test**: `make train-quick`
4. **Quick evaluation test**: `make eval-quick`
5. **Verify solve rate**: Check results match PyTorch baseline

All Python commands should use `uv run python` for consistency with the project's package management.
