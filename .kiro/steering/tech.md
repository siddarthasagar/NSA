---
inclusion: always
---

## Technology Stack

- **Python:** 3.11+ (strictly `>=3.11,<3.12`)
- **Package manager:** `uv` only (NEVER pip/conda)
- **Python execution:** ALWAYS use `uv run python` (NEVER `python`, `python3`, or direct script execution)
- **ML framework:** JAX/Flax/Optax (migrating from PyTorch)
- **Core deps:** jax, jax-metal, flax, optax, networkx, numpy, einops, scikit-learn
- **Dev tools:** pytest, pytest-cov, ruff (v0.14.2+), pyupgrade

## JAX on Apple Silicon (Mac M4 Mini)

**Device & Memory:**
- Metal backend via `jax-metal` for GPU acceleration
- JAX auto-detects Metal devices - verify with `jax.devices()`
- Unified memory: JAX arrays stay on device by default
- Use `jax.device_put()` only when explicitly needed
- Avoid unnecessary `.block_until_ready()` (forces sync)

**Performance:**
- **JIT:** Always use `@jax.jit` for training loops and forward passes (first call compiles, then fast)
- **Batching:** Optimize for unified memory (128-512 batch size for transformers)
- **Vectorization:** Use `jax.vmap` for batched ops, avoid Python multiprocessing
- **Precision:** Default `float32` for stability, consider `bfloat16` for large models
- Set globally: `jax.config.update('jax_default_matmul_precision', 'high')`

**Debugging:**
- Disable JIT: `with jax.disable_jit():`
- Check NaNs: `jax.config.update('jax_debug_nans', True)`

## Makefile Commands

**Setup:** `make main` (venv + deps), `make dev` (+ dev deps)

**Data:** `make generate-small` (100 samples), `make generate-large` (10k), `make generate-data ARGS='--samples 500'`

**Training:** `make train` (requires full_trans.json), `make train-quick` (5 epochs)

**Quality:** `make format` (run before completion), `make lint` (check only)

**Eval:** `make eval` (full), `make eval-quick` (5 tasks, 3 workers)

**Cleanup:** `make clean` (data/cache), `make clean-all` (+ venv/checkpoints)

## Architecture Patterns

- **Search:** Priority queue with tabu lists to prevent cycles
- **Parallelization:** Multiprocessing for data generation and evaluation
- **Graphs:** NetworkX for abstractions
- **ML:** Flax transformer with sinusoidal positional encoding
- **Optimization:** Optax AdamW with gradient clipping
- **State:** Functional style with immutable pytrees
