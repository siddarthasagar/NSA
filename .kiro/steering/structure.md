---
inclusion: always
---

## Core Modules

**main.py** - CLI entry point
- Usage: `python -m main <task_file.json> <task_type> [time_limit] [save_images]`
- Runs task solving in separate process with timeout

**task.py** - Search orchestration
- Loads train/test pairs from JSON
- Manages graph abstractions (nbccg, ccg, mcccg, na, lrg)
- Priority queue search with constraint acquisition
- Returns transformation sequence solution

**ARCGraph.py** - Graph operations
- 40+ transformations (extract, move, rotate, mirror, crop, etc.)
- Filter operations for node selection
- Parameter binding for dynamic arguments
- Grid ↔ graph conversion

**image.py** - Abstraction layer
- Converts grids to graph representations
- Abstractions: nbccg, nbvcg, nbhcg, ccgbr, ccgbr2, ccg, mcccg, lrg, na
- Background color detection and grid reconstruction

## Directory Structure

**cache/** - Centralized output directory for all generated artifacts
- `checkpoints/` - Model checkpoints organized by size (e.g., `small_transformer_based/results/3.2M/`)
- `data/` - JSON data files (vocab.json, full_trans.json, dataset_cache.txt, evaluation results)
- `logs/` - Training and evaluation logs
- `tta/` - Test-time adaptation data organized by task ID
- `generated_samples/` - Synthetic data from generate_transformation.py (one_trans/, two_trans/)

**learning/** - Reports, observations, and analysis documents
- All markdown reports and observation files go here
- Examples: dead_code_candidates.md, performance_analysis.md, experiment_results.md

**extended_transformations/** - Grid-level ops (crop, fill, connect, upscale, rotate, mirror, shift, truncate, recolor, magnet, beam, duplicate)

**auxilaries/** - Data generation
- `generate_transformation.py`: Parallel synthetic data generation
- `grid_transformation.py`: Core transformation sampling

**small_transformer_based/** - ML pipeline
- `flax_train.py`: JAX/Flax transformer training (current)
- `train.py`: PyTorch training (deprecated)
- `flax_eval.py`, `eval.py`: Model evaluation

**tests/** - Test suite
- `conftest.py`: Shared fixtures and test configuration
- `test_integration_data.py`: Data generation workflow tests
- `test_integration_train.py`: Training workflow tests
- `test_integration_eval.py`: Evaluation workflow tests

**dataset/** - ARC benchmark
- `training/`: 400 tasks
- `evaluation/`: 400 tasks
- Format: `{"train": [{"input": [[...]], "output": [[...]]}], "test": [...]}`

## Abstraction Types

- **nbccg:** Neighbor-based connected components (most common)
- **ccg:** Connected components by color
- **mcccg:** Multicolor connected components
- **na:** No abstraction (grid-level operations)
- **lrg:** Largest rectangular grid

## Search Strategy

1. Initialize frontier with all abstractions
2. Expand nodes via priority queue (train error + depth)
3. Apply constraint acquisition to prune invalid ops
4. Use tabu lists to prevent cycles
5. Track best solution across abstractions

## Data Formats

**Transformation sequence:**
```python
[
  {"transformation": ["extract"], "input": {...}},
  {"transformation": ["move_node"], "input": {...}}
]
```

**Generated artifacts:**
- `cache/data/full_trans.json`: Synthetic training data
- `cache/checkpoints/small_transformer_based/results/`: Model checkpoints
- `cache/data/vocab.json`: Tokenizer vocabulary
- `cache/logs/training_test.log`: Training logs
- `images/`: Task visualizations (not in cache)

## PathConfig Usage Patterns

**Importing:**
```python
from utils import PathConfig
```

**Common patterns:**
```python
# Ensure cache directories exist (call once at module init)
PathConfig.ensure_cache_dirs()

# Get checkpoint directory for a model size
checkpoint_dir = PathConfig.get_checkpoint_dir("3.2M")
# Returns: "cache/checkpoints/small_transformer_based/results/3.2M"

# Get checkpoint file path
checkpoint_path = PathConfig.get_checkpoint_path("3.2M", epoch=10, final=True)
# Returns: "cache/checkpoints/small_transformer_based/results/3.2M/checkpoint_epoch10_final.msgpack"

# Get data file path
vocab_path = PathConfig.get_data_path("vocab.json")
# Returns: "cache/data/vocab.json"

full_trans_path = PathConfig.get_data_path("full_trans.json")
# Returns: "cache/data/full_trans.json"

# Get log file path
log_path = PathConfig.get_log_path("training_test.log")
# Returns: "cache/logs/training_test.log"

# Get TTA directory for a task
tta_dir = PathConfig.get_tta_dir("00576224")
# Returns: "cache/tta/00576224"
```

**When to use PathConfig:**
- Always use for checkpoints, data files, logs, and TTA directories
- Never hardcode paths like "small_transformer_based/results/" or "vocab.json"
- Use PathConfig methods in training, evaluation, and data generation modules
- PathConfig automatically creates directories as needed
