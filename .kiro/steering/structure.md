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

**extended_transformations/** - Grid-level ops (crop, fill, connect, upscale, rotate, mirror, shift, truncate, recolor, magnet, beam, duplicate)

**auxilaries/** - Data generation
- `generate_transformation.py`: Parallel synthetic data generation
- `grid_transformation.py`: Core transformation sampling

**small_transformer_based/** - ML pipeline
- `flax_train.py`: JAX/Flax transformer training (current)
- `train.py`: PyTorch training (deprecated)
- `flax_eval.py`, `eval.py`: Model evaluation

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
- `full_trans.json`: Synthetic training data
- `images/`: Task visualizations
- `small_transformer_based/results/`: Model checkpoints
