---
inclusion: always
---

## Product Overview

NSA (Neuro-Symbolic ARC) solves Abstraction and Reasoning Corpus (ARC) benchmark tasks using a hybrid neuro-symbolic approach:

1. **Transformer proposal generation** - Small transformer trained on synthetic data proposes transformation sequences
2. **Combinatorial search with DSL** - Domain-specific language of graph transformations explores search space guided by proposals

## Key Components

- **Graph abstractions:** Input grids → graph representations (neighbor-based, connected components, etc.)
- **Transformation DSL:** 40+ operations (extract, move, rotate, mirror, crop, fill, upscale, etc.)
- **Constraint acquisition:** Prunes invalid transformations based on training examples
- **Test-time adaptation:** Fine-tunes transformer on task-specific synthetic data

## Dataset

- **Training:** 400 tasks in `dataset/training/`
- **Evaluation:** 400 tasks in `dataset/evaluation/`
- **Format:** Each task has input-output grid pairs demonstrating a pattern to learn
