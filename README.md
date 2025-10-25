### NSA: Neuro-symbolic ARC Challenge

![Teaser Image](images/teaser.png)

# Abstract
The Abstraction and Reasoning Corpus (ARC) evaluates general reasoning capabilities that are difficult for both machine learning models and combinatorial search methods. 
We propose a neuro-symbolic approach that combines a transformer for proposal generation with combinatorial search using a domain-specific language. 
The transformer narrows the search space by proposing promising search directions, which allows the combinatorial search to find the actual solution in short time.
We pre-train the trainsformer with synthetically generated data.
During test-time we generate additional task-specific training tasks and fine-tune our model. %We generate synthetic tassks to train ourpre-tranin our transformer and finto fine-tune it during test-time.% on taskeOur transformer is trained on synthetic tasks and fine-tuned for each new task via test-time adaptation.
Our results surpass comparable state of the art on the ARC evaluation set by 27\% and compare favourably on the ARC train set.

# Cache Directory Structure

All generated artifacts are organized in a centralized `cache/` directory at the project root:

```
cache/
├── checkpoints/          # Model checkpoints from training
│   └── small_transformer_based/
│       └── results/
│           └── {model_size}M/
│               ├── checkpoint_epoch{N}_final.msgpack
│               └── emergency_epoch{N}_batch{M}.msgpack
├── data/                 # Generated data files and results
│   ├── full_trans.json              # Synthetic training data
│   ├── vocab.json                   # Tokenizer vocabulary
│   ├── dataset_cache.txt            # Dataset cache metadata
│   ├── training_data_summary.json   # Training data statistics
│   ├── arga_evaluation_tta.json     # Evaluation results with TTA
│   ├── arga_training_tta_epoch{N}.json
│   ├── arga_training_no_tta.json
│   ├── arga_evaluation_no_tta.json
│   └── proposed_transformations.txt
├── logs/                 # Training and execution logs
│   └── training_test.log
└── tta/                  # Test-time adaptation artifacts
    └── {task_id}/
        ├── {task_id}.json
        └── generated_samples/
```

**Purpose of each subdirectory:**

- **checkpoints/**: Stores model weights saved during training. Checkpoints are organized by model size (e.g., `2.5M`, `5.0M`) and include both final epoch checkpoints and emergency backups.
- **data/**: Contains all JSON data files including synthetic training data (`full_trans.json`), vocabulary files, evaluation results, and transformation proposals.
- **logs/**: Stores training logs and execution traces for debugging and monitoring.
- **tta/**: Test-time adaptation artifacts generated per task, including task-specific synthetic samples used for fine-tuning.

# Code

Our solution is divided into three parts:

## 1. Data Generation

Generate synthetic training data using parallel transformation sampling:

```bash
# Generate small dataset (100 samples)
make generate-small

# Generate large dataset (10,000 samples)
make generate-large

# Custom generation
make generate-data ARGS='--samples 500 --one_trans_folder cache/generated_samples/one_trans'
```

Output is saved to `cache/data/full_trans.json`.

## 2. Pre-training

Train the transformer model on synthetic data:

```bash
# Full training
make train

# Quick training (5 epochs for testing)
make train-quick
```

Checkpoints are saved to `cache/checkpoints/small_transformer_based/results/{model_size}M/`.

## 3. Evaluation

Evaluate the trained model on ARC tasks:

```bash
# Full evaluation
make eval

# Quick evaluation (5 tasks, 3 workers)
make eval-quick
```

Results are saved to `cache/data/arga_*.json`.

# Testing

The project includes integration tests that verify core workflows:

## Running Tests

```bash
# Run full test suite with coverage
make test

# Run quick integration tests only
make test-quick

# Generate coverage report
make coverage

# Open HTML coverage report in browser
make coverage-html
```

## Test Structure

Tests are organized in the `tests/` directory:

- `test_integration_data.py`: Data generation workflow
- `test_integration_train.py`: Training workflow (minimal epochs)
- `test_integration_eval.py`: Evaluation workflow

## Coverage Analysis

Coverage reports help identify dead code and ensure core functionality is tested:

```bash
# Generate and view coverage report
make coverage

# View detailed HTML report
make coverage-html
```

Coverage reports are generated in `htmlcov/` and show:
- Line-by-line coverage for each module
- Percentage coverage per file
- Untested code paths (potential dead code)

# Development Workflow

## Setup

```bash
# Install dependencies
make main

# Install development dependencies
make dev
```

## Code Quality

```bash
# Format code (run before committing)
make format

# Check linting (without modifying files)
make lint
```

## Cleanup

```bash
# Clean cache and temporary files
make clean

# Full cleanup (including venv and checkpoints)
make clean-all
```

## Common Workflows

**Training a new model:**
```bash
make generate-large  # Generate training data
make train           # Train model
make eval            # Evaluate on ARC tasks
```

**Quick iteration:**
```bash
make generate-small  # Small dataset
make train-quick     # Fast training
make eval-quick      # Quick evaluation
make test-quick      # Verify functionality
```

**Code changes:**
```bash
# Make your changes
make format          # Format code
make lint            # Check for issues
make test            # Run tests
```