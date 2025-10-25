# NSA: Neuro-symbolic ARC Challenge - Training Pipeline
# ======================================================

.PHONY: main add-deps dev train generate-data generate-large eval help format lint test test-quick coverage coverage-html

# Default target - show help
help:
	@echo "NSA Training Pipeline Commands:"
	@echo "================================"
	@echo ""
	@echo "Setup:"
	@echo "  make main          - Create virtual environment and install dependencies"
	@echo "  make dev           - Install with dev dependencies (pytest, etc.)"
	@echo ""
	@echo "Data Generation:"
	@echo "  make generate-data - Generate training samples (pass ARGS for custom settings)"
	@echo "  make generate-small- Quick test with 100 samples (1-step chains)"
	@echo "  make generate-large- Production dataset: 10k samples, both 1-step and 2-step"
	@echo ""
	@echo "Training:"
	@echo "  make train         - Train model (requires generated data in cache/data/full_trans.json)"
	@echo "  make train-quick   - Quick training test (5 epochs, useful for debugging)"
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run full test suite with coverage"
	@echo "  make test-quick    - Run fast integration tests only"
	@echo "  make coverage      - Generate coverage report (HTML + terminal)"
	@echo "  make coverage-html - Generate and open HTML coverage report"
	@echo ""
	@echo "Code quality:"
	@echo "  make format        - Run pyupgrade (py311), ruff format, and ruff fix"
	@echo "  make lint          - Report Ruff diagnostics without fixing (fails on issues)"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval          - Evaluate trained model on test set"
	@echo "  make eval-quick    - Quick evaluation test (first 5 tasks, 3 workers)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Remove cache data and test artifacts"
	@echo "  make clean-all     - Remove everything including venv and checkpoints"
	@echo ""
	@echo "Examples:"
	@echo "  make generate-data ARGS='--samples 500 --transformations one --timeout 3.0'"
	@echo "  make train ARGS='--data_path cache/data/full_trans.json --save_iterations 50'"
	@echo ""
	@echo "Pipeline workflow:"
	@echo "  1. make main                  # Setup environment"
	@echo "  2. make generate-large        # Generate training data"
	@echo "  3. make train                 # Train model"
	@echo "  4. make eval                  # Evaluate"

# Environment setup
main:
	uv venv --clear
	uv sync

add-deps:
	uv add -r requirements.txt

dev:
	uv venv --clear
	uv sync --dev

# Training
train:
	@echo "Training model with JAX/Flax from cache/data/full_trans.json..."
	@if [ ! -f cache/data/full_trans.json ]; then \
		echo "ERROR: cache/data/full_trans.json not found. Run 'make generate-data' first."; \
		exit 1; \
	fi
	@samples=$$(cat cache/data/full_trans.json | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0"); \
	if [ "$$samples" -lt 100 ]; then \
		echo "WARNING: Only $$samples training samples found. Recommended: 10,000+"; \
		echo "Run 'make generate-large' to generate more data."; \
	else \
		echo "Found $$samples training samples. Proceeding..."; \
	fi
	uv run python -m small_transformer_based.flax_train $(ARGS)

train-quick:
	@echo "Quick training run (5 epochs) for testing with JAX/Flax..."
	$(MAKE) train ARGS="--data_path cache/data/full_trans.json --save_iterations 10 --print_iterations 5 --epochs 5 --batch_size 4 --max_length 1024"

# Data generation
generate-data:
	@echo "Generating synthetic training data..."
	uv run python -m auxilaries.generate_transformation $(ARGS)
	@echo "Data generation complete. Check cache/data/full_trans.json for samples."

generate-small:
	@echo "Generating small test dataset (100 samples, 1-step transformations)..."
	@cores=$$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo "4"); \
	workers=$$((cores > 1 ? cores - 1 : 1)); \
	$(MAKE) generate-data ARGS="--samples 100 --transformations one --timeout 2.0 --workers $$workers"
	@echo "Generated samples for quick testing."

generate-large:
	@echo "Generating large production dataset (10,000 samples, 1-step and 2-step)..."
	@cores=$$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo "4"); \
	workers=$$((cores > 1 ? cores - 1 : 1)); \
	echo "Detected $$cores CPU cores, using $$workers parallel workers"; \
	echo "This will take ~10-15 minutes with parallel processing. Progress bars will show status."; \
	$(MAKE) generate-data ARGS="--samples 10000 --transformations both --timeout 3.0 --workers $$workers"
	@echo "Large dataset generation complete!"

# Evaluation
eval:
	@echo "Evaluating trained model with JAX/Flax..."
	@if [ ! -d "cache/checkpoints/small_transformer_based/results" ]; then \
		echo "ERROR: No trained model found in cache/checkpoints/small_transformer_based/results/"; \
		echo "Run 'make train' first."; \
		exit 1; \
	fi
	uv run python -m small_transformer_based.flax_eval $(ARGS)

eval-quick:
	@echo "Quick evaluation test (first 5 tasks, 3 workers)..."
	$(MAKE) eval ARGS="--max-tasks 5 --num-workers 3"

# Testing
test:
	@echo "Running full test suite with coverage..."
	uv run pytest tests/ --cov=. --cov-report=html --cov-report=term

test-quick:
	@echo "Running quick integration tests..."
	uv run pytest tests/ -v -k "minimal"

coverage:
	@echo "Generating coverage report..."
	uv run pytest tests/ --cov=. --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

coverage-html:
	@echo "Opening coverage report in browser..."
	$(MAKE) coverage
	@if command -v open >/dev/null 2>&1; then \
		open htmlcov/index.html; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open htmlcov/index.html; \
	else \
		echo "Please open htmlcov/index.html manually in your browser"; \
	fi

# Cleanup
clean:
	@echo "Cleaning cache data and test artifacts..."
	rm -rf cache/generated_samples/
	rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	rm -rf cache/data/*.json cache/data/*.txt
	rm -rf cache/tta/
	rm -rf htmlcov/ .coverage .pytest_cache/
	@echo "Cleanup complete."

clean-all: clean
	@echo "Removing virtual environment, checkpoints, and entire cache..."
	rm -rf .venv/
	rm -rf cache/checkpoints/small_transformer_based/results/
	rm -rf cache/
	@echo "Full cleanup complete."

# Code quality and formatting
format:
	@echo "Running pyupgrade across the repo (py311)..."
	@# Apply to all Python files recursively; exit zero even if files changed
	find . -type f -name "*.py" -not -path "*/.venv/*" -print0 | xargs -0 -n 25 uv run pyupgrade --py311 --exit-zero-even-if-changed || true
	@echo "Applying Ruff formatting..."
	uv run ruff format . || true
	@echo "Applying Ruff autofixes..."
	uv run ruff check --fix . || true
	@echo "Format complete. Use 'make lint' to see remaining issues."

lint:
	@echo "Running Ruff lint (no fixes)..."
	uv run ruff check .
