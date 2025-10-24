# NSA: Neuro-symbolic ARC Challenge - Training Pipeline
# ======================================================

.PHONY: main add-deps dev train generate-data generate-large eval help

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
	@echo "  make train         - Train model (requires generated data in full_trans.json)"
	@echo "  make train-quick   - Quick training test (5 epochs, useful for debugging)"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval          - Evaluate trained model on test set"
	@echo ""
	@echo "Examples:"
	@echo "  make generate-data ARGS='--samples 500 --transformations one --timeout 3.0'"
	@echo "  make train ARGS='--data_path full_trans.json --save_iterations 50'"
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
	@echo "Training model with data from full_trans.json..."
	@if [ ! -f full_trans.json ]; then \
		echo "ERROR: full_trans.json not found. Run 'make generate-data' first."; \
		exit 1; \
	fi
	@samples=$$(cat full_trans.json | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0"); \
	if [ "$$samples" -lt 100 ]; then \
		echo "WARNING: Only $$samples training samples found. Recommended: 10,000+"; \
		echo "Run 'make generate-large' to generate more data."; \
	else \
		echo "Found $$samples training samples. Proceeding..."; \
	fi
	uv run python -m small_transformer_based.train $(ARGS)

train-quick:
	@echo "Quick training run (5 epochs) for testing..."
	$(MAKE) train ARGS="--data_path full_trans.json --save_iterations 10 --print_iterations 5"

# Data generation
generate-data:
	@echo "Generating synthetic training data..."
	uv run python -m auxilaries.generate_transformation $(ARGS)
	@echo "Data generation complete. Check full_trans.json for samples."

generate-small:
	@echo "Generating small test dataset (100 samples, 1-step transformations)..."
	$(MAKE) generate-data ARGS="--samples 100 --transformations one --timeout 2.0"
	@echo "Generated samples for quick testing."

generate-large:
	@echo "Generating large production dataset (10,000 samples, 1-step and 2-step)..."
	@echo "This will take ~30-60 minutes. Progress bars will show status."
	$(MAKE) generate-data ARGS="--samples 10000 --transformations both --timeout 3.0"
	@echo "Large dataset generation complete!"

# Evaluation
eval:
	@echo "Evaluating trained model..."
	@if [ ! -d "small_transformer_based/results" ]; then \
		echo "ERROR: No trained model found in small_transformer_based/results/"; \
		echo "Run 'make train' first."; \
		exit 1; \
	fi
	uv run python -m small_transformer_based.eval $(ARGS)

# Cleanup
clean:
	@echo "Cleaning generated data and cache files..."
	rm -rf final_data8/ generated_llm_data_two_trans1/
	rm -f full_trans.json
	rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	rm -f vocab.json dataset_cache.txt training_data_summary.json
	@echo "Cleanup complete."

clean-all: clean
	@echo "Removing virtual environment and checkpoints..."
	rm -rf .venv/
	rm -rf small_transformer_based/results/
	@echo "Full cleanup complete."
