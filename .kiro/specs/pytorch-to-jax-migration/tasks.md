# Implementation Plan

- [x] 1. Phase 0: Cleanup and Preparation
  - Remove unused LLM code and update dependencies
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 1.1 Remove llm folder and its references
  - Delete the `llm/` directory and its contents (already removed)
  - Remove import statement from `small_transformer_based/train.py` (already removed)
  - Remove any usage of `generate_selector_prompt` function (already removed)
  - _Requirements: 1.4_

- [x] 1.2 Update project dependencies
  - JAX ecosystem already added to pyproject.toml: jax>=0.7.1, flax>=0.11.1,<0.12.0, optax>=0.2.5,<0.3.0
  - torch and einops already removed from dependencies
  - All PyTorch/einops imports removed from codebase
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 1.3 Run code quality checks
  - Code formatting and linting completed
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 2. Phase 1: Implement Flax Model Architecture
  - Create Flax-based transformer model with same architecture as PyTorch version
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 5.1, 5.2, 5.3, 5.4_

- [x] 2.1 Create sinusoidal positional encoding module
  - Implement `SinusoidalPositionalEncoding` as Flax nn.Module in `small_transformer_based/flax_model.py`
  - Support max_len=6500 and configurable embedding dimension
  - Generate sin/cos position embeddings and add to input
  - _Requirements: 2.8_

- [x] 2.2 Create transformer encoder layer
  - Implement `FlaxTransformerEncoder` as Flax nn.Module
  - Include self-attention with 8 heads, layer normalization, feedforward network (2048 dims), and dropout
  - Support attention masking for padding tokens
  - _Requirements: 2.2, 2.3, 2.4, 2.5_

- [x] 2.3 Create main transformer model
  - Implement `FlaxCustomTransformer` as Flax nn.Module
  - Include embedding layer, positional encoding, 3 learnable CLS tokens, 8 transformer encoder layers, and output projection to vocab size
  - Replace einops.repeat with jnp.tile for CLS token broadcasting
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 5.2_

- [x] 2.4 Verify model parameter count
  - Write utility function to count Flax model parameters
  - Verify total parameters equal approximately 25.3M
  - Print parameter breakdown by layer
  - _Requirements: 2.6_

- [x] 2.5 Run code quality checks
  - Execute `make format` to format new code
  - Execute `make lint` to verify no issues
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

- [x] 3. Phase 2: Implement Training Pipeline
  - Create JAX/Flax/Optax-based training pipeline
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 6.1, 6.2, 6.3, 6.4_

- [x] 3.1 Create train state and initialization
  - Implemented `create_train_state()` function in `small_transformer_based/flax_train.py`
  - Model parameters initialized with random key
  - Optax AdamW optimizer with lr=5e-5 and gradient clipping (max_norm=1.0)
  - TrainState with params, optimizer state, and apply function
  - _Requirements: 3.2, 3.5_

- [x] 3.2 Implement loss computation
  - Implemented `cross_entropy_loss()` function
  - Computes loss for 3 classification tokens
  - Applies padding mask to ignore padding tokens
  - _Requirements: 3.4_

- [x] 3.3 Implement JIT-compiled training step
  - Implemented `train_step()` function with @jax.jit decorator
  - Computes forward pass, loss, and gradients using jax.grad
  - Applies gradient clipping (max_norm=1.0)
  - Updates parameters using optimizer
  - Returns new state and loss value
  - _Requirements: 3.2, 3.3, 3.4, 3.7_

- [x] 3.4 Implement data loading for JAX
  - Implemented `collate_fn_jax()` to convert batches to JAX arrays
  - Pads sequences to uniform length within batch
  - Converts tokenized data to jnp.array format
  - _Requirements: 6.1, 6.2, 6.3_

- [x] 3.5 Implement training loop
  - Implemented `train_epoch()` function
  - Iterates through batches and calls train_step
  - Tracks and logs training metrics (loss, iteration)
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [x] 3.6 Implement checkpoint management
  - Implemented `save_checkpoint()` function using Flax serialization
  - Implemented `load_checkpoint()` function to restore model state
  - Saves checkpoints at specified iteration intervals
  - Stores epoch, iteration, model params, and optimizer state
  - _Requirements: 3.6, 7.2, 7.3_

- [x] 3.7 Create main training function
  - Implemented `main()` function in `small_transformer_based/flax_train.py`
  - Loads data, creates model, initializes train state
  - Runs training loop for specified epochs
  - Supports command-line arguments (data_path, epochs, batch_size, etc.)
  - Auto-detects device (GPU/CPU)
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

- [x] 3.8 Add memory and resource safeguards
  - Add memory monitoring to detect excessive usage
  - Implement batch size auto-adjustment based on available memory
  - Add early stopping if memory usage exceeds safe threshold
  - Add progress checkpointing to allow recovery from crashes
  - Verify JAX memory preallocation settings are appropriate
  - _Requirements: 1.5, 3.7, 3.8_

- [x] 3.9 Test training pipeline with minimal resources
  - Generate small dataset: `make generate-small` (100 samples)
  - Start with very small batch size (2) and short sequences (max_length=512)
  - Run 1 epoch first to verify stability
  - Monitor memory usage throughout
  - If stable, gradually increase to batch_size=4, max_length=1024, 5 epochs
  - Verify loss decreases and checkpoints are saved in .msgpack format
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

- [x] 3.10 Run code quality checks
  - Execute `make format` to format training code
  - Execute `make lint` to verify no issues
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_



- [x] 4. Phase 3: Implement Inference Pipeline
  - Create JAX/Flax-based inference and evaluation pipeline
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

- [x] 4.1 Create flax_eval.py file
  - Create new file `small_transformer_based/flax_eval.py`
  - Import necessary JAX/Flax modules and existing utilities
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 4.2 Implement JIT-compiled batch inference
  - Create `predict_batch()` function with @jax.jit decorator
  - Take model params, input_ids, and attention_mask
  - Return logits for 3 classification tokens
  - _Requirements: 4.3_

- [x] 4.3 Implement top-k prediction extraction
  - Create `extract_first_from_logits()` function
  - Extract top-k predictions from logits for each classification token
  - Decode token IDs to transformation names using tokenizer
  - Filter out "no_trans" and already-added predictions
  - _Requirements: 4.4_

- [x] 4.4 Implement batch transformation prediction
  - Create `batch_predict_transformations()` function
  - Process multiple tasks in parallel
  - Tokenize inputs, create batches, run inference
  - Extract and return top-k transformation predictions per task
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 4.5 Implement test-time adaptation (TTA)
  - Create `evaluate_with_tta()` function
  - Generate 2500 task-specific synthetic samples
  - Fine-tune model for 15 epochs on task-specific data
  - Use fine-tuned model for prediction
  - _Requirements: 4.6, 9.2, 9.3_

- [x] 4.6 Implement main evaluation function
  - Create `evaluate_true()` function maintaining existing interface
  - Support both TTA and non-TTA modes
  - Iterate through train/val splits
  - Call batch_predict_transformations or evaluate_with_tta
  - Pass predictions to Task.solve() for DSL search
  - Track and save results to JSON
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 9.4, 9.5, 9.6_

- [x] 4.7 Create evaluation script entry point
  - Implement `main()` function in `small_transformer_based/flax_eval.py`
  - Load tokenizer and model checkpoint
  - Support command-line arguments (--tta, --num-workers, --max-tasks)
  - Call evaluate_true with appropriate parameters
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

- [x] 4.8 Run code quality checks
  - Execute `make format` to format evaluation code
  - Execute `make lint` to verify no issues
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

- [x] 5. Phase 4: Integration and Backward Compatibility
  - Integrate Flax implementation and ensure compatibility
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 8.1, 8.2, 8.3, 8.4_

- [x] 5.1 Create PyTorch to Flax checkpoint converter (optional)
  - Create `small_transformer_based/convert_checkpoint.py` if needed
  - Implement `convert_pytorch_to_flax()` function
  - Load PyTorch checkpoint and extract state_dict
  - Map PyTorch parameter names to Flax parameter tree structure
  - Handle DataParallel wrapper if present
  - Initialize Flax model with converted parameters
  - Save using Flax serialization
  - Note: Only needed if existing PyTorch checkpoints need to be converted
  - _Requirements: 7.1, 7.2_

- [x] 5.2 Update Makefile for Flax training
  - Update `train` target to use `small_transformer_based.flax_train` instead of `small_transformer_based.train`
  - Update `train-quick` target to use Flax training script
  - Ensure all command-line arguments are compatible
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 5.3 Update Makefile for Flax evaluation
  - Update `eval` target to use `small_transformer_based.flax_eval` instead of `small_transformer_based.eval`
  - Update `eval-quick` target to use Flax evaluation script
  - Ensure all command-line arguments are compatible
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 5.4 Verify tokenizer compatibility
  - Test that CustomTokenizer works with JAX arrays
  - Verify vocab.json loading works correctly
  - Verify encoding/decoding produces same results
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 5.5 Run code quality checks
  - Execute `make format` to format all updated code
  - Execute `make lint` to verify no issues
  - Fix any remaining linting errors
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 6. Phase 5: Validation and Testing
  - Validate the complete migration and measure improvements
  - _Requirements: 1.5, 9.6_

- [x] 6.1 Generate test dataset
  - Run `make generate-small` to create 100 sample dataset
  - Verify data generation completes successfully
  - _Requirements: 9.1_

- [x] 6.2 Run quick training test
  - Execute `make train-quick` (5 epochs, small batch)
  - Verify training completes without errors
  - Verify loss decreases over epochs
  - Verify checkpoints are saved correctly in .msgpack format
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

- [x] 6.3 Run quick evaluation test
  - Execute `make eval-quick` (first 5 tasks)
  - Verify inference completes without errors
  - Verify predictions are passed to DSL search
  - Verify Task.solve() executes correctly
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 9.4, 9.5_

- [x] 6.4 Measure memory footprint
  - Profile memory usage during training with JAX
  - Profile memory usage during inference with JAX
  - Compare with PyTorch baseline if available
  - Document memory usage
  - _Requirements: 1.5_

- [x] 6.5 Compare solve rates
  - Run evaluation on larger task set (e.g., 20-50 tasks)
  - Document solve rate with JAX implementation
  - Compare with PyTorch baseline if available
  - Document any performance differences
  - _Requirements: 9.6_

- [x] 6.6 Final code quality check
  - Execute `make format` on entire codebase
  - Execute `make lint` and ensure no errors
  - Review all code changes for consistency
  - Update README.md if needed to reflect JAX/Flax usage
  - _Requirements: 1.1, 1.2, 1.3, 1.4_
