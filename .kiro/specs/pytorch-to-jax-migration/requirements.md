# Requirements Document

## Introduction

This document outlines the requirements for migrating the NSA (Neuro-Symbolic ARC Challenge) project from PyTorch-based deep learning infrastructure to JAX/Flax/Optax. The migration aims to reduce memory footprint and framework overhead by replacing PyTorch and einops with the lighter JAX ecosystem, while preserving the paper's core three-phase methodology: (1) pre-training on synthetic data, (2) test-time adaptation (TTA) for task-specific fine-tuning, and (3) combinatorial search using the ARGA DSL. This enables deployment in resource-constrained agent environments for ARC-AGI testing.

## Glossary

- **NSA System**: The Neuro-Symbolic ARC Challenge implementation combining transformer-based proposal generation with combinatorial search
- **Transformer Model**: The neural network component (CustomTransformer) that proposes transformation primitives
- **Training Pipeline**: The data loading, model training, and checkpoint saving workflow
- **Inference Pipeline**: The model evaluation and prediction workflow for ARC tasks
- **JAX**: Google's high-performance numerical computing library with automatic differentiation
- **Flax**: Neural network library built on JAX using functional programming patterns
- **Optax**: Gradient processing and optimization library for JAX
- **PyTorch**: The current deep learning framework being replaced
- **einops**: Tensor operation library being replaced with JAX native operations
- **TTA**: Test-Time Adaptation - fine-tuning the model on task-specific synthetic data

## Requirements

### Requirement 1

**User Story:** As an agent developer, I want to replace PyTorch dependencies with JAX/Flax/Optax, so that the system has a smaller memory footprint and reduced framework overhead for deployment in resource-constrained environments

#### Acceptance Criteria

1. WHEN the project dependencies are updated, THE NSA System SHALL use optax>=0.2.5,<0.3.0 for optimization
2. WHEN the project dependencies are updated, THE NSA System SHALL use flax>=0.11.1,<0.12.0 for neural network definitions
3. WHEN the project dependencies are updated, THE NSA System SHALL use jax>=0.7.1 for numerical computations
4. WHEN the project dependencies are updated, THE NSA System SHALL remove torch and einops from the dependency list
5. WHEN the system is initialized, THE NSA System SHALL have a smaller memory footprint compared to the PyTorch implementation
6. WHERE GPU acceleration is available, THE NSA System SHALL utilize JAX's GPU support without requiring code changes

### Requirement 2

**User Story:** As a developer, I want the transformer model architecture migrated to Flax, so that the model maintains the same structure and parameter count (25.3M parameters)

#### Acceptance Criteria

1. WHEN the CustomTransformer class is migrated, THE Transformer Model SHALL be implemented as a Flax nn.Module
2. WHEN the model is initialized, THE Transformer Model SHALL have 8 transformer encoder layers
3. WHEN the model is initialized, THE Transformer Model SHALL have 512 embedding dimensions
4. WHEN the model is initialized, THE Transformer Model SHALL have 2048 feedforward dimensions
5. WHEN the model is initialized, THE Transformer Model SHALL have 8 attention heads
6. WHEN the model parameters are counted, THE Transformer Model SHALL contain approximately 25.3 million trainable parameters
7. WHEN the model processes input, THE Transformer Model SHALL support 3 classification tokens for multi-output prediction
8. WHEN the model processes input, THE Transformer Model SHALL apply sinusoidal positional encoding to input embeddings

### Requirement 3

**User Story:** As a developer, I want the training pipeline migrated to use JAX/Flax/Optax, so that model training produces equivalent results to the PyTorch implementation

#### Acceptance Criteria

1. WHEN training data is loaded, THE Training Pipeline SHALL tokenize and batch input-output pairs using the CustomTokenizer
2. WHEN the model is trained, THE Training Pipeline SHALL use Optax AdamW optimizer with learning rate 5e-5
3. WHEN the model is trained, THE Training Pipeline SHALL apply gradient clipping with max norm 1.0
4. WHEN the model is trained, THE Training Pipeline SHALL use CrossEntropyLoss for classification
5. WHEN the model is trained, THE Training Pipeline SHALL apply learning rate scheduling with StepLR (step_size=10, gamma=0.1)
6. WHEN training progresses, THE Training Pipeline SHALL save model checkpoints at specified iteration intervals
7. WHEN training completes an epoch, THE Training Pipeline SHALL update the learning rate according to the scheduler
8. WHERE GPU memory is available, THE Training Pipeline SHALL automatically select appropriate batch sizes (16 for unified memory architectures, 32 for discrete GPUs)

### Requirement 4

**User Story:** As a developer, I want the inference pipeline migrated to JAX/Flax, so that the model can predict transformation primitives for ARC tasks

#### Acceptance Criteria

1. WHEN a trained model checkpoint is loaded, THE Inference Pipeline SHALL restore model parameters from the checkpoint file
2. WHEN inference is performed, THE Inference Pipeline SHALL process input grids through the tokenizer
3. WHEN inference is performed, THE Inference Pipeline SHALL generate logits for 3 classification tokens
4. WHEN predictions are extracted, THE Inference Pipeline SHALL decode the top-k predictions from logits
5. WHEN batch inference is performed, THE Inference Pipeline SHALL process multiple tasks in parallel
6. WHEN TTA is enabled, THE Inference Pipeline SHALL fine-tune the model on task-specific synthetic data for 15 epochs
7. WHERE attention masking is required, THE Inference Pipeline SHALL apply padding masks to prevent attention to padding tokens

### Requirement 5

**User Story:** As a developer, I want einops operations replaced with JAX native operations, so that the codebase has no dependency on einops

#### Acceptance Criteria

1. WHEN tensor operations are performed, THE NSA System SHALL use jax.numpy for array manipulations
2. WHEN the repeat operation is needed, THE NSA System SHALL use JAX broadcasting or jnp.tile instead of einops.repeat
3. WHEN tensor reshaping is needed, THE NSA System SHALL use jnp.reshape or jnp.transpose instead of einops operations
4. WHEN the code is executed, THE NSA System SHALL not import or reference the einops library

### Requirement 6

**User Story:** As a developer, I want data loading to work with JAX arrays, so that training and inference pipelines can consume data efficiently

#### Acceptance Criteria

1. WHEN data is loaded, THE Training Pipeline SHALL convert tokenized sequences to JAX arrays
2. WHEN batches are created, THE Training Pipeline SHALL pad sequences to uniform length within each batch
3. WHEN data is transferred to accelerators, THE Training Pipeline SHALL handle device placement automatically via JAX
4. WHERE multi-worker data loading is used, THE Training Pipeline SHALL maintain compatibility with Python multiprocessing

### Requirement 7

**User Story:** As a researcher, I want checkpoint compatibility maintained, so that I can convert existing PyTorch checkpoints to JAX format

#### Acceptance Criteria

1. WHEN a PyTorch checkpoint exists, THE NSA System SHALL provide a conversion utility to transform weights to Flax format
2. WHEN checkpoints are saved, THE Training Pipeline SHALL store model parameters in a format compatible with Flax serialization
3. WHEN checkpoints are loaded, THE Inference Pipeline SHALL restore the exact model state including optimizer state for TTA
4. WHERE checkpoint paths are specified, THE NSA System SHALL maintain the same directory structure as the PyTorch implementation

### Requirement 8

**User Story:** As a developer, I want the CustomTokenizer to remain unchanged, so that existing vocabulary and tokenization logic continues to work

#### Acceptance Criteria

1. WHEN text is tokenized, THE NSA System SHALL use the existing CustomTokenizer implementation without modifications
2. WHEN vocabulary is built, THE NSA System SHALL maintain compatibility with existing vocab.json files
3. WHEN tokens are encoded, THE NSA System SHALL produce the same token IDs as the PyTorch implementation
4. WHEN tokens are decoded, THE NSA System SHALL produce the same text output as the PyTorch implementation

### Requirement 9

**User Story:** As an agent developer, I want the three-phase workflow preserved (pre-training, TTA, DSL search), so that the system maintains the paper's core methodology for ARC-AGI problem solving

#### Acceptance Criteria

1. WHEN the system is used for ARC-AGI testing, THE NSA System SHALL support pre-training the transformer on synthetic data generated via hindsight relabeling
2. WHEN a new ARC task is encountered, THE NSA System SHALL generate 2500 task-specific synthetic samples for test-time adaptation
3. WHEN TTA is performed, THE NSA System SHALL fine-tune the pre-trained model for 15 epochs on task-specific data
4. WHEN the model produces predictions, THE NSA System SHALL pass the top-k predicted transformations to the ARGA DSL combinatorial search
5. WHEN the DSL search executes, THE NSA System SHALL use the existing Task.solve() method with proposed transformations
6. WHEN the complete workflow executes, THE NSA System SHALL maintain the same solve rate as the PyTorch implementation on ARC evaluation tasks
