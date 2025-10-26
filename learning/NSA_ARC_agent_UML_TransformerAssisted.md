# ARC Agent — Transformer-assisted mode (Transformer + TTA)

This document contains the Mermaid UML diagrams for the Final ARC agent's Transformer‑assisted operation mode. It includes the focused class-level overview and the sequence diagram that shows runtime flow for using ML proposals and optional Test‑Time Adaptation (TTA) together with the symbolic search.

---

## Transformer-assisted path — Class-level overview (transformer + TTA)

```mermaid
classDiagram
    %% Focused classes for transformer-assisted flow
    classDef transform fill:#E8F5E9,stroke:#43A047,stroke-width:1px;
    classDef storage fill:#F3E5F5,stroke:#8E24AA,stroke-width:1px;
    classDef processing fill:#FFF3E0,stroke:#FB8C00,stroke-width:1px;
    classDef training fill:#E3F2FD,stroke:#1E88E5,stroke-width:1px;
    classDef evaluation fill:#FFF8E1,stroke:#FFA000,stroke-width:1px;

    class ModelArchitecture {
        +FlaxCustomTransformer(vocab_size, n_embd=512, n_layer=8)\n(*flax_model.py*)
        +SinusoidalPositionalEncoding(n_embd, max_len=6500)\n(*flax_model.py*)
        +FlaxTransformerEncoder(n_embd, n_head=8)\n(*flax_model.py*)
        +__call__(input_ids, attention_mask, deterministic)\n(*flax_model.py*)
        +count_parameters(params)\n(*flax_model.py*)
    }

    class TrainingPipeline {
        +CustomTokenizer(special_tokens, transformations)\n(*flax_train.py*)
        +CustomDataset(data, tokenizer, max_length)\n(*flax_train.py*)
        +create_train_state(rng, model, learning_rate)\n(*flax_train.py*)
        +train_step(state, batch, padding_idx)\n(*flax_train.py*)
        +train_epoch(state, train_data, tokenizer)\n(*flax_train.py*)
        +save_checkpoint(state, path, epoch)\n(*flax_train.py*)
        +balance_transformations(data)\n(*flax_train.py*)
    }

    class EvaluationPipeline {
        +create_predict_fn(model)\n(*flax_eval.py*)
        +batch_predict_transformations(state, model, tokenizer)\n(*flax_eval.py*)
        +evaluate_with_tta(state, model, tokenizer, task_id)\n(*flax_eval.py*)
        +extract_first_from_logits(tokenizer, logits, index)\n(*flax_eval.py*)
        +format_grid_for_tokenizer(grid_data)\n(*flax_eval.py*)
        +evaluate_true(state, model, tokenizer, tta)\n(*flax_eval.py*)
    }

    class DataGeneration {
        +generate_samples(number_of_samples, output_folder)\n(*generate_transformation.py*)
        +_generate_single_sample(args)\n(*generate_transformation.py*)
        +sample_and_apply(no_of_trans, transformation_ops)\n(*grid_transformation.py*)
        +prepare_and_save_transformed_data(...)\n(*grid_transformation.py*)
    }

    class Task_TA {
        +__init__(filepath, proposed_transformations=None)\n(*task.py*)
        +solve(shared_frontier=True, time_limit=1800)\n(*task.py*)
        +initialize_frontier(candidate_transformations=proposals)\n(*task.py*)
        +get_candidate_transformations(filters)\n(*task.py*)
        +expand_frontier(frontier_node)\n(*task.py*)
    }

    class ConstraintAcquisition_TA {
        +constraints_acquisition_global()\n(*task.py*)
        +constraints_acquisition_local(apply_filter_call)\n(*task.py*)
        +prune_transformations(constraints)\n(*task.py*)
    }

    class Cache_A {
        +checkpoints/: model checkpoints\n(*cache/checkpoints/*)
        +data/: vocab.json, full_trans.json\n(*cache/data/*)
        +tta/: task-specific synthetic data\n(*cache/tta/*)
        +logs/: training logs\n(*cache/logs/*)
    }

    class Utils_A {
        +PathConfig.get_checkpoint_dir()\n(*utils.py*)
        +PathConfig.get_tta_dir(task_id)\n(*utils.py*)
        +return_task_grid(task_file)\n(*plots.py*)
    }

    class ModelArchitecture transform
    class TrainingPipeline training
    class EvaluationPipeline evaluation
    class DataGeneration storage
    class Task_TA processing
    class ConstraintAcquisition_TA processing
    class Cache_A storage
    class Utils_A util

    TrainingPipeline --> ModelArchitecture : trains model
    TrainingPipeline --> Cache_A : saves checkpoints & vocab
    EvaluationPipeline --> ModelArchitecture : loads & runs inference
    EvaluationPipeline --> Cache_A : loads checkpoints
    EvaluationPipeline --> DataGeneration : triggers TTA data generation
    DataGeneration --> Cache_A : saves TTA samples
    Task_TA --> EvaluationPipeline : receives proposals
    Task_TA --> ConstraintAcquisition_TA : prunes search with proposals
    Utils_A --> Cache_A : manages paths & data access

```

### Key Methods by Component

**ModelArchitecture (flax_model.py):**
- `FlaxCustomTransformer.__call__()`, `SinusoidalPositionalEncoding.__call__()`
- `FlaxTransformerEncoder.__call__()`, `count_parameters()`

**TrainingPipeline (flax_train.py):**
- `CustomTokenizer`: `build_vocab()`, `tokenize()`, `encode()`, `decode()`
- `CustomDataset.__getitem__()`, `create_train_state()`, `train_step()` (JIT-compiled)
- `train_epoch()`, `save_checkpoint()`, `load_checkpoint()`, `balance_transformations()`

**EvaluationPipeline (flax_eval.py):**
- `create_predict_fn()` (JIT-compiled), `batch_predict_transformations()`
- `evaluate_with_tta()`, `extract_first_from_logits()`, `format_grid_for_tokenizer()`
- `evaluate_true()`, `_solve_single_task()` (parallel worker)

**DataGeneration (generate_transformation.py):**
- `generate_samples()`, `_generate_single_sample()` (parallel worker)
- `sample_and_apply()`, `prepare_and_save_transformed_data()`

**Task Integration (task.py):**
- `Task.__init__(proposed_transformations)`, `initialize_frontier(candidate_transformations)`
- `get_candidate_transformations()` (filtered by proposals)

**Cache Management (utils.py):**
- `PathConfig.get_checkpoint_dir()`, `PathConfig.get_tta_dir()`, `PathConfig.get_data_path()`

### Referenced Files
[`flax_model.py`](../small_transformer_based/flax_model.py) | [`flax_train.py`](../small_transformer_based/flax_train.py) | [`flax_eval.py`](../small_transformer_based/flax_eval.py) | [`generate_transformation.py`](../auxilaries/generate_transformation.py) | [`task.py`](../task.py) | [`utils.py`](../utils.py) | [`plots.py`](../plots.py)

## Transformer-assisted path — Sequence (proposals + optional TTA)

```mermaid
sequenceDiagram
    autonumber
    participant CLI as "flax_eval.py::main"
    participant Model as "FlaxCustomTransformer"
    participant Train as "TrainingPipeline"
    participant Eval as "EvaluationPipeline"
    participant TTA as "DataGeneration + TTA"
    participant Task as "task.py::Task"
    participant CA as "ConstraintAcquisition"
    participant Cache as "cache/*"

    %% Model Loading Phase
    CLI->>Cache: load tokenizer from vocab.json
    CLI->>Model: initialize FlaxCustomTransformer(vocab_size)
    CLI->>Cache: load checkpoint from cache/checkpoints/
    Model->>Model: restore model parameters from checkpoint
    
    %% Training Phase (if needed)
    alt Training Mode
        Train->>Cache: load training data from full_trans.json
        Train->>Train: CustomTokenizer.build_vocab(data)
        Train->>Train: CustomDataset(data, tokenizer)
        Train->>Train: create_train_state(model, learning_rate)
        loop Training Epochs
            Train->>Train: train_epoch(state, train_data, tokenizer)
            Train->>Train: train_step(state, batch) [JIT-compiled]
            Train->>Cache: save_checkpoint(state, epoch)
        end
    end

    %% Evaluation Phase
    CLI->>Eval: evaluate_true(state, model, tokenizer, tta=True/False)
    
    alt Batch Inference (No TTA)
        Eval->>Eval: batch_predict_transformations(state, model, tokenizer, tasks)
        loop For each task batch
            Eval->>Eval: format_grid_for_tokenizer(grid_data)
            Eval->>Model: predict_batch(params, input_ids, attention_mask) [JIT-compiled]
            Model-->>Eval: logits [batch, 3_cls_tokens, vocab_size]
            Eval->>Eval: extract_first_from_logits(tokenizer, logits, index)
            Eval-->>CLI: transformation predictions per task
        end
    end

    alt Test-Time Adaptation (TTA)
        loop For each task
            Eval->>TTA: check cache/tta/{task_id}/ for existing data
            alt TTA data missing
                TTA->>TTA: generate_samples(2500, output_folder, task_id)
                loop Parallel sample generation
                    TTA->>TTA: _generate_single_sample(no_of_trans, task_id)
                    TTA->>TTA: sample_and_apply(transformations, task_id)
                    TTA-->>Cache: save synthetic samples to cache/tta/{task_id}/
                end
                TTA->>Cache: save all_transformations.json
            end
            
            Eval->>Eval: clone model state for task-specific fine-tuning
            Eval->>TTA: load TTA data from cache/tta/{task_id}/
            Eval->>Train: CustomDataset(tta_data, tokenizer)
            
            loop TTA Fine-tuning (15 epochs)
                Eval->>Train: train_step(tta_state, tta_batch) [JIT-compiled]
                Train->>Train: cross_entropy_loss(logits, labels)
                Train->>Train: apply_gradients(grads)
            end
            
            Eval->>Eval: format_grid_for_tokenizer(task_grid)
            Eval->>Model: predict_batch(tta_params, input_ids, attention_mask)
            Model-->>Eval: fine-tuned logits
            Eval->>Eval: extract_first_from_logits(tokenizer, logits)
            Eval-->>CLI: task-specific predictions
        end
    end

    %% Integration with Symbolic Search
    CLI->>Task: Task(filepath, proposed_transformations=predictions)
    Task->>Task: initialize_frontier(candidate_transformations=predictions)
    Task->>CA: constraints_acquisition_global()
    CA->>CA: prune transformations based on training patterns
    CA-->>Task: filtered transformation_ops per abstraction
    
    Note over Task,CA: Search space significantly reduced by ML predictions
    
    loop Symbolic Search (Pruned Space)
        Task->>Task: get_candidate_filters()
        Task->>Task: get_candidate_transformations(filters)
        Note over Task: Only considers ML-proposed transformations
        Task->>Task: parameters_generation(filter, transformation)
        Task->>Task: expand_frontier(frontier_node)
        Task->>Task: calculate_score(apply_call)
        alt score == 0
            Task-->>CLI: solution found with ML guidance
        end
    end

    %% Results Logging
    CLI->>Cache: save evaluation results to cache/data/
    CLI->>Cache: log proposed_transformations.txt
    CLI-->>CLI: print success/failure statistics

```

---

**Legend:** Class nodes show primary file and key methods; sequence arrows show complete ML pipeline flow including training (tokenization, dataset creation, JAX/Flax training loop), evaluation (batch inference, TTA fine-tuning), data generation (parallel synthetic sample creation), and integration (ML predictions guide symbolic search space pruning).

(End of Transformer-assisted diagrams)
