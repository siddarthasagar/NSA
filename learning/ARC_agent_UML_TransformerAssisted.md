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

    class TransformerModel_A {
        +train(...)\n(*small_transformer_based/train.py*)
        +propose_transformations(...)\n(*small_transformer_based/eval.py*)
        +fine_tune_on_TTA(...)\n(*small_transformer_based/*)
    }

    class TTA_A {
        +generate_samples(task, proposals)\n(*auxilaries/generate_transformation.py*)
    }

    class Task_TA {
        +__init__(filepath, proposed_transformations=None)\n(*task.py*)
        +initialize_frontier(candidate_transformations=proposals)\n(*task.py*)
    }

    class Cache_A {
        +checkpoints / data / tta storage\n(*cache/*)
    }

    class PriorityItem_A {
        +__init__(...)\n(*priority_item.py*)
    }

    class Rules_A {
        +color_equal(...)\n(*rules.py*)
    }

    class TransformerModel_A transform
    class TTA_A storage
    class Task_TA processing
    class ARCGraph_Pure processing
    class ExtendedTransformations_Pure ext
    class Cache_A storage
    class PriorityItem_A util
    class Rules_A util

    TransformerModel_A --> Cache_A : load/save checkpoints
    TransformerModel_A --> TTA_A : use generated samples to fine-tune
    Task_TA --> TransformerModel_A : receives proposals
    Task_TA --> ARCGraph_Pure : same symbolic ops

```

### Referenced Python files (relative paths)


- [`small_transformer_based/train.py`](../small_transformer_based/train.py)
- [`small_transformer_based/eval.py`](../small_transformer_based/eval.py)
- [`small_transformer_based/flax_train.py`](../small_transformer_based/flax_train.py)
- [`small_transformer_based/flax_eval.py`](../small_transformer_based/flax_eval.py)
- [`small_transformer_based/flax_model.py`](../small_transformer_based/flax_model.py)
- [`auxilaries/generate_transformation.py`](../auxilaries/generate_transformation.py)
- [`main.py`](../main.py)
- [`task.py`](../task.py)
- [`ARCGraph.py`](../ARCGraph.py)
- [`priority_item.py`](../priority_item.py)
- [`rules.py`](../rules.py)
- [`utils.py`](../utils.py)
- [`extended_transformations/arbitrary_duplicate_grid.py`](../extended_transformations/arbitrary_duplicate_grid.py)
- [`extended_transformations/beam_grid.py`](../extended_transformations/beam_grid.py)
- [`extended_transformations/connect_grid.py`](../extended_transformations/connect_grid.py)
- [`extended_transformations/crop_grid.py`](../extended_transformations/crop_grid.py)
- [`extended_transformations/fill_grid.py`](../extended_transformations/fill_grid.py)
- [`extended_transformations/magnet_grid.py`](../extended_transformations/magnet_grid.py)
- [`extended_transformations/mirror_grid.py`](../extended_transformations/mirror_grid.py)
- [`extended_transformations/recolor_grid.py`](../extended_transformations/recolor_grid.py)
- [`extended_transformations/rotate_duplicate.py`](../extended_transformations/rotate_duplicate.py)
- [`extended_transformations/rotate_grid.py`](../extended_transformations/rotate_grid.py)
- [`extended_transformations/shift_grid.py`](../extended_transformations/shift_grid.py)
- [`extended_transformations/truncate_grid.py`](../extended_transformations/truncate_grid.py)
- [`extended_transformations/upscale_grid.py`](../extended_transformations/upscale_grid.py)
- [`extended_transformations/utils.py`](../extended_transformations/utils.py)

## Transformer-assisted path — Sequence (proposals + optional TTA)

```mermaid
sequenceDiagram
    autonumber
    participant CLI as "small_transformer_based/eval"
    participant Transformer as "TransformerModel (small_transformer_based)"
    participant TTA as "auxilaries/generate_transformation.py + cache/tta"
    participant Task as "task.py::Task"
    participant ARC as "ARCGraph.py::ARCGraph"
    participant Ext as "extended_transformations/*"
    participant Cache as "cache/*"

    CLI->>Transformer: load model (from cache/checkpoints)
    Transformer->>Transformer: propose_transformations(task_meta)
    Transformer-->>CLI: proposals list

    alt Test-time adaptation (TTA) enabled
        CLI->>TTA: generate_samples(task, proposals)
        TTA-->>Cache: save generated samples under cache/tta/{task_id}/
        Transformer->>Transformer: fine_tune_on_TTA(cache/tta/{task_id})
        Transformer-->>CLI: updated proposals
    end

    CLI->>Task: Task(filepath, proposed_transformations=proposals)
    Task->>Task: initialize_frontier(candidate_transformations=proposals)
    Note over Task: transformation_ops pruned to proposals per abstraction

    loop Search loop (pruned set)
        Task->>Task: get_candidate_filters()
        Task->>Task: get_candidate_transformations(filters)
        Task->>Task: parameters_generation(...)
        Task->>ARC: apply(...)
        ARC->>Ext: grid-based ops
        Ext-->>ARC: transformed grid
        Task->>Task: calculate_score(...)
        alt score == 0
            Task-->>CLI: success (apply_call)
        end
    end

    CLI->>Transformer: log proposals and chosen apply_call

```

---

Minimal legend: nodes show primary file and key functions; arrows show call/flow direction. Use these diagrams to jump to the exact function in the codebase (file names are inline with methods).

(End of Transformer-assisted diagrams)
