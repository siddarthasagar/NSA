# ARC Agent — Detailed UML (Two Operation Modes)

This document contains detailed Mermaid UML diagrams that show the Final ARC agent's two operation modes: (A) Pure symbolic search and (B) Transformer‑assisted (with optional Test‑Time Adaptation). Each diagram lists the primary files and the core functions called during the flows so you can quickly map code → runtime.

---

## Class-level overview

```mermaid
classDiagram
    %% High-level components and their main methods (file paths in italics)

    %% Style definitions for class diagram
    classDef ingest fill:#E3F2FD,stroke:#1E88E5,stroke-width:1px;
    classDef processing fill:#FFF3E0,stroke:#FB8C00,stroke-width:1px;
    classDef transform fill:#E8F5E9,stroke:#43A047,stroke-width:1px;
    classDef ext fill:#FFEBEE,stroke:#E53935,stroke-width:1px;
    classDef storage fill:#F3E5F5,stroke:#8E24AA,stroke-width:1px;

    class Main {
        +solve_task_id(task_file, task_type, time_limit, save_images, q)\n(*main.py*)
        +main()\n(*main.py*)
    }

    class Task {
        +__init__(filepath, proposed_transformations=None)\n(*task.py*)
        +solve(shared_frontier=True, time_limit, do_constraint_acquisition=True, save_images=False)\n(*task.py*)
        +initialize_frontier(candidate_transformations=None)\n(*task.py*)
        +expand_frontier(frontier_node)\n(*task.py*)
        +get_candidate_filters()\n(*task.py*)
        +get_candidate_transformations(apply_filters_calls)\n(*task.py*)
        +parameters_generation(apply_filters_call, transform_sig)\n(*task.py*)
        +calculate_score(apply_call)\n(*task.py*)
        +apply_solution(apply_call, abstraction, save_images=False)\n(*task.py*)
    }

    class Image {
        +__init__(task, grid=None, width=None, height=None, graph=None, name)\n(*image.py*)
        +get_connected_components_graph()\n(*image.py*)
        +get_multicolor_connected_components_graph()\n(*image.py*)
        +get_no_abstraction_graph()\n(*image.py*)
        +undo_abstraction(arc_graph, adjust_to_bounding_box=True)\n(*image.py*)
    }

    class ARCGraph {
        +__init__(graph, name, image, abstraction=None)\n(*ARCGraph.py*)
        +apply(filters, filter_params, transformation, transformation_params)\n(*ARCGraph.py*)
        +apply_filters(node, filters, filter_params)\n(*ARCGraph.py*)
        +apply_param_binding(node, transformation_params)\n(*ARCGraph.py*)
        +apply_transformation(nodes, transformation, transformation_params)\n(*ARCGraph.py*)
        +graph_to_grid() / update_graph_from_grid(grid)\n(*ARCGraph.py*)
        +undo_abstraction(adjust_to_bounding_box)\n(*ARCGraph.py*)
        +move_node/extend_node/rotate_node/insert/truncate/...\n(*ARCGraph.py* + extended_transformations/*)
    }

    class ExtendedTransformations {
        +crop_grid_based(grid, ...)\n(*extended_transformations/crop_grid.py*)
        +upscale_grid_based(grid, ...)\n(*extended_transformations/upscale_grid.py*)
        +truncate_grid_based(grid, ...)\n(*extended_transformations/truncate_grid.py*)
        +... (fill, connect, shift, recolor, beam, etc.)\n(*extended_transformations/*)
    }

    class TransformerModel {
        +train(...)\n(*small_transformer_based/flax_train.py / train.py*)
        +evaluate(...)/propose_transformations(...)\n(*small_transformer_based/eval.py / train.py*)
        +tokenize / detokenize / TTA_finetune(...)\n(*small_transformer_based/*)
    }

    class CacheAndTTA {
        +checkpoints / data / tta storage and retrieval\n(*cache/*)
    }

    %% Relationships
    Main --> Task : creates/starts
    Task --> Image : uses for abstractions
    Task --> ARCGraph : manipulates (via Image.arc_graph)
    ARCGraph --> ExtendedTransformations : delegates grid ops
    Task --> TransformerModel : (optional) receives proposals / TTA
    TransformerModel --> CacheAndTTA : stores/loads checkpoints and TTA artifacts
    Task --> CacheAndTTA : read/write TTA data and generated samples

``` 

---

## No-transformer path — Class-level overview (pure symbolic search)

```mermaid
classDiagram
    %% Focused classes for pure symbolic search
    classDef ingest fill:#E3F2FD,stroke:#1E88E5,stroke-width:1px;
    classDef processing fill:#FFF3E0,stroke:#FB8C00,stroke-width:1px;
    classDef ext fill:#FFEBEE,stroke:#E53935,stroke-width:1px;
    classDef util fill:#E0F7FA,stroke:#006064,stroke-width:1px;

    class Task_Pure {
        +__init__(filepath, proposed_transformations=None)\n(*task.py*)
        +solve(...)\n(*task.py*)
        +get_candidate_filters()\n(*task.py*)
        +parameters_generation(...)\n(*task.py*)
        +calculate_score(...)\n(*task.py*)
    }

    class Image_Pure {
        +get_connected_components_graph()\n(*image.py*)
        +undo_abstraction(...)\n(*image.py*)
    }

    class ARCGraph_Pure {
        +apply(...)\n(*ARCGraph.py*)
        +apply_transformation(...)\n(*ARCGraph.py*)
        +move_node / rotate_node / duplicate / insert / ...\n(*ARCGraph.py*)
    }

    class ExtendedTransformations_Pure {
        +crop_grid_based(...)\n(*extended_transformations/*)
        +upscale_grid_based(...)\n(*extended_transformations/*)
    }

    class PriorityItem {
        +__init__(data, abstraction, priority, secondary)\n(*priority_item.py*)
    }

    class Rules {
        +color_equal(...)\n(*rules.py*)
        +size_equal(...)\n(*rules.py*)
        +position_equal(...)\n(*rules.py*)
    }

    class Utils {
        +helper functions\n(*utils.py*)
    }

    class Task_Pure processing
    class Image_Pure ingest
    class ARCGraph_Pure processing
    class ExtendedTransformations_Pure ext
    class PriorityItem util
    class Rules util
    class Utils util

    Task_Pure --> Image_Pure : builds abstractions
    Task_Pure --> ARCGraph_Pure : manipulates
    ARCGraph_Pure --> ExtendedTransformations_Pure : delegates grid ops
    Task_Pure --> Rules : constraint acquisition
    Task_Pure --> PriorityItem : frontier items

```

## No-transformer path — Sequence (pure symbolic search)

```mermaid
sequenceDiagram
    autonumber
    participant CLI as "main.py::main"
    participant Worker as "main.py::solve_task_id"
    participant Task as "task.py::Task"
    participant Image as "image.py::Image"
    participant ARC as "ARCGraph.py::ARCGraph"
    participant Ext as "extended_transformations/*"

    CLI->>Worker: start Process with (task_file, task_type)
    Worker->>Task: Task(filepath) -- constructor parses JSON and builds Image objects
    Task->>Image: Image(... grid ...) for each train/test case
    Note over Task,Image: Image constructs base graph and assigns Image.arc_graph

    Worker->>Task: Task.solve(shared_frontier=True, time_limit...)
    Task->>Task: initialize_frontier()
    Task->>Image: Image.abstraction_ops[abstraction]()
    Image-->>Task: ARCGraph (one per abstraction)

    loop Search loop
        Task->>Task: get_candidate_filters()
        Task->>Task: get_candidate_transformations(filters)
        Task->>Task: parameters_generation(...)
        Task->>Task: expand_frontier(frontier_node)
        Task->>ARC: apply(filters, filter_params, transformation, transformation_params)
        ARC->>ARC: apply_param_binding(...)
        ARC->>ARC: apply_transformation(...)
        ARC->>Ext: call grid-based transform (if applicable)
        Ext-->>ARC: transformed grid
        ARC->>ARC: update_graph_from_grid(grid)
        Task->>Task: calculate_score(apply_call)
        Task->>Image: undo_abstraction(...)
        Task->>Task: compare grids and compute score
        alt score == 0
            Task->>Worker: return solution (apply_call, abstraction)
        end
    end

    Worker-->>CLI: prints JSON result

```

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


(End of diagrams)
```
