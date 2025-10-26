# ARC Agent — No-transformer mode (Pure symbolic search)

This document contains the Mermaid UML diagrams for the Final ARC agent's No-transformer (pure symbolic search) operation mode. It includes the focused class-level overview and the sequence diagram that shows runtime flow for solving a task using pure symbolic search (no ML proposals / TTA).

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

### Referenced Python files (relative paths)


- [`main.py`](../main.py)
- [`task.py`](../task.py)
- [`image.py`](../image.py)
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

Minimal legend: nodes show primary file and key functions; arrows show call/flow direction. Use these diagrams to jump to the exact function in the codebase (file names are inline with methods).

(End of No-transformer diagrams)
