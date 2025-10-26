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
    classDef search fill:#F3E5F5,stroke:#8E24AA,stroke-width:1px;

    class Task_Pure {
        +__init__(filepath, proposed_transformations=None)\n(*task.py*)
        +solve(...)\n(*task.py*)
        +initialize_frontier()\n(*task.py*)
        +search_shared_frontier()\n(*task.py*)
        +expand_frontier(frontier_node)\n(*task.py*)
        +get_candidate_filters()\n(*task.py*)
        +get_candidate_transformations(filters)\n(*task.py*)
        +parameters_generation(...)\n(*task.py*)
        +calculate_score(apply_call)\n(*task.py*)
        +get_static_inserted_objects()\n(*task.py*)
        +get_static_object_attributes(abstraction)\n(*task.py*)
    }

    class ConstraintAcquisition {
        +constraints_acquisition_global()\n(*task.py*)
        +constraints_acquisition_local(apply_filter_call)\n(*task.py*)
        +apply_constraint(rule, apply_filter_call)\n(*task.py*)
        +prune_transformations(constraints)\n(*task.py*)
        +constraints_param_generation(...)\n(*task.py*)
    }

    class SearchManager {
        +frontier: PriorityQueue\n(*task.py*)
        +tabu_list: dict\n(*task.py*)
        +frontier_hash: dict\n(*task.py*)
        +current_best_scores: dict\n(*task.py*)
    }

    class Image_Pure {
        +get_connected_components_graph()\n(*image.py*)
        +get_non_black_components_graph()\n(*image.py*)
        +get_multicolor_connected_components_graph()\n(*image.py*)
        +get_no_abstraction_graph()\n(*image.py*)
        +undo_abstraction(arc_graph, adjust_to_bounding_box)\n(*image.py*)
    }

    class ARCGraph_Pure {
        +apply(filters, filter_params, transformation, transformation_params)\n(*ARCGraph.py*)
        +apply_filters(node, filters, filter_params)\n(*ARCGraph.py*)
        +apply_param_binding(node, transformation_params)\n(*ARCGraph.py*)
        +apply_transformation(nodes, transformation, transformation_params)\n(*ARCGraph.py*)
        +filter_by_color / filter_by_size / filter_by_degree\n(*ARCGraph.py*)
        +param_bind_neighbor_by_color / param_bind_neighbor_by_size\n(*ARCGraph.py*)
        +undo_abstraction(adjust_to_bounding_box)\n(*ARCGraph.py*)
        +move_node / rotate_node / duplicate / insert / ...\n(*ARCGraph.py*)
    }

    class ExtendedTransformations_Pure {
        +crop_grid_based(...)\n(*extended_transformations/*)
        +upscale_grid_based(...)\n(*extended_transformations/*)
        +magnet_grid_based(...)\n(*extended_transformations/*)
        +beam_grid_based(...)\n(*extended_transformations/*)
    }

    class PriorityItem {
        +__init__(data, abstraction, priority, secondary)\n(*priority_item.py*)
    }

    class Rules {
        +color_equal(input_seq, output_seq)\n(*rules.py*)
        +size_equal(input_seq, output_seq)\n(*rules.py*)
        +position_equal(input_seq, output_seq)\n(*rules.py*)
        +list_of_rules: list\n(*rules.py*)
    }

    class Utils {
        +helper functions\n(*utils.py*)
    }

    class Task_Pure processing
    class ConstraintAcquisition search
    class SearchManager search
    class Image_Pure ingest
    class ARCGraph_Pure processing
    class ExtendedTransformations_Pure ext
    class PriorityItem util
    class Rules util
    class Utils util

    Task_Pure --> Image_Pure : builds abstractions
    Task_Pure --> ARCGraph_Pure : manipulates
    Task_Pure --> ConstraintAcquisition : prunes search space
    Task_Pure --> SearchManager : manages frontier & tabu
    ARCGraph_Pure --> ExtendedTransformations_Pure : delegates grid ops
    ConstraintAcquisition --> Rules : applies constraints
    SearchManager --> PriorityItem : frontier items

```

### Key Methods by Component

**Task (task.py):**
- `solve()`, `initialize_frontier()`, `search_shared_frontier()`, `expand_frontier()`
- `get_candidate_filters()`, `get_candidate_transformations()`, `parameters_generation()`
- `calculate_score()`, `get_static_inserted_objects()`, `get_static_object_attributes()`
- `constraints_acquisition_global()`, `constraints_acquisition_local()`, `prune_transformations()`

**ARCGraph (ARCGraph.py):**
- `apply()`, `apply_filters()`, `apply_param_binding()`, `apply_transformation()`
- `filter_by_color()`, `filter_by_size()`, `filter_by_degree()`, `filter_by_neighbor_**()`
- `param_bind_neighbor_by_color()`, `param_bind_neighbor_by_size()`, `param_bind_node_by_shape()`
- `undo_abstraction()`, `undo_abstraction1()`, `undo_abstraction2()`
- Transformations: `move_node()`, `rotate_node()`, `extract()`, `duplicate()`, `insert()`, etc.

**Image (image.py):**
- Abstractions: `get_connected_components_graph()`, `get_non_black_components_graph()`
- `get_multicolor_connected_components_graph()`, `get_no_abstraction_graph()`
- `undo_abstraction()`

**Rules (rules.py):**
- `color_equal()`, `size_equal()`, `position_equal()`, `list_of_rules`

### Referenced Files
[`main.py`](../main.py) | [`task.py`](../task.py) | [`image.py`](../image.py) | [`ARCGraph.py`](../ARCGraph.py) | [`priority_item.py`](../priority_item.py) | [`rules.py`](../rules.py) | [`utils.py`](../utils.py) | [`extended_transformations/*`](../extended_transformations/)

## No-transformer path — Sequence (pure symbolic search)

```mermaid
sequenceDiagram
    autonumber
    participant CLI as "main.py::main"
    participant Worker as "main.py::solve_task_id"
    participant Task as "task.py::Task"
    participant CA as "ConstraintAcquisition"
    participant SM as "SearchManager"
    participant Image as "image.py::Image"
    participant ARC as "ARCGraph.py::ARCGraph"
    participant Ext as "extended_transformations/*"

    CLI->>Worker: start Process with (task_file, task_type)
    Worker->>Task: Task(filepath) -- constructor parses JSON and builds Image objects
    Task->>Image: Image(... grid ...) for each train/test case
    Note over Task,Image: Image constructs base graph and assigns Image.arc_graph

    Worker->>Task: Task.solve(shared_frontier=True, time_limit...)
    Task->>Task: initialize_frontier()
    
    loop For each abstraction
        Task->>Image: Image.abstraction_ops[abstraction]()
        Image-->>Task: ARCGraph (input/output pairs)
        Task->>CA: constraints_acquisition_global()
        CA->>CA: analyze training patterns
        CA-->>Task: pruned transformation list
        Task->>Task: get_static_inserted_objects()
        Task->>Task: get_static_object_attributes(abstraction)
        Task->>SM: initialize frontier with dummy node
        Task->>Task: expand_frontier(dummy_node)
        alt solution found in initialization
            Task->>Worker: return solution
        end
    end

    loop Main search loop
        SM->>SM: get next frontier node from priority queue
        alt node abstraction on tabu list
            SM->>SM: add to waiting list, continue
        else node score >= current best
            SM->>SM: add abstraction to tabu list
        end
        
        Task->>Task: get_candidate_filters()
        loop Generate filter combinations
            Task->>ARC: test filter on all training inputs
            ARC-->>Task: filtered nodes
        end
        
        Task->>Task: get_candidate_transformations(filters)
        loop For each filter + transformation combo
            Task->>CA: constraints_acquisition_local(filter)
            CA->>CA: check constraints against training data
            CA-->>Task: valid transformations only
            Task->>Task: parameters_generation(filter, transformation)
            loop Generate parameter combinations
                Task->>ARC: apply_param_binding(node, params)
                ARC->>ARC: resolve dynamic parameters
                ARC-->>Task: bound parameters
            end
        end
        
        Task->>Task: expand_frontier(frontier_node)
        loop For each candidate apply_call
            Task->>ARC: apply(filters, filter_params, transformation, transformation_params)
            ARC->>ARC: apply_filters(node, filters, filter_params)
            ARC->>ARC: apply_param_binding(node, transformation_params)
            ARC->>ARC: apply_transformation(nodes, transformation, params)
            ARC->>Ext: call grid-based transform (if applicable)
            Ext-->>ARC: transformed grid
            ARC->>ARC: update_graph_from_grid(grid)
            
            Task->>Task: calculate_score(apply_call)
            loop For each training example
                Task->>Image: undo_abstraction(transformed_graph)
                Image->>ARC: undo_abstraction(adjust_to_bounding_box)
                ARC-->>Image: reconstructed Image
                Image-->>Task: reconstructed grid
                Task->>Task: compare with expected output
                Task->>Task: count pixel mismatches
            end
            Task->>Task: aggregate scores across training examples
            
            alt score == 0
                Task->>Worker: return solution (apply_call, abstraction)
            else score improved
                SM->>SM: add to frontier with priority = score
            end
        end
        
        SM->>SM: update tabu lists and waiting queues
        alt time limit exceeded
            Task->>Worker: return best solution found
        end
    end

    Worker-->>CLI: prints JSON result

```

---

**Legend:** Class nodes show primary file and key methods; sequence arrows show call/flow direction with critical phases: initialization (constraint acquisition, static object detection), search (filter generation, parameter binding, transformation application), and evaluation (scoring, grid reconstruction).

(End of No-transformer diagrams)
