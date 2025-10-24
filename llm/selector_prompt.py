selector_prompt = (
    "Each task will demonstrate a transformation from an input grid to an output grid. At the end, you will receive a new input grid. Your task is to determine its corresponding output grid. "
    "Your goal is to analyze a given set of input-output grids and deduce the underlying pattern or rule that transforms the input into the output. "
    "You are asked to provide an abstraction, filter, binding, and transformation that is applied to an input grid to correctly transfer it to the output grid.\n\n"
    "Abstraction:\n"
    "Graph abstraction allows us to search for a solution at a macroscopic level. In other words, we are modifying groups of pixels at once, instead of modifying each individual pixel separately.\n\n"
    "Possible abstractions to choose from:\n"
    "ccg: Creates a graph where each node represents a group of adjacent pixels of the same color.\n"
    "ccgbr: Similar to ccg, but nodes identified as background (most common color and includes a corner) are removed.\n"
    "ccgbr2: Similar to ccgbr, but background nodes also include those touching any edge of the image.\n"
    "nbvcg: Creates a graph where each node is a group of vertically adjacent pixels of the same color, excluding background color.\n"
    "nbhcg: Creates a graph where each node is a group of horizontally adjacent pixels of the same color, excluding background color.\n"
    "nbccg: Similar to ccg, but excludes black pixels (color = 0).\n"
    "lrg: Creates a graph where each node is a rectangle of adjacent pixels of the same color, identified from largest to smallest, excluding black pixels.\n"
    "mcccg: Creates a graph where each node represents a group of adjacent pixels of any non-background color.\n"
    "na: Creates a graph with the entire grid as one multi-color node.\n\n"
    "Filters:\n"
    "Filters are used to select specific nodes in the graph based on certain criteria.\n\n"
    "Possible filters to choose from:\n"
    "filter_by_color: Returns true if a node has a given color. Optionally, exclude nodes with the given color.\n"
    "filter_by_size: Returns true if a node has a size equal to a given size. Optionally, exclude nodes of the given size.\n"
    "filter_by_degree: Returns true if a node has a degree equal to a given degree. Optionally, exclude nodes of the given degree.\n"
    "filter_by_neighbor_size: Returns true if a node has a neighbor of a given size. Optionally, exclude nodes with neighbors of the given size.\n"
    "filter_by_neighbor_color: Returns true if a node has a neighbor of a given color. Optionally, exclude nodes with neighbors of the given color.\n"
    "filter_by_neighbor_degree: Returns true if a node has a neighbor of a given degree. Optionally, exclude nodes with neighbors of the given degree.\n\n"
    "When specifying a filter, also provide its parameters in a dictionary format like this:\n"
    "{{filter: filter_by_size, filter_params: {{size: 2, exclude: False}}}}\n\n"
    "Transformations:\n"
    "Transformations are actions that change the properties or position of nodes.\n\n"
    "Possible transformations to choose from:\n"
    "update_color: Updates the color of a node to a given color. Parameters: {{color}}\n"
    "move_node: Moves a node by 1 pixel in a specified direction. Parameters: {{direction}}\nextend_node: Extends a node in a given direction, optionally allowing overlap with other nodes. Parameters: {{direction, overlap}}\n"
    "move_node_max: Moves a node in a specified direction until it hits another node or the edge of the image. Parameters: {{direction}}\n"
    "rotate_node: Rotates a node around its center point in a specified rotational direction. Parameters: {{rotation_dir}}\n"
    "add_border: Adds a border with a specified color around a node. Parameters: {{border_color}}\n"
    "fill_rectangle: Fills the rectangle containing a node with a specified color, optionally allowing overlap with other nodes. Parameters: {{fill_color, overlap}}\n"
    "hollow_rectangle: Hollows the rectangle containing a node with a specified color. Parameters: {{fill_color}}\n"
    "mirror: Mirrors a node with respect to a specified axis. Parameters: {{mirror_axis}}\n"
    "flip: Flips a node in a specified direction (horizontal, vertical, diagonal left/right). Parameters: {{mirror_direction}}\n"
    "remove_node: Removes a node from the graph.\n"
    "insert: Inserts a pattern at a specified location relative to a given node and point. Parameters: {{object_id, point, relative_pos}}\n\n"
    "Parameter Bindings:\n"
    "Parameter bindings are used to dynamically select parameters for filters or transformations based on the properties of nodes.\n\n"
    "Possible parameter bindings to choose from:\n"
    "param_bind_neighbor_by_size: Returns a neighbor of a node that satisfies a given size filter.\n"
    "param_bind_neighbor_by_color: Returns a neighbor of a node that satisfies a given color filter.\n"
    "param_bind_node_by_shape: Returns any node in the graph with the same shape as a given node.\n"
    "param_bind_node_by_size: Returns any node in the graph that satisfies a given size filter.\n"
    "param_bind_neighbor_by_degree: Returns a neighbor of a node that satisfies a given degree filter.\n\n"
    "When analyzing the feedback from the evaluator, focus on the key aspects that need adjustment to improve the transformation results. "
    "Consider any patterns or rules that were correctly or incorrectly applied and identify areas where unnecessary changes occurred or where the transformations could be more accurate. "
    "Training Examples:\n"
    "{training_examples}\n"
    "Based on the above examples, deduce the pattern and provide the transformation for the test input. "
    "Provide a detailed step-by-step explanation of your thought process."
    "Chain-of-Thought Prompting:\n"
    "To ensure accurate reasoning, provide a detailed step-by-step explanation of your thought process when analyzing the input-output grids and deducing the pattern. "
    "Explain how you identify the abstraction, filter, binding, and transformation. Include any intermediate steps and considerations that lead to your final solution. "
    "Ensure each step is explained clearly and concisely.\n\n"
    "Output should be in a form of a dictionary containing abstraction, filter, filter_params, transformation and transformation_params. Remember to specify exclude parameter if you use filter_by_size or filter_by_color.\n\n"
    # "{{abstraction: mcccg, filter: filter_by_size, filter_params: {{size: 2, exclude: False}}, transformation: update_color, transformation_params: {{color: 3}}, parameter_binding: param_bind_neighbor_by_size}}\n\n"
    # "parameter_binding is the only parameter that you can omit. If you don't want to use parameter_binding just omit it in the dictionary and just output:\n\n"
    # "{{abstraction: mcccg, filter: filter_by_size, filter_params: {{size: 2, exclude: False}}, transformation: update_color, transformation_params: {{color: 3}}}}\n\n"
)


def generate_selector_prompt(train_data):
    if isinstance(train_data, dict):
        train_data = train_data["train"]
    examples = []
    for example in train_data:
        input_grid = example["input"]
        output_grid = example["output"]

        # Convert the grids into the desired format
        input_grid_str = "\n".join(["|".join(map(str, row)) for row in input_grid])
        output_grid_str = "\n".join(["|".join(map(str, row)) for row in output_grid])

        # Format the examples as desired
        examples.append(f"Input:\n{input_grid_str}\nOutput:\n{output_grid_str}\n")

    training_examples_text = "\n".join(examples)

    prompt_text = selector_prompt.format(training_examples=training_examples_text)

    return prompt_text
