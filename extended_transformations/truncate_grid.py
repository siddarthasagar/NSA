from copy import deepcopy
from extended_transformations.utils import *


def truncate_grid_based(grid, color1, color2, grid_size, truncate_type, mirror):
    if truncate_type == "position_based":
        target_color = grid[color1][color2]
        return [
            [
                0 if (i, j) != (color1, color2) and cell == target_color else cell
                for j, cell in enumerate(row)
            ]
            for i, row in enumerate(grid)
        ]
    if truncate_type == "inferior_based":
        transformed_grid = deepcopy(grid)
        rectangles = find_connected_components_multicolor(grid, color1, color2)
        if mirror:

            def condition(rect):
                return rect["count_1"] >= grid_size
        else:
            min_count_1 = min(rect["count_1"] for rect in rectangles)

            def condition(rect):
                return rect["count_1"] == min_count_1

        pixels_to_recolor = [
            (r, c) for rect in rectangles if condition(rect) for (r, c) in rect["pixels"]
        ]
        for r, c in pixels_to_recolor:
            transformed_grid[r][c] = 0
        return transformed_grid
