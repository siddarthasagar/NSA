from extended_transformations.utils import *


def upscale_grid_based(grid, factor, mirror, upscale_type, color, border_color, fill_color):
    if upscale_type == "standard":
        upscaled_grid = []
        for i, row in enumerate(grid):
            new_rows = [[] for _ in range(factor)]
            for j, value in enumerate(row):
                if value == 0:
                    for k in range(factor):
                        new_rows[k].extend([0] * factor)
                elif value == border_color:
                    for k in range(factor):
                        for col_off in range(factor):
                            if ((i * factor + k) + (j * factor + col_off)) % 2 == 0:
                                val = color
                            else:
                                val = fill_color
                            new_rows[k].append(val)
            upscaled_grid.extend(new_rows)
        return

    if upscale_type == "pixel_based":
        if mirror:
            grid = swap_with_zero(grid)
        upscaled_grid = [
            [value for value in row for _ in range(factor)] for row in grid for _ in range(factor)
        ]
        tiled_grid = [row * factor for _ in range(factor) for row in grid]
        transformed_grid = [
            [u if u == t else 0 for u, t in zip(u_row, t_row)]
            for u_row, t_row in zip(upscaled_grid, tiled_grid)
        ]
        return transformed_grid

    if upscale_type == "unique_colors":
        factor = count_unique_colors_except_zero(grid)

    return tuple(
        tuple(value for value in row for _ in range(factor)) for row in grid for _ in range(factor)
    )
