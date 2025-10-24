from collections import defaultdict
import copy
from extended_transformations.utils import *


def recolor_grid_based(grid, recolor_type, color1, color2, shifting_direction):
    if recolor_type == "fill_blank":
        transformed_grid = copy.deepcopy(grid)
        objects = find_connected_components(grid, target_colors={color1}, connectivity=8)

        for idx, obj in enumerate(objects, 1):
            pixels = obj["pixels"]
            rows = [cell[0] for cell in pixels]
            min_row = min(rows)
            max_row = max(rows)
            height = max_row - min_row + 1
            midpoint = min_row + height // 2

            for i, j in pixels:
                if i >= midpoint:
                    transformed_grid[i][j] = color2

        return transformed_grid

    elif recolor_type == "moving_recolor":
        num_rows, num_cols = len(grid), len(grid[0])
        new_grid = [[color1] * num_cols for _ in range(num_rows)]
        shift_map = {
            "left": (0, -1, range(num_rows), range(num_cols)),
            "right": (0, 1, range(num_rows), reversed(range(num_cols))),
            "up": (-1, 0, range(num_rows), range(num_cols)),
            "down": (1, 0, range(num_rows), reversed(range(num_cols))),
        }
        dr, dc, row_iter, col_iter = shift_map[shifting_direction]
        for r in row_iter:
            for c in col_iter:
                if grid[r][c] != 0:
                    new_r, new_c = r + dr, c + dc
                    if 0 <= new_r < num_rows and 0 <= new_c < num_cols:
                        new_grid[new_r][new_c] = grid[r][c]
        return new_grid

    if recolor_type == "nearest_pixels":
        objects = find_connected_components(grid, background_color=0, connectivity=4)
        transformed_grid = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
        recolored_object_pixels = set()
        for obj in objects:
            if obj["color"] == color1:
                nearest_color = find_nearest_color(grid, obj)
                for r, c in obj["pixels"]:
                    transformed_grid[r][c] = nearest_color
                    recolored_object_pixels.add((r, c))
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if (r, c) not in recolored_object_pixels:
                    transformed_grid[r][c] = 0
        return transformed_grid

    elif recolor_type == "line_inheritance":
        return [[row[0] if cell == color1 else cell for cell in row] for row in grid]

    elif recolor_type == "square_spread":
        H, W = len(grid), len(grid[0])
        squares = defaultdict(list)
        for i in range(H - 1):
            for j in range(W - 1):
                square = (
                    (grid[i][j], grid[i][j + 1]),
                    (grid[i + 1][j], grid[i + 1][j + 1]),
                )
                if any(v != 0 for row in square for v in row):
                    squares[square].append((i, j))
        unique = next(((sq, pos[0]) for sq, pos in squares.items() if len(pos) == 1), None)
        unique_square, (i, j) = unique
        color = next(v for row in unique_square for v in row if v != 0)
        positions = {
            (p, q)
            for offset in [0, 1]
            for k in range(W - 1)
            for p in [i + offset, i + offset + 1]
            for q in [k, k + 1]
            if grid[p][q] != 0
        }.union(
            {
                (p, q)
                for offset in [0, 1]
                for k in range(H - 1)
                for p in [k, k + 1]
                for q in [j + offset, j + offset + 1]
                if grid[p][q] != 0
            }
        )
        for p, q in positions:
            grid[p][q] = color
        return grid

    elif recolor_type == "border_based":
        for comp in find_connected_components(grid, target_colors={color1}):
            min_r, max_r, min_c, max_c = find_bounding_rectangle(comp["pixels"])
            ir_min, ir_max, ic_min, ic_max = min_r + 1, max_r - 1, min_c + 1, max_c - 1
            sub = extract_rectangle(grid, (ir_min, ic_min, ir_max, ic_max))
            unique = {c for row in sub for c in row if c not in (0, color1)}
            if len(unique) < 2:
                continue
            c1, c2 = list(unique)[:2]
            sub_np = np.array(sub)
            sub_np = np.where(sub_np == c1, c2, np.where(sub_np == c2, c1, sub_np))
            for i, r in enumerate(range(ir_min, ir_max + 1)):
                grid[r][ic_min : ic_max + 1] = sub_np[i].tolist()
        return grid
