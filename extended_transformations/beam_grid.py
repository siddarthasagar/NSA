import numpy as np
import copy
from extended_transformations.utils import *


def beam_grid_based(grid, color1=0, color2: int = 0, beam_type="color_inheritance"):
    if beam_type == "box_based":
        flat_grid = [val for row in grid for val in row]
        background_color = max(set(flat_grid), key=flat_grid.count)
        object_colors = set(flat_grid) - {background_color}
        objects = find_connected_components(
            grid,
            target_colors=object_colors,
            background_color=background_color,
            connectivity=4,
        )
        transformed_grid = [list(row) for row in grid]
        for obj in objects:
            object_cells = set(obj["pixels"])
            min_row, max_row, min_col, max_col = find_bounding_rectangle(obj["pixels"])

            delta_cells = {
                (row, col)
                for row in range(min_row, max_row + 1)
                for col in range(min_col, max_col + 1)
                if (row, col) not in object_cells
            }

            for row, col in delta_cells:
                transformed_grid[row][col] = color1

            perimeter_cells = (
                {(min_row, col) for col in range(min_col, max_col + 1)}.union(
                    {(max_row, col) for col in range(min_col, max_col + 1)}
                )
                .union({(row, min_col) for row in range(min_row, max_row + 1)})
                .union({(row, max_col) for row in range(min_row, max_row + 1)})
            )

            perimeter_diff = perimeter_cells - object_cells

            center_row = (min_row + max_row) / 2
            center_col = (min_col + max_col) / 2

            shooting_directions = {}
            for row, col in perimeter_diff:
                dr = -1 if row < center_row else (1 if row > center_row else 0)
                dc = -1 if col < center_col else (1 if col > center_col else 0)
                if abs(row - center_row) > abs(col - center_col):
                    dc = 0
                elif abs(col - center_col) > abs(row - center_row):
                    dr = 0
                if dr != 0 or dc != 0:
                    shooting_directions[(row, col)] = (dr, dc)

            for (row, col), (dr, dc) in shooting_directions.items():
                current_row, current_col = row + dr, col + dc
                while 0 <= current_row < len(grid) and 0 <= current_col < len(grid[0]):
                    if (current_row, current_col) in object_cells:
                        break
                    transformed_grid[current_row][current_col] = color1
                    current_row += dr
                    current_col += dc
        return transformed_grid

    elif beam_type == "infect":
        transformed_grid = copy.deepcopy(grid)
        rows, cols = len(grid), len(grid[0]) if grid else 0
        connected_components = find_connected_components(
            transformed_grid,
            target_colors=[color1],
            background_color=-1,
            connectivity=4,
        )
        rectangle_labels = [[-1] * cols for _ in range(rows)]
        rectangles = [component["pixels"] for component in connected_components]
        for idx, component in enumerate(connected_components):
            for r, c in component["pixels"]:
                rectangle_labels[r][c] = idx
        edge_pixels = (
            [(0, c, 1, 0) for c in range(cols) if transformed_grid[0][c] == color2]
            + [(rows - 1, c, -1, 0) for c in range(cols) if transformed_grid[rows - 1][c] == color2]
            + [(r, 0, 0, 1) for r in range(rows) if transformed_grid[r][0] == color2]
            + [(r, cols - 1, 0, -1) for r in range(rows) if transformed_grid[r][cols - 1] == color2]
        )
        recolored_rectangles = set()
        for r, c, dr, dc in edge_pixels:
            nr, nc = r + dr, c + dc
            while 0 <= nr < rows and 0 <= nc < cols:
                if transformed_grid[nr][nc] == color1:
                    rect_id = rectangle_labels[nr][nc]
                    if rect_id not in recolored_rectangles:
                        for rr, cc in rectangles[rect_id]:
                            transformed_grid[rr][cc] = color2
                        recolored_rectangles.add(rect_id)
                else:
                    transformed_grid[nr][nc] = color2

                for adj_r, adj_c in get_neighbors((nr, nc), (rows, cols), connectivity=4):
                    if transformed_grid[adj_r][adj_c] == color1:
                        adj_rect_id = rectangle_labels[adj_r][adj_c]
                        if adj_rect_id not in recolored_rectangles:
                            for arr, acc in rectangles[adj_rect_id]:
                                transformed_grid[arr][acc] = color2
                            recolored_rectangles.add(adj_rect_id)
                nr, nc = nr + dr, nc + dc
        return transformed_grid

    elif beam_type == "linspace":
        n_rows, n_cols = len(grid), len(grid[0]) if grid else 0
        positions = sorted(
            (i, j) for i, row in enumerate(grid) for j, val in enumerate(row) if val == color1
        )
        delta_row, delta_col = (
            positions[1][0] - positions[0][0],
            positions[1][1] - positions[0][1],
        )
        new_grid = [row[:] for row in grid]
        last_row, last_col = positions[-1]
        while 0 <= last_row + delta_row < n_rows and 0 <= last_col + delta_col < n_cols:
            last_row += delta_row
            last_col += delta_col
            new_grid[last_row][last_col] = color2
        return new_grid

    elif beam_type == "rectangle_shooting":
        rows, cols = len(grid), len(grid[0])
        positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == color1]
        min_row = min(r for r, _ in positions)
        max_row = max(r for r, _ in positions)
        min_col = min(c for _, c in positions)
        max_col = max(c for _, c in positions)
        new_grid = copy.deepcopy(grid)
        all_cells_to_set = set()
        directions = {
            "up": lambda: [
                (r, c)
                for c in range(min_col, max_col + 1)
                if all(grid[r_clear][c] == 0 for r_clear in range(min_row))
                for r in range(min_row)
            ],
            "down": lambda: [
                (r, c)
                for c in range(min_col, max_col + 1)
                if all(grid[r_clear][c] == 0 for r_clear in range(max_row + 1, rows))
                for r in range(max_row + 1, rows)
            ],
            "left": lambda: [
                (r, c)
                for r in range(min_row, max_row + 1)
                if all(grid[r][c_clear] == 0 for c_clear in range(min_col))
                for c in range(min_col)
            ],
            "right": lambda: [
                (r, c)
                for r in range(min_row, max_row + 1)
                if all(grid[r][c_clear] == 0 for c_clear in range(max_col + 1, cols))
                for c in range(max_col + 1, cols)
            ],
        }
        for cells in directions.values():
            all_cells_to_set.update(cells())
        for r, c in all_cells_to_set:
            new_grid[r][c] = color1
        return new_grid

    elif beam_type == "space_based":
        transformed_grid = [row.copy() for row in grid]
        rows, cols = len(grid), len(grid[0])
        separation_rows = [
            i
            for i in range(1, rows - 1)
            if all(cell == 0 for cell in grid[i])
            and any(grid[k][j] != 0 for k in range(0, i) for j in range(cols))
            and any(grid[k][j] != 0 for k in range(i + 1, rows) for j in range(cols))
        ]
        separation_cols = [
            j
            for j in range(1, cols - 1)
            if all(grid[i][j] == 0 for i in range(rows))
            and any(grid[i][k] != 0 for i in range(rows) for k in range(0, j))
            and any(grid[i][k] != 0 for i in range(rows) for k in range(j + 1, cols))
        ]
        if len(separation_rows) == 1 and not separation_cols:
            for j in range(cols):
                transformed_grid[separation_rows[0]][j] = color1
        elif len(separation_cols) == 1 and not separation_rows:
            for i in range(rows):
                transformed_grid[i][separation_cols[0]] = color1
        else:
            raise ValueError(
                "Multiple separation spaces found. Only one separation space is allowed."
            )
        return transformed_grid

    elif beam_type == "most_color_line":
        transformed_grid = [row.copy() for row in grid]
        rows, cols = len(grid), len(grid[0])
        mid_col = cols // 2
        line_of_color1_row = next(
            (i for i, row in enumerate(grid) if all(cell == color1 for cell in row)),
            None,
        )
        color_counts = Counter(
            cell for i in range(line_of_color1_row) for cell in grid[i] if cell not in {0, color1}
        )
        most_frequent_color = min(
            (color for color, count in color_counts.items() if count == max(color_counts.values())),
            default=0,
        )
        transformed_grid[-1][mid_col] = most_frequent_color
        return transformed_grid

    elif beam_type == "color_inheritance":
        grid_np = np.array(grid)
        flat = grid_np.flatten()
        background_color = np.bincount(flat).argmax()
        unique, counts = np.unique(flat, return_counts=True)
        mask = unique != background_color
        beam_color = unique[mask][np.argmin(counts[mask])]
        beam_cells = np.argwhere(grid_np == beam_color)
        beam_center = (
            beam_cells.mean(axis=0)
            if beam_cells.size
            else np.array([len(grid) / 2, len(grid[0]) / 2])
        )
        object_colors = set(unique) - {background_color, beam_color}
        if object_colors:
            object_cells = np.argwhere(np.isin(grid_np, list(object_colors)))
            object_center = (
                object_cells.mean(axis=0)
                if object_cells.size
                else np.array([len(grid) / 2, len(grid[0]) / 2])
            )
        else:
            object_center = np.array([len(grid) / 2, len(grid[0]) / 2])
        delta = object_center - beam_center
        if abs(delta[0]) > abs(delta[1]):
            dr, dc = (1 if delta[0] > 0 else -1, 0)
        elif abs(delta[1]) > abs(delta[0]):
            dr, dc = (0, 1 if delta[1] > 0 else -1)
        else:
            dr, dc = (1 if delta[0] > 0 else -1, 1 if delta[1] > 0 else -1)
        transformed_grid = grid_np.copy()
        not_zero = np.argwhere(transformed_grid != 0)
        if dr == -1 and dc == 0:
            start_idx = not_zero[:, 0].argmin()
        elif dr == 1 and dc == 0:
            start_idx = not_zero[:, 0].argmax()
        elif dr == 0 and dc == 1:
            start_idx = not_zero[:, 1].argmax()
        elif dr == 0 and dc == -1:
            start_idx = not_zero[:, 1].argmin()
        else:
            start_idx = 0
        i0, j0 = not_zero[start_idx]
        i, j = i0 + dr, j0 + dc
        while 0 <= i < transformed_grid.shape[0] and 0 <= j < transformed_grid.shape[1]:
            transformed_grid[i, j] = beam_color
            i += dr
            j += dc
        return transformed_grid.tolist()
