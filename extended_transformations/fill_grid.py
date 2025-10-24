def fill_grid_based(grid, object, color, color1):
    if object == "empty_rectangle":
        h, w = len(grid), len(grid[0]) if grid else 0
        grid_filled = [
            [color if i in (0, h - 1) or j in (0, w - 1) else cell for j, cell in enumerate(row)]
            for i, row in enumerate(grid)
        ]
        return tuple(map(tuple, grid_filled))

    if object == "empty_rectangle_dynamic":
        value = next((val for row in grid for val in row if val != 0), None)
        rows, cols = len(grid), len(grid[0])
        return [
            [value if i in {0, rows - 1} or j in {0, cols - 1} else 0 for j in range(cols)]
            for i in range(rows)
        ]

    if object == "maximal_square":
        rows, cols = len(grid), len(grid[0])
        grid_copy = [row[:] for row in grid]
        while True:
            dp = [
                [1 if grid_copy[i][j] == 0 and (i == 0 or j == 0) else 0 for j in range(cols)]
                for i in range(rows)
            ]
            max_size, max_i, max_j = 0, -1, -1

            for i in range(1, rows):
                for j in range(1, cols):
                    if grid_copy[i][j] == 0:
                        dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                    if dp[i][j] > max_size:
                        max_size, max_i, max_j = dp[i][j], i, j
            if max_size < 2:
                break
            for i in range(max_i - max_size + 1, max_i + 1):
                for j in range(max_j - max_size + 1, max_j + 1):
                    grid[i][j], grid_copy[i][j] = color, -1
        return grid

    if object == "fill_and_swap":
        color2 = color

        def detect_direction(grid, color1):
            rows = len(grid)
            cols = len(grid[0])
            leftmost_column = [grid[row][0] for row in range(rows)]
            rightmost_column = [grid[row][cols - 1] for row in range(rows)]

            if color1 in leftmost_column:
                return "left"
            elif color1 in rightmost_column:
                return "right"
            else:
                raise ValueError(f"No color {color1} found in the leftmost or rightmost column.")

        def swap_and_modify_grid(grid, color1, color2):
            swapped_grid = []
            for row in grid:
                new_row = []
                for cell in row:
                    if cell == 0:
                        new_cell = 2
                    elif cell == 2:
                        new_cell = 0
                    else:
                        new_cell = cell
                    new_row.append(new_cell)
                swapped_grid.append(new_row)

            for i in range(len(swapped_grid)):
                for j in range(len(swapped_grid[0])):
                    if swapped_grid[i][j] == color1:
                        swapped_grid[i][j] = color2

            return swapped_grid

        direction = detect_direction(grid, color1)
        swapped_grid = swap_and_modify_grid(grid, color1, color2)
        if direction == "left":
            new_grid = [
                swapped_row[::-1] + original_row
                for swapped_row, original_row in zip(swapped_grid, grid)
            ]
        elif direction == "right":
            new_grid = [
                original_row + swapped_row[::-1]
                for original_row, swapped_row in zip(grid, swapped_grid)
            ]
        else:
            raise ValueError("Invalid duplication direction.")
        return new_grid

    if object == "checkboard":
        return [
            [color if (r % 2 == 0 or c % 2 == 0) else 0 for c in range(len(grid[0]))]
            for r in range(len(grid))
        ]
