# TODO: Update this function to be able to scale across x and y axes with different scale factors


from typing import List, Union, Tuple
from collections import Counter

Grid = List[List[Union[int, float]]]
Metadata = List[List[List[Tuple[int, int, Union[int, float]]]]]


def scale_grid_simple(grid: Grid, scale_factor: float) -> Grid:
    """
    Scale a grid by a given factor using pure Python (no numpy required).

    Args:
        grid: 2D array (list of lists) of numbers
        scale_factor: Factor to scale the grid by (e.g., 2.0 doubles the size, 0.5 halves it)

    Returns:
        Scaled grid as a 2D array
    """
    if not grid or not grid[0]:
        return []

    if scale_factor <= 0:
        raise ValueError("Scale factor must be positive")

    rows = len(grid)
    cols = len(grid[0])

    new_rows = int(rows * scale_factor)
    new_cols = int(cols * scale_factor)

    if new_rows == 0 or new_cols == 0:
        return []

    if scale_factor > 1:
        return _upscale_grid_simple(grid, new_rows, new_cols)
    else:
        return _downscale_grid_simple(grid, new_rows, new_cols)


def _upscale_grid_simple(grid: Grid, new_rows: int, new_cols: int) -> Grid:
    """Helper function to upscale a grid by repeating elements."""
    result = []
    orig_rows = len(grid)
    orig_cols = len(grid[0])

    for i in range(new_rows):
        row = []
        for j in range(new_cols):
            orig_i = int(i / (new_rows / orig_rows))
            orig_j = int(j / (new_cols / orig_cols))
            row.append(grid[orig_i][orig_j])
        result.append(row)

    return result


def _downscale_grid_simple(grid: Grid, new_rows: int, new_cols: int) -> Grid:
    """Helper function to downscale a grid by choosing the least frequent value in the entire grid."""
    result = []
    orig_rows = len(grid)
    orig_cols = len(grid[0])

    grid_counts = {}
    for r in range(orig_rows):
        for c in range(orig_cols):
            val = grid[r][c]
            grid_counts[val] = grid_counts.get(val, 0) + 1

    for i in range(new_rows):
        row = []
        for j in range(new_cols):
            start_i = int(i * orig_rows / new_rows)
            end_i = int((i + 1) * orig_rows / new_rows)
            start_j = int(j * orig_cols / new_cols)
            end_j = int((j + 1) * orig_cols / new_cols)

            block_values = []
            for r in range(start_i, end_i):
                for c in range(start_j, end_j):
                    if r < orig_rows and c < orig_cols:
                        block_values.append(grid[r][c])

            if not block_values:
                row.append(0)
            else:
                if all(isinstance(x, int) for x in block_values):
                    min_count = float('inf')
                    chosen_value = block_values[0]
                    for val in block_values:
                        count = grid_counts.get(val, 0)
                        if count < min_count:
                            min_count = count
                            chosen_value = val
                    row.append(chosen_value)
                else:
                    row.append(sum(block_values) / len(block_values))

        result.append(row)

    return result


def scale_grid_nearest_neighbor_simple(grid: Grid, scale_factor: float) -> Grid:
    """
    Scale a grid using nearest neighbor interpolation (simpler but faster).
    """
    if not grid or not grid[0]:
        return []

    if scale_factor <= 0:
        raise ValueError("Scale factor must be positive")

    rows = len(grid)
    cols = len(grid[0])

    new_rows = int(rows * scale_factor)
    new_cols = int(cols * scale_factor)

    if new_rows == 0 or new_cols == 0:
        return []

    result = []
    for i in range(new_rows):
        row = []
        for j in range(new_cols):
            orig_i = min(int(i / scale_factor), rows - 1)
            orig_j = min(int(j / scale_factor), cols - 1)
            row.append(grid[orig_i][orig_j])
        result.append(row)

    return result


def downscale_grid_with_metadata(grid: Grid, target_rows: int, target_cols: int) -> Tuple[Grid, Metadata]:
    """
    Downscale grid while preserving metadata that maps original positions to downscaled cells.

    Returns:
        - downscaled grid
        - metadata[i][j] = list of (orig_r, orig_c, original_value)
    """
    orig_rows = len(grid)
    orig_cols = len(grid[0])
    downscaled_grid = []
    metadata: Metadata = [[[] for _ in range(target_cols)] for _ in range(target_rows)]

    for i in range(target_rows):
        row = []
        for j in range(target_cols):
            start_i = int(i * orig_rows / target_rows)
            end_i = int((i + 1) * orig_rows / target_rows)
            start_j = int(j * orig_cols / target_cols)
            end_j = int((j + 1) * orig_cols / target_cols)

            block = []
            for r in range(start_i, end_i):
                for c in range(start_j, end_j):
                    val = grid[r][c]
                    block.append(val)
                    metadata[i][j].append((r, c, val))

            if not block:
                row.append(0)
            else:
                if all(isinstance(x, int) for x in block):
                    counts = Counter(block)
                    chosen = min(counts.items(), key=lambda x: x[1])[0]
                    row.append(chosen)
                else:
                    row.append(sum(block) / len(block))
        downscaled_grid.append(row)

    return downscaled_grid, metadata


def upscale_with_metadata(transformed_grid: Grid, metadata: Metadata, original_shape: Tuple[int, int]) -> Grid:
    """
    Upscale a transformed downscaled grid back to original size using stored metadata.
    Each original cell is updated using the transformed value of the corresponding downscaled block.
    """
    orig_rows, orig_cols = original_shape
    upscaled_grid = [[0 for _ in range(orig_cols)] for _ in range(orig_rows)]

    for i in range(len(transformed_grid)):
        for j in range(len(transformed_grid[0])):
            new_val = transformed_grid[i][j]
            for r, c, _ in metadata[i][j]:
                upscaled_grid[r][c] = new_val

    return upscaled_grid
