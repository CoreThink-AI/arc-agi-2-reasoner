from typing import List, Union, Tuple, Dict, Any
from collections import Counter

Grid = List[List[Union[int, float]]]

# Enhanced metadata stores values and positions for each downscaled block
EnhancedMetadata = List[List[Dict[str, Any]]]


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
    """Scale a grid using nearest neighbor interpolation (simpler but faster)."""
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


def downscale_grid_with_metadata(grid: Grid, target_rows: int, target_cols: int) -> Tuple[Grid, EnhancedMetadata]:
    """
    Downscale grid while preserving metadata that stores original values and positions for each block.

    Returns:
        - downscaled grid
        - metadata[i][j] = {
              "positions": List of (r, c),
              "values": 2D block values from original grid
          }
    """
    orig_rows = len(grid)
    orig_cols = len(grid[0])
    downscaled_grid = []
    metadata: EnhancedMetadata = [[{} for _ in range(target_cols)] for _ in range(target_rows)]

    for i in range(target_rows):
        row = []
        for j in range(target_cols):
            start_i = int(i * orig_rows / target_rows)
            end_i = int((i + 1) * orig_rows / target_rows)
            start_j = int(j * orig_cols / target_cols)
            end_j = int((j + 1) * orig_cols / target_cols)

            block = []
            positions = []
            for r in range(start_i, end_i):
                row_block = []
                for c in range(start_j, end_j):
                    row_block.append(grid[r][c])
                    positions.append((r, c))
                block.append(row_block)

            # Store full block in metadata
            metadata[i][j] = {
                "positions": positions,
                "values": block
            }

            # Compute summary value for the block
            flat_vals = [val for row_b in block for val in row_b]
            if not flat_vals:
                row.append(0)
            elif all(isinstance(x, int) for x in flat_vals):
                counts = Counter(flat_vals)
                chosen = min(counts.items(), key=lambda x: x[1])[0]
                row.append(chosen)
            else:
                row.append(sum(flat_vals) / len(flat_vals))
        downscaled_grid.append(row)

    return downscaled_grid, metadata


def upscale_with_metadata(transformed_grid: Grid, downscaled_grid: Grid, metadata: EnhancedMetadata, original_shape: Tuple[int, int]) -> Grid:
    """
    Upscale a transformed downscaled grid back to original size using enhanced metadata.
    If a transformed value is unchanged from the original downscaled value, restore the original block.
    If changed, apply the transformed value to all original positions.
    """
    orig_rows, orig_cols = original_shape
    upscaled_grid = [[0 for _ in range(orig_cols)] for _ in range(orig_rows)]

    for i in range(len(transformed_grid)):
        for j in range(len(transformed_grid[0])):
            transformed_val = transformed_grid[i][j]
            original_val = downscaled_grid[i][j]
            positions = metadata[i][j]["positions"]
            original_block = metadata[i][j]["values"]

            if transformed_val == original_val:
                # Restore original block
                h = len(original_block)
                w = len(original_block[0]) if h > 0 else 0
                idx = 0
                for bi in range(h):
                    for bj in range(w):
                        r, c = positions[idx]
                        upscaled_grid[r][c] = original_block[bi][bj]
                        idx += 1
            else:
                # Fill block with transformed value
                for r, c in positions:
                    upscaled_grid[r][c] = transformed_val

    return upscaled_grid
