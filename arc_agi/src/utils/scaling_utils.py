from typing import List, Union

# TODO: Update this function to be able to scale across x and y axes with different scale factors


def scale_grid_simple(grid: List[List[Union[int, float]]], scale_factor: float) -> List[List[Union[int, float]]]:
    """
    Scale a grid by a given factor using pure Python (no numpy required).
    
    Args:
        grid: 2D array (list of lists) of numbers
        scale_factor: Factor to scale the grid by (e.g., 2.0 doubles the size, 0.5 halves it)
    
    Returns:
        Scaled grid as a 2D array
    
    Examples:
        >>> grid = [[1, 2], [3, 4]]
        >>> scale_grid_simple(grid, 2.0)
        [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
        
        >>> scale_grid_simple(grid, 0.5)
        [[1]]
    """
    if not grid or not grid[0]:
        return []
    
    if scale_factor <= 0:
        raise ValueError("Scale factor must be positive")
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Calculate new dimensions
    new_rows = int(rows * scale_factor)
    new_cols = int(cols * scale_factor)
    
    if new_rows == 0 or new_cols == 0:
        return []
    
    # For upscaling (scale_factor > 1)
    if scale_factor > 1:
        return _upscale_grid_simple(grid, new_rows, new_cols)
    # For downscaling (scale_factor < 1)
    else:
        return _downscale_grid_simple(grid, new_rows, new_cols)

def _upscale_grid_simple(grid: List[List[Union[int, float]]], new_rows: int, new_cols: int) -> List[List[Union[int, float]]]:
    """Helper function to upscale a grid by repeating elements."""
    result = []
    orig_rows = len(grid)
    orig_cols = len(grid[0])
    
    for i in range(new_rows):
        row = []
        for j in range(new_cols):
            # Map new position back to original grid
            orig_i = int(i / (new_rows / orig_rows))
            orig_j = int(j / (new_cols / orig_cols))
            row.append(grid[orig_i][orig_j])
        result.append(row)
    
    return result

def _downscale_grid_simple(grid: List[List[Union[int, float]]], new_rows: int, new_cols: int) -> List[List[Union[int, float]]]:
    """Helper function to downscale a grid by choosing the least frequent value in the entire grid."""
    result = []
    orig_rows = len(grid)
    orig_cols = len(grid[0])
    
    # First, count all values in the entire grid
    grid_counts = {}
    for r in range(orig_rows):
        for c in range(orig_cols):
            val = grid[r][c]
            grid_counts[val] = grid_counts.get(val, 0) + 1
    
    for i in range(new_rows):
        row = []
        for j in range(new_cols):
            # Calculate the range of original cells that map to this new cell
            start_i = int(i * orig_rows / new_rows)
            end_i = int((i + 1) * orig_rows / new_rows)
            start_j = int(j * orig_cols / new_cols)
            end_j = int((j + 1) * orig_cols / new_cols)
            
            # Extract the block of original cells
            block_values = []
            for r in range(start_i, end_i):
                for c in range(start_j, end_j):
                    if r < orig_rows and c < orig_cols:
                        block_values.append(grid[r][c])
            
            if not block_values:
                row.append(0)  # Default value if no cells in block
            else:
                # For integer grids, choose the value with the lowest count in the entire grid
                if all(isinstance(x, int) for x in block_values):
                    # Find the value in the block that has the lowest count in the entire grid
                    min_count = float('inf')
                    chosen_value = block_values[0]  # Default to first value
                    
                    for val in block_values:
                        count = grid_counts.get(val, 0)
                        if count < min_count:
                            min_count = count
                            chosen_value = val
                    
                    row.append(chosen_value)
                else:
                    # For float grids, take the mean
                    row.append(sum(block_values) / len(block_values))
        
        result.append(row)
    
    return result

def scale_grid_nearest_neighbor_simple(grid: List[List[Union[int, float]]], scale_factor: float) -> List[List[Union[int, float]]]:
    """
    Scale a grid using nearest neighbor interpolation (simpler but faster).
    
    Args:
        grid: 2D array (list of lists) of numbers
        scale_factor: Factor to scale the grid by
    
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
    
    result = []
    for i in range(new_rows):
        row = []
        for j in range(new_cols):
            # Map to original coordinates
            orig_i = min(int(i / scale_factor), rows - 1)
            orig_j = min(int(j / scale_factor), cols - 1)
            row.append(grid[orig_i][orig_j])
        result.append(row)
    
    return result