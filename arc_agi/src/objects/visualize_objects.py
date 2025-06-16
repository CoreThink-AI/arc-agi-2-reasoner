import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Set
from collections import deque


def find_objects(grid: List[List[int]]) -> List[Set[Tuple[int, int]]]:
    """
    Finds connected components (objects) in a grid.
    Cells with 0 are ignored. Connectivity is 4-directional.
    Returns a list of sets of coordinates (row, col) for each object.
    """
    n_rows, n_cols = len(grid), len(grid[0])
    visited = [[False] * n_cols for _ in range(n_rows)]
    objects = []

    def bfs(r: int, c: int, value: int) -> Set[Tuple[int, int]]:
        queue = deque([(r, c)])
        obj_cells = set()
        visited[r][c] = True

        while queue:
            x, y = queue.popleft()
            obj_cells.add((x, y))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < n_rows and 0 <= ny < n_cols:
                    if not visited[nx][ny] and grid[nx][ny] == value and grid[nx][ny] != 0:
                        visited[nx][ny] = True
                        queue.append((nx, ny))
        return obj_cells

    for i in range(n_rows):
        for j in range(n_cols):
            if not visited[i][j] and grid[i][j] != 0:
                obj = bfs(i, j, grid[i][j])
                if obj:
                    objects.append(obj)

    return objects


def visualize_objects(grid: List[List[int]], objects: List[Set[Tuple[int, int]]]):
    """
    Visualizes the grid with objects highlighted using unique colors and dark grey grid lines.
    """
    display_grid = np.zeros((len(grid), len(grid[0]), 3), dtype=float)
    cmap = plt.cm.get_cmap('tab20', len(objects))

    for idx, obj in enumerate(objects):
        color = cmap(idx)[:3]
        for x, y in obj:
            display_grid[x, y] = color

    plt.figure(figsize=(8, 8))
    plt.imshow(display_grid)
    plt.title(f"{len(objects)} object(s) detected")
    
    # Add grid lines
    plt.grid(True, which='both', color='#636363', linestyle='-', linewidth=1)
    plt.xticks(np.arange(-0.5, len(grid[0]), 1), [])
    plt.yticks(np.arange(-0.5, len(grid), 1), [])
    
    plt.show()


def process_grid(grid: List[List[int]]) -> List[Set[Tuple[int, int]]]:
    objects = find_objects(grid)
    visualize_objects(grid, objects)
    return objects


# # Suppose you loaded a grid from JSON like this:
# with open(r"C:\Users\Anugyan\PycharmProjects\ARC-AGI-2-CT\ARC-AGI-2\data\training\task_4.json", 'r') as f:
#     task_data = json.load(f)
#
# # Pick one of the train grids (e.g., first one)
# grid = task_data['train'][1]['input']
# objects = process_grid(grid)
#
# # Print the coordinates of the objects
# for i, obj in enumerate(objects, 1):
#     print(f"Object {i}: {sorted(obj)}")
