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


