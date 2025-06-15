"""
Base object classes for ARC-AGI object analysis.
This module defines the base classes for analyzing objects in ARC-AGI problems.
"""

from typing import List, Tuple, Dict, Any, Set
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from .llm_inference import call_llm
from .visualize_objects import find_objects
from .llm_reasoner import prepare_llm_prompt


@dataclass(frozen=True)
class Coordinates:
    """
    Represents a 2D coordinate in the grid.

    Attributes:
        x (int): Row index (0-based)
        y (int): Column index (0-based)

        Grid coordinates use 0-based indexing, meaning the top-left cell is (0, 0)
        A coordinate (row, col) = (4, 10) refers to the cell in the 5th row from the top and the 11th column from the left
        To determine whether two cells (a, b) and (c, d) are adjacent (including diagonals), apply the rule:
        (a - c)^2 + (b - d)^2 <= 2
    """
    x: int  # row index
    y: int  # column index

    def to_tuple(self) -> Tuple[int, int]:
        """Return the coordinate as a tuple (x, y)."""
        return (self.x, self.y)

    @classmethod
    def from_tuple(cls, coords: Tuple[int, int]) -> 'Coordinates':
        """Create a Coordinates instance from a tuple (x, y)."""
        return cls(*coords)

    def is_adjacent_to(self, other: 'Coordinates') -> bool:
        """
        Check if another coordinate is adjacent to this one,
        including diagonals, using the rule:
            (x1 - x2)^2 + (y1 - y2)^2 <= 2
        """
        dx = self.x - other.x
        dy = self.y - other.y
        return dx * dx + dy * dy <= 2

    def distance_squared_to(self, other: 'Coordinates') -> int:
        """Return the squared Euclidean distance to another coordinate."""
        dx = self.x - other.x
        dy = self.y - other.y
        return dx * dx + dy * dy


class Grid:
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the Grid object from a 2D list. Raises ValueError if not 2D.
        """
        if not all(isinstance(row, list) for row in grid):
            raise ValueError("Grid must be a 2D list of lists.")
        if not all(len(row) == len(grid[0]) for row in grid):
            raise ValueError("All rows in the grid must have the same length.")
        self.grid = np.array(grid, dtype=int)
        if self.grid.ndim != 2:
            raise ValueError("Grid must be 2D.")

        self.ARC_COLORS = [
            "#000000",  # 0 = black (background)
            "#0074D9",  # 1 = blue
            "#2ECC40",  # 2 = green
            "#FF4136",  # 3 = red
            "#FFDC00",  # 4 = yellow
            "#AAAAAA",  # 5 = gray
            "#F012BE",  # 6 = magenta
            "#FF851B",  # 7 = orange
            "#870C25",  # 8 = dark red
            "#B10DC9",  # 9 = purple
        ]

    @classmethod
    def from_numpy(cls, np_grid: np.ndarray) -> 'Grid':
        """
        Create a Grid object from a 2D numpy array.
        """
        if not isinstance(np_grid, np.ndarray):
            raise TypeError("Input must be a numpy ndarray.")
        if np_grid.ndim != 2:
            raise ValueError("NumPy array must be 2D.")
        return cls(np_grid.tolist())

    def to_numpy(self) -> np.ndarray:
        """
        Return a copy of the grid as a NumPy array.
        """
        return self.grid.copy()

    def to_list(self) -> List[List[int]]:
        """
        Return the grid as a list of lists.
        """
        return self.grid.tolist()

    def shape(self) -> tuple:
        """
        Return the shape of the grid as (rows, columns).
        """
        return self.grid.shape

    def __str__(self) -> str:
        """
        Pretty string representation of the grid.
        """
        return "\n".join(" ".join(str(cell) for cell in row) for row in self.to_list())

    def visualize(self) -> None:
        """
        Display the grid using matplotlib with color mapping.
        """

        fig, ax = plt.subplots(figsize=(5, 5))
        cmap = plt.matplotlib.colors.ListedColormap(self.ARC_COLORS)
        ax.imshow(self.grid, cmap=cmap, vmin=0, vmax=9)
        ax.set_title("Grid Visualization")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, which='both', color='#636363', linestyle='-', linewidth=0.5)
        ax.set_xticks(np.arange(-.5, self.grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.grid.shape[0], 1), minor=True)
        plt.show()

    def find_background(self, provider: str, model: str, temperature: float, max_tokens: int) -> int:
        """
        Use an LLM to determine the background color and its coordinates in the grid.
        Returns:
            int: background color of the grid, -1 if no specific background color found
        """
        self.background_color = 0
        try:
            # Convert grid to string format
            grid_str = "\n".join(" ".join(str(cell) for cell in row) for row in self.to_list())

            # Create prompt
            prompt = (
                "You are given a 2D grid of integers from 0 to 9.\n"
                "Each integer represents a color. 0 is usually black and often represents the background. But that is "
                "not mandatory. A different color can also constitute the background.\n"
                "Some colored cells (non-background) form objects in this grid, but you are not given their coordinates"
                "The cells which logically and intuitively are not a part of these objects form the background.\n"
                "Your task is to:\n"
                "-  Identify the background color used in the grid.\n"
                "It is important to remember that there can be cases where no particular color seems to form the "
                "background. In such cases, you can return '-1' as an output"
                "Output format should be a JSON object like: {\"background_color\": 0}\n"
                "Respond with the JSON object. Add explanations after that.\n\n"
                "Here's the grid:\n"
                f"Grid:\n{grid_str}"
            )

            # Call LLM
            response = call_llm(
                provider=provider,
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            self.background_color = int(response.split('"background_color":')[1].split('}')[0].strip())
            return self.background_color

        except Exception as e:
            print(f"[LLM background detection error] {e}")
            self.background_color = -1
            return self.background_color

    def find_objects_in_grid(self) -> List[Set[Tuple[int, int]]]:
        """
            Rule-based method to find objects in the grid
            :returns List of objects found
        """
        self.objects = find_objects(self.grid.tolist())
        return self.objects

    def find_objects_reasoner(self, provider: str, model: str, temperature: float, max_tokens: int) -> str:
        """
            Reasoning layer for the objects detected
            :returns Reasoning from the LLM in string format
        """
        self.reasoning = ""
        try:
            add_prompt = f"The background color for this grid is {self.background_color}. The cells in this color can be" \
                         f"considered a part of the background and most probably not any object"
            prompt = prepare_llm_prompt(self.grid.tolist(), self.objects, add_prompt)

            # Call the LLM
            self.reasoning = call_llm(
                provider=provider,
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return self.reasoning
        except Exception as e:
            print(f"[LLM reasoning error] {e}")
            return self.reasoning


class BaseObject:
    """
    Base class for all objects in ARC-AGI problems.
    This class provides common functionality for object analysis.
    """

    def __init__(self, grid: List[List[int]], coordinates: Set[Tuple[int, int]]):
        """
        Initialize a base object.

        Args:
            grid (np.ndarray): The full grid containing the object.
            coordinates (Set[Tuple[int, int]]): Set of (x, y) tuples making up the object.
        """
        self.grid = Grid(grid).to_list()
        self.coordinates = [Coordinates.from_tuple(t) for t in coordinates]
        self._calculate_bounds()
        self._calculate_centroid()
        self._analyze_colors()
        self.ARC_COLORS = [
            "#000000",  # 0 = black (background)
            "#0074D9",  # 1 = blue
            "#2ECC40",  # 2 = green
            "#FF4136",  # 3 = red
            "#FFDC00",  # 4 = yellow
            "#AAAAAA",  # 5 = gray
            "#F012BE",  # 6 = magenta
            "#FF851B",  # 7 = orange
            "#870C25",  # 8 = dark red
            "#B10DC9",  # 9 = purple
        ]
        # Create a blank (black) grid
        self.masked_grid = [[0 for _ in row] for row in self.grid]

        # Copy original color values for object's coordinates
        for coord in self.coordinates:
            x, y = coord.x, coord.y
            self.masked_grid[x][y] = self.grid[x][y]

    def visualize(self) -> None:
        """
        Visualize the object by showing only its coordinates with original colors,
        setting all other grid cells to black (0).
        """
        # Convert to numpy array for plotting
        arr = np.array(self.masked_grid)
        cmap = plt.matplotlib.colors.ListedColormap(self.ARC_COLORS)

        # Plot
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(arr, cmap=cmap, vmin=0, vmax=9)
        ax.set_title("Object Visualization")
        ax.set_xticks([])
        ax.set_yticks([])

        # Optional: show gridlines
        ax.grid(True, which='both', color='#636363', linestyle='-', linewidth=0.5)
        ax.set_xticks(np.arange(-.5, arr.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, arr.shape[0], 1), minor=True)

        plt.show()

    def _calculate_bounds(self) -> None:
        """Calculate the bounding box of the object."""
        x_coords = [coord.x for coord in self.coordinates]
        y_coords = [coord.y for coord in self.coordinates]

        self.x1 = min(x_coords)
        self.x2 = max(x_coords)
        self.y1 = min(y_coords)
        self.y2 = max(y_coords)

        self.width = self.x2 - self.x1 + 1
        self.height = self.y2 - self.y1 + 1

    def _calculate_centroid(self) -> None:
        """Calculate the centroid of the object."""
        x_sum = sum(coord.x for coord in self.coordinates)
        y_sum = sum(coord.y for coord in self.coordinates)
        n = len(self.coordinates)

        self.centroid = Coordinates(x=x_sum // n, y=y_sum // n)

    def _analyze_colors(self) -> None:
        """Analyze the colors present in the object."""
        self.colors = set()
        for coord in self.coordinates:
            color = self.grid[coord.x][coord.y]
            self.colors.add(int(color))

        self.colors = sorted(list(self.colors))
        self.is_single_color = len(self.colors) == 1
        self.color = self.colors[0] if self.is_single_color else None

    def _analyze_shape(self, provider: str, model: str, temperature: float, max_tokens: int) -> None:
        """Get insights on the shape of the object using an LLM."""
        try:
            # Convert grid to string format
            grid_str = "\n".join(" ".join(str(cell) for cell in row) for row in self.masked_grid)

            # Convert coordinates to string
            coord_str = ", ".join(f"({c.x}, {c.y})" for c in self.coordinates)

            # Create prompt for LLM
            prompt = (
                f"You are given a rectangular 2D grid of shape {self.get_grid_size()}.\n"
                "Each pixel is represented by an integer between 0 and 9, where 0 means black (background), and other "
                "values are colors. The pixels in colors represent an object\n"
                "Here is the full grid:\n"
                f"{grid_str}\n\n"
                "The object is defined by the following coordinates:\n"
                f"{coord_str}\n\n"
                "Your task is to analyse the shape of this object, remember that the object can also be irregular and"
                "have cavities inside it.\n"
                "Describe the shape of this object in a single statement, try to include as much detail as possible\n"
            )

            # Call the LLM
            self.shape_type = call_llm(
                provider=provider,
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )

        except Exception as e:
            print(f"[LLM shape analysis error] {e}")
            self.shape_type = "unknown"

    def get_placement(self) -> Coordinates:
        """Get the placement coordinate (center of bounding box)."""
        return Coordinates(x=self.x1 + self.width // 2, y=self.y1 + self.height // 2)

    def get_size(self) -> int:
        """Get the size of the object (number of coordinates)."""
        return len(self.coordinates)

    def get_grid_size(self) -> Tuple[int, int]:
        """Get the size of the object's grid (width, height)."""
        return (self.width, self.height)

    def get_cavities(self, provider: str, model: str, temperature: float, max_tokens: int) -> List[List[Tuple]]:
        """Detect enclosed cavities within the object using LLM assistance."""
        try:
            # Convert the full grid into string form
            grid_str = "\n".join(" ".join(str(cell) for cell in row) for row in self.masked_grid)
            coord_str = ", ".join(f"({c.x}, {c.y})" for c in self.coordinates)

            # Prompt to the LLM
            prompt = (
                f"You are given a rectangular 2D grid of shape {self.get_grid_size()}.\n"
                "Each pixel is represented by an integer between 0 and 9, where 0 means black (background), and other "
                "values are colors. The pixels in colors represent an object\n"
                "Here is the full grid:\n"
                f"{grid_str}\n\n"
                "The object is defined by the following coordinates:\n"
                f"{coord_str}\n\n"
                f"Here is the analysis of the shape of the object: {self.shape_type}\n"
                "Your task is to find out the cavities inside this object and return the coordinates that constitute "
                "the object.\n"
                "Return the output in this format: \n"
                "Cavity 1 : <List of coordinates of Cavity 1>\n"
                "Cavity 2 : <List of coordinates of Cavity 2 >\n"
                ".....................\n"
                "Cavity n : <List of coordinates of Cavity n>\n"
                "where 'n' is the number of cavities you detected. Return the coordinates only, don't give any explanation. "
            )

            # Call the LLM
            raw_response = call_llm(
                provider=provider,
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            import re
            # Extract all lists of coordinate tuples
            cavities = []
            pattern = re.compile(r"\[\((.*?)\)\]")  # Matches stuff inside one set of square brackets

            matches = re.findall(r"\[\s*(.*?)\s*\]", raw_response)

            for match in matches:
                coords = [
                    tuple(map(int, coord.strip("() ").split(",")))
                    for coord in match.split("),") if coord.strip()
                ]
                cavities.append(coords)
            return cavities

        except Exception as e:
            print(f"[LLM cavity detection error] {e}")
            return []

    def get_bottom_left(self) -> Coordinates:
        """Get the bottom-left coordinate of the object."""
        return Coordinates(self.x1, self.y2)

    def get_top_right(self) -> Coordinates:
        """Get the top-right coordinate of the object."""
        return Coordinates(self.x2, self.y1)

    def to_dict(self, provider: str, model: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Convert object properties to a dictionary."""
        self._analyze_shape(provider, model, temperature, max_tokens)
        self.get_cavities(provider, model, temperature, max_tokens)
        return {
            "grid": self.grid,
            "colors": self.colors,
            "color": self.color,
            "size": self.get_size(),
            "grid_size": self.get_grid_size(),
            "type": self.__class__.__name__,
            "coordinates": [coord.to_tuple() for coord in self.coordinates],
            "shape": self.shape_type,
            "x1": self.x1,
            "x2": self.x2,
            "y1": self.y1,
            "y2": self.y2,
            "placement": self.get_placement().to_tuple(),
            "centroid": self.centroid.to_tuple(),
            "cavities": [cavity for cavity in self.get_cavities(provider, model, temperature, max_tokens)],
            "bottom_left": self.get_bottom_left().to_tuple(),
            "top_right": self.get_top_right().to_tuple()
        }



# task_json_path = input("Enter path of task json: ")
# import json
# from visualize import read_json_as_string
# # visualize
# task_data = json.loads(read_json_as_string(task_json_path))
# # Pick one of the grids (e.g., first one)
# print("Enter space separated grid params (ex. train 1 input) which means 1st grid of the train sample's input")
# grid_params = input("Enter space separated grid params: ").split(" ")
# grid = task_data[grid_params[0]][int(grid_params[1]) - 1][grid_params[2]]
# x = Grid(grid)
# x.visualize()
# import os
# os.environ["OPENAI_API_KEY"] = ""
# print("background: ", x.find_background('openai', 'gpt-4.1-2025-04-14', 0.0, 4096))
# from visualize_objects import find_objects
# objects = find_objects(x.to_list())
# for i in objects:
#     print(i)
# y_ = objects[2]
# y = BaseObject(x.to_list(), y_)
# y.visualize()
# print(y.to_dict('openai', 'gpt-4.1-2025-04-14', 0.0, 4096))




