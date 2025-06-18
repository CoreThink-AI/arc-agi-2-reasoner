from typing import List, Set, Tuple
import re
from arc_agi.src.utils.llm_utils import call_llm


def are_adjacent(obj1: Set[Tuple[int, int]], obj2: Set[Tuple[int, int]]) -> bool:
    """
    Returns True if any pixel in obj1 is adjacent to any pixel in obj2 using the given adjacency rule.
    """
    for (a, b) in obj1:
        for (c, d) in obj2:
            if (a - c) ** 2 + (b - d) ** 2 <= 2:
                return True
    return False


def group_and_merge_adjacent_objects(objects: List[Set[Tuple[int, int]]]) -> List[Set[Tuple[int, int]]]:
    """
    Groups and merges adjacent objects. Returns a list of merged object sets (as sets of coordinates).
    """
    n = len(objects)
    visited = [False] * n
    merged_groups = []

    def dfs(i: int, current_coords: Set[Tuple[int, int]]):
        visited[i] = True
        current_coords.update(objects[i])
        for j in range(n):
            if not visited[j] and are_adjacent(objects[i], objects[j]):
                dfs(j, current_coords)

    for i in range(n):
        if not visited[i]:
            group_coords = set()
            dfs(i, group_coords)
            merged_groups.append(group_coords)

    return merged_groups


def prepare_llm_prompt(grid: List[List[int]], objects: List[Set[Tuple[int, int]]]) -> str:
    """
    Constructs a prompt for an LLM to evaluate the logical validity of detected objects in a grid.
    The prompt includes a description of the grid, a list of detected objects, and clear instructions for reasoning.
    """
    n_rows, n_cols = len(grid), len(grid[0]) if grid else 0

    prompt = (
        "You are a reasoning engine tasked with evaluating whether detected objects in a colored grid are logical "
        "from a human perceptual standpoint.\n\n"
        f"The grid below has size {n_rows} rows × {n_cols} columns. Each cell contains a value from '0' to '9', "
        "where each value represents a different color. Based on these colors, objects might be visualized in the grid.\n\n"
        "Your task is to assess whether the detected objects make sense.\n"
        "Importantly, diagonally adjacent cells can also belong to the same object. When the grid is viewed as a whole, "
        "a human should be able to perceive distinct objects based on visual structure and layout. You must also consider "
        "how the placement and orientation of objects affect their perceptual coherence.\n\n"
        "Here is the grid:\n"
    )

    # Add grid visualization
    prompt += "Grid:\n"
    for row in grid:
        prompt += " ".join(str(cell) for cell in row) + "\n"

    # Coordinate system explanation
    prompt += (
        "\nGrid coordinates start at (0, 0), with the top-left cell as the origin.\n"
        "A coordinate like (4, 10) refers to the 5th row from the top and 11th column from the left.\n"
        "Two cells (a, b) and (c, d) are considered adjacent if (a - c)^2 + (b - d)^2 <= 2. You must apply this rule "
        "strictly when determining adjacency between cells.\n"
    )

    # List the detected objects
    prompt += "\nThe following objects have been detected in the grid:\n"
    for idx, obj in enumerate(objects):
        sorted_coords = sorted(list(obj))
        prompt += f"Object {idx + 1} (size={len(obj)}): {sorted_coords}\n"

    # Reasoning task instructions
    prompt += (
        "\nPlease evaluate whether each of the above objects is a valid and logical object based on perceptual reasoning.\n"
        "You must reject objects that are highly irregular, fragmented, or chaotic, unless they clearly form a recognizable structure or pattern.\n"
        "Prioritize shape and spatial coherence over color uniformity. A valid object may contain multiple colors, especially if used in a structured, patterned, or symmetric way.\n"
        "Reject multi-color objects only when their layout appears chaotic or random, and the spatial structure lacks coherence.\n"
        "Valid objects should exhibit structural integrity, regularity, symmetry, or recognizable forms, and be interpretable as a perceptual whole.\n"
        "You should prefer rejecting objects that appear visually noisy or made of unrelated fragments, especially if the shape or color layout does not make sense.\n"
        "\nFor each object, provide:\n"
        "1. Whether the object is valid.\n"
        "2. Your reasoning for that decision.\n"
    )

    # Specify output format
    prompt += "\nReturn the output in the following format only:\n"
    prompt += """
                1. Object 1: <Valid / Not Valid> , <Reason>
                2. Object 2: <Valid / Not Valid> , <Reason>
                3. ...

                List of Valid Objects:
                1. Object X: [(x1, y1), (x2, y2), ...]
                2. Object Y: [(x3, y3), (x4, y4), ...]
                """

    return prompt


def extract_valid_objects(
    grid: List[List[int]],
    objects: List[Set[Tuple[int, int]]],
    provider: str,
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 4096
) -> List[Set[Tuple[int, int]]]:
    """
    Parses the LLM's output to extract only the valid objects from the "List of Valid Objects" section.

    Args:
        llm_output: The string output from the LLM.

    Returns:
        A list of sets, each containing (row, col) tuples representing a valid object.
    """
    llm_output = call_llm(provider, prepare_llm_prompt(grid, objects), model, temperature, max_tokens)
    print(llm_output)
    valid_objects = []

    # Find the "List of Valid Objects" section
    match = re.search(r"List of Valid Objects:\s*(.*)", llm_output, re.DOTALL)
    if not match:
        return valid_objects  # No valid objects listed

    section = match.group(1)

    # Match object lines like: "1. Object 2: [(3, 0), (3, 1), (3, 2), (3, 3)]"
    object_lines = re.findall(r"\d+\.\s*Object\s*\d+:\s*\[([^\]]+)\]", section)

    for line in object_lines:
        coords = re.findall(r"\((\d+),\s*(\d+)\)", line)
        object_coords = set((int(r), int(c)) for r, c in coords)
        valid_objects.append(object_coords)

    return valid_objects



