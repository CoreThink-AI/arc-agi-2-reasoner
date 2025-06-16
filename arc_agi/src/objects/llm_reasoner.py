import json
from typing import List, Tuple, Dict, Set
from openai import OpenAI
import os
from arc_agi.src.objects.visualize_objects import process_grid

with open(os.path.join(os.path.dirname(__file__), "object_types.json"), "r") as f:
    objects_data = json.load(f)


def prepare_llm_prompt(grid: List[List[int]], objects: List[Set[Tuple[int, int]]], add_prompt: str = "") -> str:
    """
    Constructs a prompt for an LLM to evaluate the logical validity of detected objects in a grid.
    The prompt includes a description of the grid size and the list of detected objects with their coordinates.
    """
    n_rows, n_cols = len(grid), len(grid[0]) if grid else 0

    prompt = (
        "You are a reasoning engine tasked with evaluating whether detected objects in a colored grid are logical "
        "from a human perceptual standpoint. \n\n"
        f"The grid below has size {n_rows} rows × {n_cols} columns. Each cell contains a value ranging from '0' to '9'"
        "where each value represents a different color. Based on these different colors, object(s) might be visualized"
        "in the grid. Your task is to verify whether these mentioned objects make sense. These objects can have regions"
        "of different colors, can we regular or irregular, can have cavities inside them and also can be embedded, and "
        "inside another object. The objects can be of varying sizes and can also have diagonal orientation and not only"
        "orthogonal orientation. This is very important to note that diagonally adjacent cells can also be part of one "
        "object. Overall when the grid visualized as a whole, these objects are such that a human will be"
        "able to see them and make out an object. Keep in mind you might have to consider the placement and orientation"
        "of different objects with respect to each other0. You have to think on all these links. Here's the grid: \n"

    )

    # Add grid visualization
    prompt += "Grid:\n"
    for row in grid:
        prompt += " ".join(str(cell) for cell in row) + "\n"

    prompt += (
        "\n"
        "Grid coordinates start at index 0, so the top-left cell is (0, 0).\n"
        "The grid is oriented in such a way that a coordinate (4, 10) means the cell is in the 5th (4+1) row from the"
        "top and in the 11th (10 + 1) column from the left "
        "Let's assume you have two cells with coordinates (a, b) and (c, d), then they will be considered adjacent if"
        "(a-c)**2 + (b-d)**2 <= 2. Apply this rule, when you want to see if two cells are adjacent."
        "Remember, this rule is a must"
    )

    # Add list of detected objects
    prompt += "\nThe following objects have been mentioned to be detected in the above grid:\n"

    for idx, obj in enumerate(objects):
        sorted_coords = sorted(list(obj))
        prompt += f"Object {idx+1} (size={len(obj)}): {sorted_coords}\n"

    prompt += (
        "\nPlease reason about whether each of these detected objects are valid and logical based on human-like "
        "symbolic interpretation and perceptual reasoning.\n"
        "Provide two things about each mentioned object: (1) whether you think it is a valid object based on your"
        "reasoning. (2) Your reasoning behind your answer to part one. \n"
        "In the end also provide the list of objects that you think are there in the grid, this might be a different"
        "list from what has been given to you, as you might think that a particular object is not a valid object based"
        "on the mentioned criteria or more importantly, you might think that a particular object is a part of a bigger"
        "object and they should be merged. Remember, that you are allowed to either select an object from the provided"
        "list of objects or merge two objects and create a new one if you think that's reasonable. You can't return any"
        "other objects apart from these two cases.\n"
        ""
    )
    # examples of object types
    prompt += (
        f"Here are a few examples of the possible types of objects that can be expected in the grid: \n"
        f"{objects_data} \n\n"
        f"These are some examples to give you and idea of the type of objects to expect in the grids. This is not an"
        f"exhaustive list and therefore, there can be other objects too."
    )
    # additional prompt
    prompt += add_prompt
    # example output format
    example_output = """
    Here's an example output for your reference:
    ### Object 1 Analysis:
    1. **Validity**: Yes, Object 1 is a valid object.
    2. **Reasoning**: 
       - Object 1 is composed of cells with the color '6'. The cells are connected in a way that forms a coherent, continuous shape. 
       - The shape resembles a zigzag or a diagonal line that moves from the top left to the bottom right, which is a recognizable pattern.
       - The object is visually distinct and does not appear to be part of a larger object, as it is isolated in the upper part of the grid.

    ### Object 2 Analysis:
    1. **Validity**: Yes, Object 2 is a valid object.
    2. **Reasoning**: 
       - Object 2 is composed of cells with the color '2'. The cells form a coherent, continuous shape that resembles a 'T' or a hammer-like structure.
       - The shape is distinct and separate from Object 1, located in the lower part of the grid.
       - The arrangement of the cells is such that it forms a recognizable pattern, making it a valid object.

    ### List of Objects in the Grid:
    1. Object 1: [(1, 1), (1, 2), (1, 3), (2, 1), (2, 4), (3, 2), (3, 5), (4, 3), (4, 6), (5, 4), (5, 5), (5, 6)]
    2. Object 2: [(7, 2), (7, 3), (7, 4), (8, 2), (8, 5), (9, 3), (9, 4), (9, 5)]
    """

    prompt += example_output

    return prompt


def query_llm_for_reasoning_openai(prompt: str, model: str = "gpt-4.1-2025-04-14") -> str:
    """
    Sends the prompt to the OpenAI API and returns the response text.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert visual reasoning agent."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=4096
    )
    return response.choices[0].message.content


def llm_reasoner(task_json_path, category, grid_number, grid_type):
    with open(task_json_path, 'r') as f:
        task_data = json.load(f)

    grid = task_data[category][grid_number - 1][grid_type]
    objects = process_grid(grid)

    # Create prompt
    prompt = prepare_llm_prompt(grid, objects)

    # Query the LLM
    llm_response = query_llm_for_reasoning_openai(prompt)

    print("LLM Response:")
    print(llm_response)


# llm_reasoner(9, 'train', 3, 'input')



