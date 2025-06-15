import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Union
import os

# Define a simple colormap for ARC colors (0–9)
ARC_COLORS = [
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


def plot_grid(grid: List[List[int]], title: str, ax: plt.Axes):
    """Plot a single grid using matplotlib."""
    arr = np.array(grid)
    cmap = plt.matplotlib.colors.ListedColormap(ARC_COLORS)
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=9)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add grid lines with dark grey color
    ax.grid(True, which='both', color='#636363', linestyle='-', linewidth=0.5)
    ax.set_xticks(np.arange(-.5, arr.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, arr.shape[0], 1), minor=True)
    ax.set_xticks([])
    ax.set_yticks([])


def visualize_task(task: Union[Dict, str]):
    """
    Visualizes an ARC task from a dictionary or a JSON file path.

    Args:
        task: either a dictionary containing the task or a string path to a JSON file
    """
    if isinstance(task, str):
        with open(task, 'r') as f:
            task = json.load(f)

    num_train = len(task['train'])
    num_test = len(task['test'])

    # Each pair has input and output, so we plot 2 per row
    fig, axes = plt.subplots(
        num_train + num_test, 2,
        figsize=(6, 3 * (num_train + num_test))
    )

    if (num_train + num_test) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, pair in enumerate(task['train']):
        plot_grid(pair['input'], f"Train {i + 1} Input", axes[i, 0])
        plot_grid(pair['output'], f"Train {i + 1} Output", axes[i, 1])

    for j, pair in enumerate(task['test']):
        idx = num_train + j
        plot_grid(pair['input'], f"Test {j + 1} Input", axes[idx, 0])
        plot_grid(pair['output'], f"Test {j + 1} Output", axes[idx, 1])

    plt.tight_layout()
    plt.show()


def read_json_as_string(file_path: str) -> str:
    """
    Reads a JSON file and returns its contents as a string.

    Args:
        file_path: Path to the JSON file.

    Returns:
        A string representation of the JSON file's contents.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def rename_json_files(directory: str):
    """
    Renames all .json files in the given directory to the format task_1.json, task_2.json, ...

    Args:
        directory: Path to the directory containing .json files.
    """
    # List and filter only .json files
    json_files = [f for f in os.listdir(directory) if f.lower().endswith('.json')]
    json_files.sort()  # Ensure consistent order

    for i, filename in enumerate(json_files, start=1):
        old_path = os.path.join(directory, filename)
        new_name = f"task_{i}.json"
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name}")

    print("\nRenaming completed.")

# rename_json_files(r"C:\Users\Anugyan\PycharmProjects\ARC-AGI-2-CT\ARC-AGI-2\data\training")


# # Paste your JSON string
# task_json_path = r"C:\Users\Anugyan\PycharmProjects\ARC-AGI-2-CT\ARC-AGI-2\data\training\task_9.json"
#
#
# task_data = json.loads(read_json_as_string(task_json_path))
# visualize_task(task_data)
