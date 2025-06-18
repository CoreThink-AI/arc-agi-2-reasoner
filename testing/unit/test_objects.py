import os
import json
from contextlib import redirect_stdout

os.environ["OPENAI_API_KEY"] = ""

from arc_agi.src.utils.visualization_utils import read_json_as_string
from arc_agi.src.objects.base import Grid, BaseObject


VALID_JSON_PATH = r"C:\Users\Anugyan\PycharmProjects\ARC-AGI-2-CT\arc-agi-2-reasoner\testing\samples\tasks_20_40\valid_tasks.json"
OUTPUT_FOLDER = r"C:\Users\Anugyan\PycharmProjects\ARC-AGI-2-CT\arc-agi-2-reasoner\testing\samples\tasks_20_40"
DATA_FOLDER = r"C:\Users\Anugyan\PycharmProjects\ARC-AGI-2-CT\arc-agi-2-reasoner\data"

# Load all task file paths from JSON
with open(VALID_JSON_PATH, "r") as f:
    task_paths = json.load(f)["valid_tasks"]  # This should be a list of strings (file paths)

# Process each task
for task_path in task_paths:
    try:
        output_path = os.path.join(OUTPUT_FOLDER, f"{task_path}.txt")

        task_data = json.loads(read_json_as_string(os.path.join(DATA_FOLDER, f"{task_path}.json")))

        # You can customize which grid to pick
        train_test = ["train", "test"]
        input_output = ["input", "output"]
        with open(output_path, "w", encoding="utf-8") as f:
            print(f"Task: {task_path}")
            for i in train_test:
                for j in range(len(task_data[i])):
                    for k in input_output:
                        grid_data = task_data[i][j][k]
                        x = Grid(grid_data)
                        print(f"{i} {j + 1} {k}")
                        print(x)
                        print("\n")
                        x.visualize()
                        objects = x.find_objects_in_grid('openai', 'gpt-4.1-nano-2025-04-14', 0.0, 4096)
                        objects = x.objects_concatenator('openai', 'gpt-4.1-nano-2025-04-14', 0.0, 4096)
                        print("background:", x.background_color)
                        for obj in objects:
                            print("\n")
                            print(obj)
                            y = BaseObject(x.to_list(), obj)
                            y.visualize()
                            # object_metadata = y.to_dict('openai', 'gpt-4.1-nano-2025-04-14', 0.0, 4096)
                            # print(object_metadata)
    except Exception as e:
        print(f"Error processing {task_path}: {e}")

"""
cd arc-agi-2-reasoner
python -m testing.unit.test_objects

"""

