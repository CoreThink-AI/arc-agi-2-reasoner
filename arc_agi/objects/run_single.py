import json
from visualize import read_json_as_string, visualize_task
from visualize_objects import process_grid
from llm_reasoner import llm_reasoner

# Paste your JSON string
task_json_path = input("Enter path of task json: ")

# visualize
task_data = json.loads(read_json_as_string(task_json_path))
visualize_task(task_data)

# Pick one of the grids (e.g., first one)
print("Enter space separated grid params (ex. train 1 input) which means 1st grid of the train sample's input")
grid_params = input("Enter space separated grid params: ").split(" ")
grid = task_data[grid_params[0]][int(grid_params[1]) - 1][grid_params[2]]
objects = process_grid(grid)

# Print the coordinates of the objects
print("Coordinates inside the objects: ")
for i, obj in enumerate(objects, 1):
    print(f"Object {i}: {sorted(obj)}")

llm_reasoner(task_json_path, grid_params[0], int(grid_params[1]), grid_params[2])


