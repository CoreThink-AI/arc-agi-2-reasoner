from arc_agi.src.objects.visualize import read_json_as_string
import json
import os
from arc_agi.src.objects.base import Grid, BaseObject
from arc_agi.src.objects.self_relations import SpatialRelations
from arc_agi.src.objects.visualize_objects import find_objects
from dotenv import load_dotenv
load_dotenv()
# visualize

task_json_path = "data/31f7f899.json"
task_data = json.loads(read_json_as_string(task_json_path))

# Pick one of the grids (e.g., first one)
print("Enter space separated grid params (ex. train 1 input) which means 1st grid of the train sample's input")
grid_params = input("Enter space separated grid params: ").split(" ")
grid = task_data[grid_params[0]][int(grid_params[1]) - 1][grid_params[2]]
x = Grid(grid)
x.visualize()

print("background: ", x.find_background('openai', 'gpt-4.1-2025-04-14', 0.0, 4096))

objects = find_objects(x.to_list())
print(objects)
y1 = BaseObject(x.to_list(), objects[2])
y1.visualize()
y2 = BaseObject(x.to_list(), objects[3])
y2.visualize()
sr = SpatialRelations(y1, y2)
sr.get_metadata('openai', 'gpt-4.1-2025-04-14', 0.0, 4096)
print(sr.to_dict())
