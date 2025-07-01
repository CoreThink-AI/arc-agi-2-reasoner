from typing import List, Tuple, Dict
import json
import os

from arc_agi.src.utils.visualization_utils import read_json_as_string
from arc_agi.src.objects.base import Grid, BaseObject


def objects_match(obj1, obj2) -> bool:
    return obj1 == obj2


def count_matching_objects(input_objs, output_objs) -> int:
    matched = 0
    used = set()
    for in_obj in input_objs:
        for i, out_obj in enumerate(output_objs):
            if i in used:
                continue
            if objects_match(in_obj, out_obj):
                matched += 1
                used.add(i)
                break
    return matched


def find_objects_metadata(
    input_grid: Grid,
    output_grid: Grid
) -> Tuple[List, List[Dict], List, List[Dict]]:
    input_objects = input_grid.find_objects_in_grid('anthropic', 'claude-opus-4-20250514', 0.0, 32000)
    output_objects = output_grid.find_objects_in_grid('anthropic', 'claude-opus-4-20250514', 0.0, 32000)
    input_objects_metadata = []
    output_objects_metadata = []
    for input_obj in input_objects:
        input_obj = BaseObject(input_grid.tolist(), input_obj)
        input_obj_metadata = input_obj.to_dict('anthropic', 'claude-opus-4-20250514', 0.0, 32000)
        input_objects_metadata.append(input_obj_metadata)

    for output_obj in output_objects:
        output_obj = BaseObject(output_grid.tolist(), output_obj)
        output_obj_metadata = output_obj.to_dict('anthropic', 'claude-opus-4-20250514', 0.0, 32000)
        output_objects_metadata.append(output_obj_metadata)

    return input_objects, input_objects_metadata, output_objects, output_objects_metadata


task_data = json.loads(read_json_as_string(r"C:\Users\Anugyan\PycharmProjects\ARC-AGI-2-CT\arc-agi-2-reasoner\data\28a6681f.json"))

train = []
test = []

for i in range(len(task_data["train"])):
    train.append([task_data["train"][i]["input"], task_data["train"][i]["output"]])

for j in range(len(task_data["test"])):
    train.append([task_data["test"][j]["input"], task_data["train"][j]["output"]])

flag = True
for i in train:
    train_grid = Grid(i[0])
    test_grid = Grid(i[1])
    print("train: ", train_grid)
    print("test: ", test_grid)

    if not find_objects_metadata(train_grid, test_grid):
        flag = False
        break

print(flag)
