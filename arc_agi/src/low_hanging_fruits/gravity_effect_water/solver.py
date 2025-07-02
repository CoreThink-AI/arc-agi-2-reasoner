from typing import List, Tuple, Dict
import json
import os
from collections import Counter

from arc_agi.src.utils.visualization_utils import read_json_as_string
from arc_agi.src.objects.base import Grid, BaseObject
from arc_agi.src.utils.llm_utils import call_llm


class Solver:
    def __init__(self, task_data):
        self.task_data = task_data
        self.train = []
        self.test = []

        for i in range(len(task_data["train"])):
            self.train.append([task_data["train"][i]["input"], task_data["train"][i]["output"]])

        for j in range(len(task_data["test"])):
            self.test.append([task_data["test"][j]["input"], task_data["test"][j]["output"]])

    def find_objects_metadata(
        self,
        input_grid: Grid,
        output_grid: Grid
    ) -> Tuple[List, List[Dict], List, List[Dict]]:

        input_objects = input_grid.find_objects_in_grid('anthropic', 'claude-opus-4-20250514', 0.0, 32000)
        output_objects = output_grid.find_objects_in_grid('anthropic', 'claude-opus-4-20250514', 0.0, 32000)
        input_objects_metadata = []
        output_objects_metadata = []
        for input_obj in input_objects:
            input_obj = BaseObject(input_grid.to_list(), input_obj)
            input_obj_metadata = {"color": input_obj.color, "size": input_obj.get_size() , "grid_size": input_obj.get_grid_size(), "centroid": input_obj.centroid.to_tuple()}
            input_objects_metadata.append(input_obj_metadata)

        for output_obj in output_objects:
            output_obj = BaseObject(output_grid.to_list(), output_obj)
            output_obj_metadata = {"color": output_obj.color, "size": output_obj.get_size() , "grid_size": output_obj.get_grid_size(), "centroid": output_obj.centroid.to_tuple()}
            output_objects_metadata.append(output_obj_metadata)

        return input_objects, input_objects_metadata, output_objects, output_objects_metadata

    def gravity_label_from_object_centroids(self, input_grid: Grid, output_grid: Grid) -> Tuple[str, int]:
        (
            input_objects,
            input_metadata,
            output_objects,
            output_metadata
        ) = self.find_objects_metadata(input_grid, output_grid)

        # Helper: serialize metadata for easy comparison
        def serialize(obj_dict):
            return str({
                "color": obj_dict.get("color"),
                "size": obj_dict.get("size"),
                "grid_size": obj_dict.get("grid_size"),
                "centroid": obj_dict.get("centroid")
            })

        input_serialized = {serialize(obj): obj for obj in input_metadata}
        output_serialized = {serialize(obj): obj for obj in output_metadata}

        # Find uncommon objects
        input_keys = set(input_serialized.keys())
        output_keys = set(output_serialized.keys())

        input_only_keys = input_keys - output_keys
        output_only_keys = output_keys - input_keys

        uncommon_output_colors = [output_serialized[k]['color'] for k in output_only_keys]
        unique_colors = set(uncommon_output_colors)

        if len(unique_colors) == 1:
            color_of_uncommon_objects = unique_colors.pop()
        else:
            color_of_uncommon_objects = -1

        # If both sets are empty, all objects matched → no uncommon objects
        if not input_only_keys and not output_only_keys:
            return 'B', color_of_uncommon_objects

        # Compare y-coordinate of centroids: input vs output
        input_centroids = [input_serialized[k]['centroid'][0] for k in input_only_keys]
        output_centroids = [output_serialized[k]['centroid'][0] for k in output_only_keys]

        if all(out_y > min(input_centroids) for out_y in output_centroids):
            return 'A', color_of_uncommon_objects
        else:
            return 'B', color_of_uncommon_objects

    def prompt_builder(self, test_input: List[List[int]], water_color: int) -> str:
        prompt = (
            f"You are given examples of input-output grid transformations. Each grid is a 2D array of integers from 0 to 9, "
            f"representing different colors. The transformation in these examples involves water (color {water_color}) falling "
            f"under gravity. All other colors represent either background or static objects that do not move.\n\n"
            "Please follow these physics-inspired rules that govern how water (color "
            f"{water_color}) behaves:\n\n"
            "1. Water falls **straight down** if there is no object or water below it.\n"
            "2. If water hits a non-water object while falling, it will attempt to **flow left and right** over it.\n"
            "3. Water **spreads sideways** along the top surface of objects, and continues falling down from the edges.\n"
            "4. Water can **accumulate in pits or crevices**, forming puddles until they overflow.\n"
            "5. Water only moves **downward or sideways**, never upward.\n"
            "6. Water never displaces existing objects — it flows around or over them.\n"
            "7. The background and non-water object colors are static and remain unchanged.\n\n"
            "Below are training examples. Each includes an Input Grid and its corresponding Output Grid.\n\n"
        )

        for idx, (inp, out) in enumerate(self.train):
            input_grid = Grid(inp)
            output_grid = Grid(out)
            input_objects, input_metadata, output_objects, output_metadata = self.find_objects_metadata(input_grid,
                                                                                                        output_grid)
            background_color = input_grid.find_background('anthropic', 'claude-opus-4-20250514', 0.0, 32000)

            prompt += f"### Training Example {idx + 1}:\n"
            prompt += f"The background color is {background_color}.\n"
            prompt += "Input Grid:\n" + "\n".join(" ".join(str(cell) for cell in row) for row in inp) + "\n"
            prompt += f"Detected objects: {input_metadata}. Water is represented by color {water_color}.\n"
            prompt += "Output Grid:\n" + "\n".join(" ".join(str(cell) for cell in row) for row in out) + "\n"
            prompt += f"Detected objects: {output_metadata}. Notice how the water has fallen under gravity.\n\n"

        test_grid = Grid(test_input)
        background_color_test = test_grid.find_background('anthropic', 'claude-opus-4-20250514', 0.0, 32000)
        test_objects, test_metadata, _, _ = self.find_objects_metadata(test_grid, Grid([[0]]))  # dummy output

        prompt += "### Test Input Grid:\n"
        prompt += f"The background color is {background_color_test}.\n"
        prompt += "\n".join(" ".join(str(cell) for cell in row) for row in test_input) + "\n"
        prompt += f"Detected objects: {test_metadata}. Water is represented by color {water_color}.\n"
        prompt += "### Predict the Output Grid in the same format. Do not explain, just output the grid:\n"

        return prompt

    def predict_output_grid(self, prompt):
        output_grid = call_llm(
            provider='anthropic',
            prompt=prompt,
            model='claude-opus-4-20250514',
            temperature=0.0,
            max_tokens=32000
        )
        return output_grid

# run
task_data = json.loads(read_json_as_string(r"C:\Users\Anugyan\PycharmProjects\ARC-AGI-2-CT\arc-agi-2-reasoner\data\28a6681f.json"))

solver = Solver(task_data)

count_A, count_B = 0, 0
water_color = []
for i in range(len(solver.train)):
    input_grid = Grid(solver.train[i][0])
    output_grid = Grid(solver.train[i][1])

    count, color = solver.gravity_label_from_object_centroids(input_grid, output_grid)
    if count == "A":
        count_A += 1
        water_color.append(color)
    else:
        count_B += 1

if count_A > count_B:
    counter = Counter(water_color)

    # Get the most common element
    water_color, count = counter.most_common(1)[0]
    test_grid = solver.test[0][0]
    prompt = solver.prompt_builder(test_grid, water_color)
    print(solver.predict_output_grid(prompt))




"""
set OPENAI_API_KEY=sk-...
set CEREBRAS_API_KEY=sk-...
set ANTHROPIC_API_KEY=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
set TOGETHER_API_KEY=sk-....
set AZURE_OPENAI_API_KEY=sk-.....
set AZURE_OPENAI_ENDPOINT=sdfsd
python -m arc_agi.src.low_hanging_fruits.gravity_effect_water.solver

"""

