from typing import List, Tuple, Dict, Any, Set
from arc_agi.objects.base import Grid, BaseObject
from arc_agi.objects.llm_inference import  call_llm


class SpatialRelations:
    def __init__(self, object_a: BaseObject, object_b: BaseObject):
        self.object_a = object_a
        self.object_b = object_b
        self.object_a_metadata = None
        self.object_b_metadata = None
        self.provider = None
        self.model = None
        self.temperature = None
        self.max_tokens = None

    def get_metadata(self, provider: str, model: str, temperature: float, max_tokens: int) -> None:
        self.object_a_metadata = self.object_a.to_dict(provider, model, temperature, max_tokens)
        self.object_b_metadata = self.object_b.to_dict(provider, model, temperature, max_tokens)
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _ask_llm(self, relation_type: str) -> str:
        prompt = (
            f"Given the metadata of two objects, determine the {relation_type}.\n"
            f"Object A:\n{self.object_a_metadata}\n\n"
            f"Object B:\n{self.object_b_metadata}\n\n"
            f"Answer with a concise and clear explanation of the {relation_type}."
        )
        return call_llm(self.provider,prompt, self.model, self.temperature, self.max_tokens)

    def above_below_relationship(self):
        """
        Determine if one object is above or below the other one.
        """
        return self._ask_llm("above/below relationship")

    def left_right_relationship(self):
        """
        Determine if one object is to the left or right of the other object"
        """
        return self._ask_llm("left/right relationship")

    def containment_relationship(self):
        """
        Determine if one object contains the other object.
        """
        return self._ask_llm("containment relationship")

    def contact_relationship(self):
        """
        Determine if the objects are in contact.
        """
        return self._ask_llm("contact relationship")

    def pattern_color_placement_size_shape_match(self):
        """
        Compare pattern, color, placement, size, and shape of two objects.
        """
        return self._ask_llm("pattern, color, placement, size, and shape match")

    def cavity_relationship(self):
        """
        Determine if a object fits into the other objects cavity
        """
        return self._ask_llm("cavity relationship")

    def perimeter_relationship(self):
        """
        Determine how the perimeters of the two objects relate.
        """
        return self._ask_llm("perimeter relationship")

    def content_relationship(self):
        """
        Analyze the contents of the objects for relationship (e.g., shared items, parts).
        """
        return self._ask_llm("content relationship")

    def to_dict(self) -> Dict[str, Any]:
        """Convert self-relation properties to a dictionary."""
        return {
            "above_below_relationship": self.above_below_relationship(),
            "left_right_relationship": self.left_right_relationship(),
            "containment_relationship": self.containment_relationship(),
            "contact_relationship": self.contact_relationship(),
            "pattern_color_placement_size_shape_relationship": self.pattern_color_placement_size_shape_match(),
            "cavity_relationship": self.cavity_relationship(),
            "perimeter_relationship": self.perimeter_relationship(),
            "content_relationship": self.content_relationship()
        }


# Example usage
# if __name__ == "__main__":
#     task_json_path = input("Enter path of task json: ")
#     import json
#     from arcagi.objects.visualize import read_json_as_string
#     # visualize
#     task_data = json.loads(read_json_as_string(task_json_path))
#     # Pick one of the grids (e.g., first one)
#     print("Enter space separated grid params (ex. train 1 input) which means 1st grid of the train sample's input")
#     grid_params = input("Enter space separated grid params: ").split(" ")
#     grid = task_data[grid_params[0]][int(grid_params[1]) - 1][grid_params[2]]
#     x = Grid(grid)
#     x.visualize()
#     import os
#     os.environ["OPENAI_API_KEY"] = "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"
#     print("background: ", x.find_background('openai', 'gpt-4.1-2025-04-14', 0.0, 4096))
#     from arcagi.objects.visualize_objects import find_objects
#     objects = find_objects(x.to_list())
#     y1 = BaseObject(x.to_list(), objects[2])
#     y1.visualize()
#     y2 = BaseObject(x.to_list(), objects[3])
#     y2.visualize()
#     sr = SpatialRelations(y1, y2)
#     sr.get_metadata('openai', 'gpt-4.1-2025-04-14', 0.0, 4096)
