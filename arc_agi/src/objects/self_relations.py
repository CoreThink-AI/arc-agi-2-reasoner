from typing import List, Tuple, Dict, Any, Set
from arc_agi.src.objects.base import Grid, BaseObject
from arc_agi.src.utils.llm_utils import call_llm


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
        # to_dict is async; run to completion synchronously
        import asyncio
        self.object_a_metadata = asyncio.run(self.object_a.to_dict(provider, model, temperature, max_tokens))
        self.object_b_metadata = asyncio.run(self.object_b.to_dict(provider, model, temperature, max_tokens))
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
        return call_llm(prompt, self.provider, self.model, self.temperature, self.max_tokens)

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

