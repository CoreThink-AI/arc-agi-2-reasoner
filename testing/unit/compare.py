from arc_agi.src.objects.base import BaseObject, Coordinates
import json
import asyncio
import numpy as np
from arc_agi.src.objects.object_finder import find_objects
from arc_agi.src.patterns.object_comparison import compare_object_lists, analyze_changes

def create_base_object_sync(grid, coord_tuples):
    """
    Helper: given a 2D list (or array) and a set of (x, y) tuples,
    construct a BaseObject synchronously.
    """
    # BaseObject constructor expects Set[Tuple[int, int]] directly
    data = asyncio.run(
        BaseObject(grid, coord_tuples).to_dict(
            provider="openai", 
            model="gpt-4.1-mini", 
            temperature=0.0, 
            max_tokens=4096
        )
    )
    del data["grid"]
    return data

async def create_base_object(grid, coord_tuples):
    """
    Async wrapper that runs the sync function in a thread pool
    """
    return await asyncio.to_thread(create_base_object_sync, grid, coord_tuples)

async def main():
    with open("data/16b78196.json") as f:
        json_data = json.load(f)
    grid_a = json_data["train"][0]["input"]
    grid_b = json_data["train"][0]["output"]
    input_obj = find_objects(grid_a)
    output_obj = find_objects(grid_b)
    print("Making Input Objects")
    # Create all input object tasks concurrently
    input_tasks = [create_base_object(grid_a, obj) for obj in input_obj]
    before_list = await asyncio.gather(*input_tasks)
    
    print("Making Output Objects")
    # Create all output object tasks concurrently
    output_tasks = [create_base_object(grid_b, obj) for obj in output_obj]
    after_list = await asyncio.gather(*output_tasks)

    # Your before_list and after_list from to_dict() calls
    comparison = compare_object_lists(before_list, after_list)

    print(f"Added: {len(comparison.added)}")
    print(f"Removed: {len(comparison.removed)}")
    print(f"Retained: {len(comparison.retained)}")

if __name__ == "__main__":
    asyncio.run(main())
