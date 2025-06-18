from arc_agi.src.objects.base import Grid, BaseObject
from arc_agi.src.patterns.find_patterns import unit_patterns
import json
import asyncio
import numpy as np
from arc_agi.src.objects.object_finder import find_objects

def create_base_object_sync(grid, coord_tuples):
    """
    Helper: given a 2D list (or array) and a set of (x, y) tuples,
    construct a BaseObject synchronously.
    """
    # BaseObject constructor expects Set[Tuple[int, int]] directly
    data = BaseObject(grid, coord_tuples).to_dict(
        provider="openai", 
        model="gpt-4.1-mini", 
        temperature=0.0, 
        max_tokens=4096
    )
    del data["grid"]
    return data

async def create_base_object(grid, coord_tuples):
    """
    Async wrapper that runs the sync function in a thread pool
    """
    return await asyncio.to_thread(create_base_object_sync, grid, coord_tuples)

async def main():
    with open("data/31f7f899.json") as f:
        json_data = json.load(f)
    grid_input = json_data["train"][0]["input"]
    grid_output = json_data["train"][0]["output"]
    grid_a = Grid(grid_input)
    input_obj = grid_a.find_objects_in_grid('openai', 'gpt-4.1-mini', 0.0, 4096)
    print(input_obj)
    input_obj = grid_a.objects_concatenator('openai', 'gpt-4.1-mini', 0.0, 4096)
    print(input_obj)
    grid_b = Grid(grid_output)
    output_obj = grid_b.find_objects_in_grid('openai', 'gpt-4.1-mini', 0.0, 4096)
    output_obj = grid_b.objects_concatenator('openai', 'gpt-4.1-mini', 0.0, 4096)
    print("Making Input Objects")
    # Create all input object tasks concurrently
    input_tasks = [create_base_object(grid_input, obj) for obj in input_obj]
    before_list = await asyncio.gather(*input_tasks)
    print(f"Created {len(before_list)} input objects")
    
    print("Making Output Objects")
    # Create all output object tasks concurrently
    output_tasks = [create_base_object(grid_output, obj) for obj in output_obj]
    after_list = await asyncio.gather(*output_tasks)
    print(f"Created {len(after_list)} output objects")
    
    print("Finding Patterns")
    pattern_params,counts = await unit_patterns(grid_input, grid_output, before_list, after_list)
    print(pattern_params)
    print(counts)

if __name__ == "__main__":
    asyncio.run(main())