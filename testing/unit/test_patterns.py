from arc_agi.objects.base import BaseObject, Coordinates
from arc_agi.patterns.find_patterns import unit_patterns
import json
import asyncio
import numpy as np
from testing.samples.sample1 import grid_a,grid_b,input_obj,output_obj

def create_base_object_sync(grid, coord_tuples):
    """
    Helper: given a 2D list (or array) and a set of (x, y) tuples,
    construct a BaseObject synchronously.
    """
    # BaseObject constructor expects Set[Tuple[int, int]] directly
    data = BaseObject(grid, coord_tuples).to_dict(
        provider="anthropic", 
        model="claude-3-opus-20240229", 
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
    print("Making Input Objects")
    # Create all input object tasks concurrently
    input_tasks = [create_base_object(grid_a, obj) for obj in input_obj]
    before_list = await asyncio.gather(*input_tasks)
    print(f"Created {len(before_list)} input objects")
    
    print("Making Output Objects")
    # Create all output object tasks concurrently
    output_tasks = [create_base_object(grid_b, obj) for obj in output_obj]
    after_list = await asyncio.gather(*output_tasks)
    print(f"Created {len(after_list)} output objects")
    
    print("Finding Patterns")
    results = await unit_patterns(grid_a, grid_b, before_list, after_list)
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
