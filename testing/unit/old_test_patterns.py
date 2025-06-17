from arc_agi.src.objects.base import BaseObject, Coordinates
from arc_agi.src.patterns.find_patterns import unit_patterns
import json
import asyncio
import numpy as np
import logging
import os
import re
from datetime import datetime
from arc_agi.src.objects.object_finder import find_objects
import pytest 

# Setup logging directory
os.makedirs("logs", exist_ok=True)

def setup_logger(test_name: str):
    """Setup a logger for each test case"""
    logger = logging.getLogger(test_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{test_name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger

def create_base_object_sync(grid, coord_tuples):
    """
    Helper: given a 2D list (or array) and a set of (x, y) tuples,
    construct a BaseObject synchronously.
    """
    # BaseObject constructor expects Set[Tuple[int, int]] directly
    data = BaseObject(grid, coord_tuples).to_dict(
        provider="openai", 
        model="gpt-4.1", 
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

def normalize_string(s: str) -> str:
    """Normalize string by converting to lowercase and removing spaces and punctuation"""
    if not isinstance(s, str):
        s = str(s)
    # Remove all non-alphanumeric characters and convert to lowercase
    return re.sub(r'[^a-zA-Z0-9]', '', s.lower())

# Load pattern data once at module level
with open("testing/samples/patterns_data.json") as d:
    PATTERN_DATA = json.load(d)

@pytest.mark.parametrize("pattern_key", list(PATTERN_DATA.keys()))
@pytest.mark.asyncio
async def test_pattern_case(pattern_key):
    """Test individual pattern case with logging"""
    logger = setup_logger(f"test_pattern_{pattern_key}")
    
    try:
        logger.info(f"Starting test for pattern key: {pattern_key}")
        
        # Load pattern data
        with open(f"data/{pattern_key}.json") as f:
            json_data = json.load(f)
        
        grid_a = json_data["train"][0]["input"]
        grid_b = json_data["train"][0]["output"]
        logger.info(f"Loaded grids - Input: {np.array(grid_a).shape}, Output: {np.array(grid_b).shape}")
        
        # Find objects
        input_obj = find_objects(grid_a)
        output_obj = find_objects(grid_b)
        logger.info(f"Found {len(input_obj)} input objects and {len(output_obj)} output objects")
        
        logger.info("Creating Input Objects")
        # Create all input object tasks concurrently
        input_tasks = [create_base_object(grid_a, obj) for obj in input_obj]
        before_list = await asyncio.gather(*input_tasks)
        logger.info(f"Created {len(before_list)} input objects")
        
        logger.info("Creating Output Objects")
        # Create all output object tasks concurrently
        output_tasks = [create_base_object(grid_b, obj) for obj in output_obj]
        after_list = await asyncio.gather(*output_tasks)
        logger.info(f"Created {len(after_list)} output objects")
        
        logger.info("Finding Patterns")
        pattern_params, counts = await unit_patterns(grid_a, grid_b, before_list, after_list)
        logger.info(f"Pattern parameters: {pattern_params}")
        logger.info(f"Pattern counts: {counts}")
        
        # Assertion
        expected_pattern = PATTERN_DATA[pattern_key][0]
        logger.info(f"Expected pattern: {expected_pattern}")
        
        # Normalize strings for comparison
        normalized_expected = normalize_string(expected_pattern)
        normalized_counts = {normalize_string(k): v for k, v in counts.items()}
        
        logger.info(f"Normalized expected pattern: {normalized_expected}")
        logger.info(f"Normalized counts keys: {list(normalized_counts.keys())}")
        
        assert normalized_expected in normalized_counts, f"Expected pattern '{expected_pattern}' (normalized: '{normalized_expected}') not found in normalized counts: {list(normalized_counts.keys())}"
        logger.info(f"Test PASSED: Found expected pattern '{expected_pattern}' in counts")
        
    except Exception as e:
        logger.error(f"Test FAILED with error: {str(e)}")
        logger.exception("Full traceback:")
        raise
    finally:
        # Close logger handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

#if __name__ == "__main__":
#    asyncio.run(main())
