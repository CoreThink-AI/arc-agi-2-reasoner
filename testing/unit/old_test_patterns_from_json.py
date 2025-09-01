from arc_agi.src.objects.base import BaseObject, Coordinates
from arc_agi.src.patterns.find_patterns import unit_patterns
import json
import asyncio
import numpy as np
import logging
import os
import re
from datetime import datetime
import pytest 
import shutil

# Setup logging directory - delete old logs and recreate
if os.path.exists("logs"):
    shutil.rmtree("logs")
os.makedirs("logs", exist_ok=True)

def setup_logger(test_name: str):
    """Setup a logger for each test case"""
    logger = logging.getLogger(test_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create temporary file handler (will be renamed later with pass/fail status)
    temp_log_file = f"logs/{test_name}_temp.log"
    file_handler = logging.FileHandler(temp_log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger, file_handler, temp_log_file

def finalize_log_file(temp_log_file: str, test_name: str, passed: bool):
    """Rename the temporary log file to include pass/fail status"""
    status = "PASS" if passed else "FAIL"
    final_log_file = f"logs/{test_name}_{status}.log"
    
    # Close any open handles and rename
    try:
        if os.path.exists(temp_log_file):
            os.rename(temp_log_file, final_log_file)
    except Exception as e:
        print(f"Warning: Could not rename log file: {e}")

def create_base_object_sync(grid, coord_tuples):
    """
    Helper: given a 2D list (or array) and a set of (x, y) tuples,
    construct a BaseObject synchronously.
    """
    # BaseObject constructor expects Set[Tuple[int, int]] directly
    data = asyncio.run(
        BaseObject(grid, coord_tuples).to_dict(
            provider="openai", 
            model="gpt-4.1", 
            temperature=0.0, 
            max_tokens=4096
        )
    )
    return data

async def create_base_object(grid, coord_tuples):
    """
    Async wrapper that runs the sync function in a thread pool
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, create_base_object_sync, grid, coord_tuples)

def normalize_string(s: str) -> str:
    """Normalize string by converting to lowercase and removing spaces and punctuation"""
    return re.sub(r'[^a-z0-9]', '', s.lower())

# Load pattern data and object data once at module level
with open("testing/samples/patterns_data.json") as d:
    PATTERN_DATA = json.load(d)

with open("testing/samples/pattern_testing_objects.json") as f:
    OBJECT_DATA = json.load(f)

async def run_pattern_case_from_json(pattern_key):
    """Test individual pattern case with logging using pre-loaded JSON objects"""
    logger, file_handler, temp_log_file = setup_logger(f"test_pattern_{pattern_key}")
    
    try:
        logger.info(f"Starting test for pattern key: {pattern_key}")
        
        if pattern_key not in PATTERN_DATA:
            logger.error(f"Pattern key {pattern_key} not found in pattern data")
            pytest.fail(f"Pattern key {pattern_key} not found in pattern data")
            
        if pattern_key not in OBJECT_DATA:
            logger.error(f"Pattern key {pattern_key} not found in object data")
            pytest.fail(f"Pattern key {pattern_key} not found in object data")
        
        expected_patterns = PATTERN_DATA[pattern_key]
        logger.info(f"Expected patterns: {expected_patterns}")
        
        # Load actual grid data from data folder
        try:
            with open(f"data/{pattern_key}.json") as f:
                json_data = json.load(f)
            logger.info(f"Loaded grid data file for {pattern_key}")
            
        except FileNotFoundError:
            logger.error(f"Grid data file not found: data/{pattern_key}.json")
            pytest.fail(f"Grid data file not found: data/{pattern_key}.json")
        except Exception as e:
            logger.error(f"Error loading grid data: {str(e)}")
            pytest.fail(f"Error loading grid data: {str(e)}")
        
        # Get the object coordinate data for this pattern key
        task_data = OBJECT_DATA[pattern_key]
        
        # Track if we find any expected pattern across all training examples
        overall_found_expected = False
        
        for train_idx, train_example in enumerate(task_data.get("train", [])):
            logger.info(f"Processing training example {train_idx}")
            
            # Load the corresponding grid data for this training example
            try:
                grid_a = json_data["train"][train_idx]["input"]
                grid_b = json_data["train"][train_idx]["output"]
                logger.info(f"Loaded grids for example {train_idx} - Input: {np.array(grid_a).shape}, Output: {np.array(grid_b).shape}")
            except IndexError:
                logger.error(f"Training example {train_idx} not found in grid data")
                continue
            except Exception as e:
                logger.error(f"Error loading grids for example {train_idx}: {str(e)}")
                continue
            
            input_objects = train_example.get("input", [])
            output_objects = train_example.get("output", [])
            
            logger.info(f"Found {len(input_objects)} input objects and {len(output_objects)} output objects")
            
            # Convert coordinate data to coordinate tuples for input objects
            input_coord_sets = []
            for obj_idx, obj_coords in enumerate(input_objects):
                if not obj_coords:
                    continue
                    
                coord_tuples = set()
                for coord in obj_coords:
                    if isinstance(coord, list) and len(coord) == 2:
                        coord_tuples.add((coord[0], coord[1]))
                
                if coord_tuples:
                    input_coord_sets.append(coord_tuples)
                    logger.info(f"Input object {obj_idx} has {len(coord_tuples)} coordinates")
            
            # Convert coordinate data to coordinate tuples for output objects
            output_coord_sets = []
            for obj_idx, obj_coords in enumerate(output_objects):
                if not obj_coords:
                    continue
                    
                coord_tuples = set()
                for coord in obj_coords:
                    if isinstance(coord, list) and len(coord) == 2:
                        coord_tuples.add((coord[0], coord[1]))
                
                if coord_tuples:
                    output_coord_sets.append(coord_tuples)
                    logger.info(f"Output object {obj_idx} has {len(coord_tuples)} coordinates")
            
            if not input_coord_sets and not output_coord_sets:
                logger.warning(f"No valid objects found for training example {train_idx}")
                continue
            
            try:
                logger.info("Creating Input Objects")
                # Create all input object tasks concurrently
                input_tasks = [create_base_object(grid_a, obj_coords) for obj_coords in input_coord_sets]
                before_list = await asyncio.gather(*input_tasks) if input_tasks else []
                logger.info(f"Created {len(before_list)} input objects")
                
                logger.info("Creating Output Objects")
                # Create all output object tasks concurrently
                output_tasks = [create_base_object(grid_b, obj_coords) for obj_coords in output_coord_sets]
                after_list = await asyncio.gather(*output_tasks) if output_tasks else []
                logger.info(f"Created {len(after_list)} output objects")
                
                logger.info("Finding Patterns")
                pattern_params, counts = await unit_patterns(grid_a, grid_b, before_list, after_list)
                logger.info(f"Pattern parameters: {pattern_params}")
                logger.info(f"Pattern counts: {counts}")
                
                # Check if any expected pattern is found
                found_expected = False
                for expected in expected_patterns:
                    normalized_expected = normalize_string(expected)
                    normalized_counts = {normalize_string(k): v for k, v in counts.items()}
                    
                    if normalized_expected in normalized_counts:
                        found_expected = True
                        count = normalized_counts[normalized_expected]
                        logger.info(f"✓ Found expected pattern '{expected}' with count {count} in training example {train_idx}")
                        break
                
                if found_expected:
                    overall_found_expected = True
                
                if not found_expected:
                    logger.warning(f"Expected patterns {expected_patterns} not found in training example {train_idx}")
                    logger.info(f"Available patterns: {list(counts.keys())}")
                    
            except Exception as e:
                logger.error(f"Error processing training example {train_idx}: {str(e)}")
                continue
        
        logger.info(f"Completed test for pattern key: {pattern_key}")
        
        if not overall_found_expected:
            logger.error(f"Expected patterns {expected_patterns} not found across all training examples")
            pytest.fail(f"Expected patterns {expected_patterns} not found across all training examples")
        
        # Close the file handler before renaming
        file_handler.close()
        logger.removeHandler(file_handler)
        finalize_log_file(temp_log_file, f"test_pattern_{pattern_key}", overall_found_expected)
        
    except Exception as e:
        logger.error(f"Test failed for pattern key {pattern_key}: {str(e)}")
        # Close the file handler before renaming
        if 'file_handler' in locals():
            file_handler.close()
            logger.removeHandler(file_handler)
        finalize_log_file(temp_log_file, f"test_pattern_{pattern_key}", False)
        raise

# Dynamic test generation using pytest parametrize
@pytest.mark.asyncio
@pytest.mark.parametrize("pattern_key", list(PATTERN_DATA.keys()))
async def test_pattern_from_json(pattern_key):
    """Test pattern detection for each pattern key loaded from JSON"""
    await run_pattern_case_from_json(pattern_key)

if __name__ == "__main__":
    # For manual testing
    import asyncio
    
    async def run_single_test(pattern_key):
        await run_pattern_case_from_json(pattern_key)
    
    # Test a specific pattern
    if len(PATTERN_DATA) > 0:
        first_key = list(PATTERN_DATA.keys())[0]
        print(f"Testing pattern: {first_key}")
        asyncio.run(run_single_test(first_key))
