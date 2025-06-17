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
import uuid
import time

# Setup logging directory - delete old logs and recreate
log_dir = "logs_parallel"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir, exist_ok=True)

def setup_logger(test_name: str):
    """Setup a logger for each test case"""
    unique_id = str(uuid.uuid4())[:8]
    logger_name = f"{test_name}_{unique_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    logger.handlers.clear()
    
    temp_log_file = f"{log_dir}/{test_name}_temp_{unique_id}.log"
    file_handler = logging.FileHandler(temp_log_file)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger, file_handler, temp_log_file

def finalize_log_file(temp_log_file: str, test_name: str, passed: bool):
    """Rename the temporary log file to include pass/fail status"""
    status = "PASS" if passed else "FAIL"
    unique_id = temp_log_file.split('_temp_')[-1].replace('.log', '')
    final_log_file = f"{log_dir}/{test_name}_{status}_{unique_id}.log"
    
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
    data = BaseObject(grid, coord_tuples).to_dict(
        provider="openai", 
        model="gpt-4.1", 
        temperature=0.0, 
        max_tokens=4096
    )
    return data

async def create_base_object(grid, coord_tuples):
    """
    Async wrapper that runs the sync function in a thread pool
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, create_base_object_sync, grid, coord_tuples)

def normalize_string(s: str) -> str:
    """Normalize string by converting to lowercase and removing spaces and punctuation"""
    return re.sub(r'[^a-z0-9]', '', s.lower())

with open("testing/samples/patterns_data.json") as d:
    PATTERN_DATA = json.load(d)

with open("testing/samples/pattern_testing_objects.json") as f:
    OBJECT_DATA = json.load(f)

async def run_single_pattern_test(pattern_key):
    """Run a single pattern test - same logic as the individual file but returns result"""
    logger, file_handler, temp_log_file = setup_logger(f"test_pattern_{pattern_key}")
    
    start_time = time.time()
    try:
        logger.info(f"Starting test for pattern key: {pattern_key}")
        
        if pattern_key not in PATTERN_DATA:
            logger.error(f"Pattern key {pattern_key} not found in pattern data")
            return {"pattern_key": pattern_key, "status": "FAIL", "error": "Pattern key not found in pattern data"}
            
        if pattern_key not in OBJECT_DATA:
            logger.error(f"Pattern key {pattern_key} not found in object data")
            return {"pattern_key": pattern_key, "status": "FAIL", "error": "Pattern key not found in object data"}
        
        expected_patterns = PATTERN_DATA[pattern_key]
        logger.info(f"Expected patterns: {expected_patterns}")
        
        try:
            with open(f"data/{pattern_key}.json") as f:
                json_data = json.load(f)
            logger.info(f"Loaded grid data file for {pattern_key}")
            
        except FileNotFoundError:
            logger.error(f"Grid data file not found: data/{pattern_key}.json")
            return {"pattern_key": pattern_key, "status": "FAIL", "error": "Grid data file not found"}
        except Exception as e:
            logger.error(f"Error loading grid data: {str(e)}")
            return {"pattern_key": pattern_key, "status": "FAIL", "error": f"Error loading grid data: {str(e)}"}
        
        task_data = OBJECT_DATA[pattern_key]
        
        overall_found = False
        
        for train_idx, train_example in enumerate(task_data.get("train", [])):
            logger.info(f"Processing training example {train_idx}")
            
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
                input_tasks = [create_base_object(grid_a, obj_coords) for obj_coords in input_coord_sets]
                before_list = await asyncio.gather(*input_tasks) if input_tasks else []
                logger.info(f"Created {len(before_list)} input objects")
                
                logger.info("Creating Output Objects")
                output_tasks = [create_base_object(grid_b, obj_coords) for obj_coords in output_coord_sets]
                after_list = await asyncio.gather(*output_tasks) if output_tasks else []
                logger.info(f"Created {len(after_list)} output objects")
                
                logger.info("Finding Patterns")
                pattern_params, counts = await unit_patterns(grid_a, grid_b, before_list, after_list)
                logger.info(f"Pattern parameters: {pattern_params}")
                logger.info(f"Pattern counts: {counts}")
                
                found_expected = False
                for expected in expected_patterns:
                    normalized_expected = normalize_string(expected)
                    normalized_counts = {normalize_string(k): v for k, v in counts.items()}
                    
                    if normalized_expected in normalized_counts:
                        found_expected = True
                        overall_found = True
                        count = normalized_counts[normalized_expected]
                        logger.info(f"✓ Found expected pattern '{expected}' with count {count} in training example {train_idx}")
                        break
                
                if not found_expected:
                    logger.warning(f"Expected patterns {expected_patterns} not found in training example {train_idx}")
                    logger.info(f"Available patterns: {list(counts.keys())}")
                    
            except Exception as e:
                logger.error(f"Error processing training example {train_idx}: {str(e)}")
                continue
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Completed test for pattern key: {pattern_key} in {duration:.2f} seconds")
        
        if not overall_found:
            logger.error(f"Expected patterns {expected_patterns} not found across all training examples")
        
        file_handler.close()
        logger.removeHandler(file_handler)
        finalize_log_file(temp_log_file, f"test_pattern_{pattern_key}", overall_found)
        
        return {
            "pattern_key": pattern_key, 
            "status": "PASS" if overall_found else "FAIL", 
            "duration": duration,
            "found_pattern": overall_found
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logger.error(f"Test failed for pattern key {pattern_key}: {str(e)}")
        if 'file_handler' in locals():
            file_handler.close()
            logger.removeHandler(file_handler)
        finalize_log_file(temp_log_file, f"test_pattern_{pattern_key}", False)
        return {
            "pattern_key": pattern_key, 
            "status": "FAIL", 
            "duration": duration,
            "error": str(e)
        }

@pytest.mark.asyncio
async def test_all_patterns_parallel():
    """Run all pattern tests concurrently and provide summary"""
    print(f"\n🚀 Starting parallel execution of {len(PATTERN_DATA)} pattern tests...")
    
    start_time = time.time()
    
    # Create tasks for all pattern tests
    tasks = [run_single_pattern_test(pattern_key) for pattern_key in PATTERN_DATA.keys()]
    
    # Run all tests concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Process results
    passed = []
    failed = []
    errors = []
    
    for result in results:
        if isinstance(result, Exception):
            errors.append(str(result))
        elif result["status"] == "PASS":
            passed.append(result)
        else:
            failed.append(result)
    
    # Print summary
    print(f"\n📊 PARALLEL TEST SUMMARY")
    print(f"━" * 50)
    print(f"Total Duration: {total_duration:.2f} seconds")
    print(f"Total Tests: {len(PATTERN_DATA)}")
    print(f"✅ Passed: {len(passed)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"💥 Errors: {len(errors)}")
    print(f"━" * 50)
    
    if passed:
        print(f"\n✅ PASSED TESTS ({len(passed)}):")
        for result in passed:
            print(f"  • {result['pattern_key']} ({result['duration']:.2f}s)")
    
    if failed:
        print(f"\n❌ FAILED TESTS ({len(failed)}):")
        for result in failed:
            duration = result.get('duration', 0)
            error = result.get('error', 'Pattern not found')
            print(f"  • {result['pattern_key']} ({duration:.2f}s) - {error}")
    
    if errors:
        print(f"\n💥 ERROR TESTS ({len(errors)}):")
        for error in errors:
            print(f"  • {error}")
    
    print(f"\n📁 Individual logs saved in: {log_dir}/")
    print(f"🏃‍♂️ Average time per test: {total_duration/len(PATTERN_DATA):.2f}s")
    
    # Assert that we have some passes (you can adjust this logic)
    assert len(passed) > 0, f"No tests passed! {len(failed)} failed, {len(errors)} errors"
    
    print(f"🎉 Parallel testing completed successfully!")

if __name__ == "__main__":
    import asyncio
    print("Running all pattern tests in parallel...")
    asyncio.run(test_all_patterns_parallel())
