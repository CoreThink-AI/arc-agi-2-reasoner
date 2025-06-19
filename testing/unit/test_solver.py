from arc_agi.src.solver.solver import (
    get_formatted_examples, 
    get_prompts, 
    extract_matrix_from_response, 
    matrix_to_arr, 
    get_solved_outputs
)
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
log_dir = "logs_solver"
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

# Test data - sample ARC tasks for testing
# Note: This data is now loaded dynamically from hints.json and data/ directory
# The following constants are kept for reference but not used in the updated tests

# SAMPLE_ARC_TASKS = {
#     "581f7754": {
#         "train": [
#             {
#                 "input": [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 8, 8, 8, 1, 1, 1, 1], [1, 8, 4, 8, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 8, 1], [1, 1, 1, 1, 8, 8, 4, 1], [1, 1, 1, 1, 1, 1, 8, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 8, 1, 8, 1, 1, 1, 1], [1, 8, 4, 8, 1, 1, 1, 1], [1, 8, 1, 8, 1, 1, 1, 1], [1, 8, 8, 8, 1, 1, 1, 1], [1, 1, 1, 1, 1, 4, 1, 1]],
#                 "output": [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 8, 8, 8, 1], [1, 1, 1, 1, 8, 4, 8, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 8, 1, 1], [1, 1, 1, 8, 8, 4, 1, 1], [1, 1, 1, 1, 1, 8, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 8, 1, 8, 1], [1, 1, 1, 1, 8, 4, 8, 1], [1, 1, 1, 1, 8, 1, 8, 1], [1, 1, 1, 1, 8, 8, 8, 1], [1, 1, 1, 1, 1, 4, 1, 1]]
#             }
#         ],
#         "test": [
#             {
#                 "input": [[8, 8, 8, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 6, 4, 6, 8, 8, 8, 8, 8, 8, 8, 4, 6, 4, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 8, 8, 8, 8], [6, 8, 8, 8, 8, 8, 2, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 4, 8, 8, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 4, 4, 4, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 3, 3, 3], [8, 2, 8, 4, 8, 4, 8, 8, 4, 4, 4, 8, 8, 8, 8, 8, 8, 3, 8, 3], [8, 8, 8, 4, 4, 4, 8, 8, 4, 4, 6, 8, 8, 8, 8, 8, 8, 3, 8, 3], [8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 3, 6, 3], [4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]],
#                 "output": [[8, 8, 8, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 3, 3], [8, 8, 8, 4, 8, 4, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 3, 8, 3], [8, 8, 8, 4, 4, 4, 8, 8, 4, 4, 4, 8, 8, 8, 8, 8, 8, 3, 8, 3], [6, 8, 8, 8, 6, 8, 8, 8, 4, 4, 6, 8, 8, 4, 6, 4, 8, 3, 6, 3], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 4, 4, 4, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [4, 8, 8, 6, 4, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [2, 2, 8, 8, 8, 8, 2, 2, 8, 8, 8, 8, 2, 8, 8, 8, 8, 8, 8, 2], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]]
#             }
#         ]
#     }
# }

# SAMPLE_HINTS = {
#     "581f7754": "There seem to be rectangular regions with a set of digits within. You need to connect the content of each object to the content of the other objects in the order of the unit digits at the bottom."
# }

def test_get_formatted_examples():
    """Test the get_formatted_examples function"""
    logger, file_handler, temp_log_file = setup_logger("test_get_formatted_examples")
    
    try:
        logger.info("Testing get_formatted_examples function")
        
        # Load hints from hints.json file to get a task key
        hints_file_path = "testing/samples/solver_data/hints.json"
        try:
            with open(hints_file_path, 'r') as f:
                hints_data = json.load(f)
            logger.info(f"Loaded hints from {hints_file_path}")
        except Exception as e:
            logger.error(f"Failed to load hints from {hints_file_path}: {str(e)}")
            raise
        
        # Use the first available task key
        task_key = list(hints_data.keys())[0]
        
        # Load test data from data directory
        data_file_path = f"data/{task_key}.json"
        try:
            with open(data_file_path, 'r') as f:
                arc_input = json.load(f)
            logger.info(f"Loaded test data from {data_file_path}")
        except Exception as e:
            logger.error(f"Failed to load test data from {data_file_path}: {str(e)}")
            raise
        
        # Test with sample data
        example_inputs = arc_input["train"]
        
        result = get_formatted_examples(example_inputs)
        
        # Verify the result is a string
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Result should not be empty"
        
        # Verify it contains expected elements
        assert "input" in result.lower(), "Result should contain input visualization"
        assert "output" in result.lower(), "Result should contain output visualization"
        
        logger.info("✓ get_formatted_examples test passed")
        finalize_log_file(temp_log_file, "test_get_formatted_examples", True)
        
    except Exception as e:
        logger.error(f"✗ get_formatted_examples test failed: {str(e)}")
        finalize_log_file(temp_log_file, "test_get_formatted_examples", False)
        raise

def test_get_prompts():
    """Test the get_prompts function"""
    logger, file_handler, temp_log_file = setup_logger("test_get_prompts")
    
    try:
        logger.info("Testing get_prompts function")
        
        # Load hints from hints.json file
        hints_file_path = "testing/samples/solver_data/hints.json"
        try:
            with open(hints_file_path, 'r') as f:
                hints_data = json.load(f)
            logger.info(f"Loaded hints from {hints_file_path}")
        except Exception as e:
            logger.error(f"Failed to load hints from {hints_file_path}: {str(e)}")
            raise
        
        # Use the first available task key
        task_key = list(hints_data.keys())[0]
        
        # Load test data from data directory
        data_file_path = f"data/{task_key}.json"
        try:
            with open(data_file_path, 'r') as f:
                arc_input = json.load(f)
            logger.info(f"Loaded test data from {data_file_path}")
        except Exception as e:
            logger.error(f"Failed to load test data from {data_file_path}: {str(e)}")
            raise
        
        hint = hints_data[task_key]
        
        prompt_arr, ground_truths_arr = get_prompts(arc_input, hint)
        
        # Verify the results
        assert isinstance(prompt_arr, list), "prompt_arr should be a list"
        assert isinstance(ground_truths_arr, list), "ground_truths_arr should be a list"
        assert len(prompt_arr) > 0, "prompt_arr should not be empty"
        assert len(ground_truths_arr) > 0, "ground_truths_arr should not be empty"
        
        # Verify each prompt is a string
        for prompt in prompt_arr:
            assert isinstance(prompt, str), "Each prompt should be a string"
            assert len(prompt) > 0, "Each prompt should not be empty"
        
        # Verify each ground truth is a string
        for gt in ground_truths_arr:
            assert isinstance(gt, str), "Each ground truth should be a string"
            assert len(gt) > 0, "Each ground truth should not be empty"
        
        logger.info(f"✓ get_prompts test passed - Generated {len(prompt_arr)} prompts and {len(ground_truths_arr)} ground truths")
        finalize_log_file(temp_log_file, "test_get_prompts", True)
        
    except Exception as e:
        logger.error(f"✗ get_prompts test failed: {str(e)}")
        finalize_log_file(temp_log_file, "test_get_prompts", False)
        raise

def test_extract_matrix_from_response():
    """Test the extract_matrix_from_response function"""
    logger, file_handler, temp_log_file = setup_logger("test_extract_matrix_from_response")
    
    try:
        logger.info("Testing extract_matrix_from_response function")
        
        # Test cases
        test_cases = [
            {
                "input": "Here is the matrix:\n```\n1|2|3\n4|5|6\n7|8|9\n```\nThat's the answer.",
                "expected_contains": "1|2|3"
            },
            {
                "input": "The result is [1, 2, 3, 4, 5, 6]",
                "expected_contains": "[1, 2, 3, 4, 5, 6]"
            },
            {
                "input": "No matrix here",
                "expected_contains": ""
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Testing case {i+1}")
            result = extract_matrix_from_response(test_case["input"])
            
            if test_case["expected_contains"]:
                assert test_case["expected_contains"] in result, f"Case {i+1}: Expected content not found in result"
            else:
                assert result == "", f"Case {i+1}: Expected empty result for invalid input"
        
        logger.info("✓ extract_matrix_from_response test passed")
        finalize_log_file(temp_log_file, "test_extract_matrix_from_response", True)
        
    except Exception as e:
        logger.error(f"✗ extract_matrix_from_response test failed: {str(e)}")
        finalize_log_file(temp_log_file, "test_extract_matrix_from_response", False)
        raise

def test_matrix_to_arr():
    """Test the matrix_to_arr function"""
    logger, file_handler, temp_log_file = setup_logger("test_matrix_to_arr")
    
    try:
        logger.info("Testing matrix_to_arr function")
        
        # Test cases
        test_cases = [
            {
                "input": "```\n1|2|3\n4|5|6\n7|8|9\n```",
                "expected": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            },
            {
                "input": "1|2|3\n4|5|6",
                "expected": [[1, 2, 3], [4, 5, 6]]
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Testing case {i+1}")
            result = matrix_to_arr(test_case["input"])
            
            assert result == test_case["expected"], f"Case {i+1}: Expected {test_case['expected']}, got {result}"
        
        logger.info("✓ matrix_to_arr test passed")
        finalize_log_file(temp_log_file, "test_matrix_to_arr", True)
        
    except Exception as e:
        logger.error(f"✗ matrix_to_arr test failed: {str(e)}")
        finalize_log_file(temp_log_file, "test_matrix_to_arr", False)
        raise

async def run_single_solver_test(task_key):
    """Run a single solver test - returns result for parallel execution"""
    logger, file_handler, temp_log_file = setup_logger(f"test_solver_{task_key}")
    
    start_time = time.time()
    try:
        logger.info(f"Starting solver test for task key: {task_key}")
        
        # Load hints from hints.json file
        hints_file_path = "testing/samples/solver_data/hints.json"
        try:
            with open(hints_file_path, 'r') as f:
                hints_data = json.load(f)
            logger.info(f"Loaded hints from {hints_file_path}")
        except Exception as e:
            logger.error(f"Failed to load hints from {hints_file_path}: {str(e)}")
            return {"task_key": task_key, "status": "FAIL", "error": f"Failed to load hints: {str(e)}"}
        
        if task_key not in hints_data:
            logger.error(f"Task key {task_key} not found in hints data")
            return {"task_key": task_key, "status": "FAIL", "error": "Task key not found in hints data"}
        
        # Load test data from data directory
        data_file_path = f"data/{task_key}.json"
        try:
            with open(data_file_path, 'r') as f:
                arc_input = json.load(f)
            logger.info(f"Loaded test data from {data_file_path}")
        except Exception as e:
            logger.error(f"Failed to load test data from {data_file_path}: {str(e)}")
            return {"task_key": task_key, "status": "FAIL", "error": f"Failed to load test data: {str(e)}"}
        
        hint = hints_data[task_key]
        logger.info(f"Testing solver with hint: {hint[:100]}...")
        
        # Test get_formatted_examples
        try:
            examples_str = get_formatted_examples(arc_input["train"])
            logger.info(f"Generated formatted examples string of length: {len(examples_str)}")
        except Exception as e:
            logger.error(f"Error in get_formatted_examples: {str(e)}")
            return {"task_key": task_key, "status": "FAIL", "error": f"get_formatted_examples failed: {str(e)}"}
        
        # Test get_prompts
        try:
            prompt_arr, ground_truths_arr = get_prompts(arc_input, hint)
            logger.info(f"Generated {len(prompt_arr)} prompts and {len(ground_truths_arr)} ground truths")
        except Exception as e:
            logger.error(f"Error in get_prompts: {str(e)}")
            return {"task_key": task_key, "status": "FAIL", "error": f"get_prompts failed: {str(e)}"}
        
        # Test extract_matrix_from_response with sample responses
        try:
            sample_responses = [
                "Here is the matrix:\n```\n1|2|3\n4|5|6\n7|8|9\n```",
                "The result is [1, 2, 3, 4, 5, 6]",
                "No matrix here"
            ]
            
            for i, response in enumerate(sample_responses):
                extracted = extract_matrix_from_response(response)
                logger.info(f"Extracted matrix {i+1}: {extracted[:50]}...")
        except Exception as e:
            logger.error(f"Error in extract_matrix_from_response: {str(e)}")
            return {"task_key": task_key, "status": "FAIL", "error": f"extract_matrix_from_response failed: {str(e)}"}
        
        # Test matrix_to_arr
        try:
            sample_matrix = "```\n1|2|3\n4|5|6\n7|8|9\n```"
            arr_result = matrix_to_arr(sample_matrix)
            logger.info(f"Matrix to array result: {arr_result}")
        except Exception as e:
            logger.error(f"Error in matrix_to_arr: {str(e)}")
            return {"task_key": task_key, "status": "FAIL", "error": f"matrix_to_arr failed: {str(e)}"}
        
        # Run get_solved_outputs and compare with ground truth
        try:
            logger.info("Running get_solved_outputs to generate predictions...")
            responses, arr_responses = get_solved_outputs(arc_input, hint)
            logger.info(f"Generated {len(arr_responses)} predictions")
            
            # Get ground truth solutions from test data
            ground_truth_solutions = []
            for test_entry in arc_input['test']:
                if 'output' in test_entry:
                    ground_truth_solutions.append(test_entry['output'])
            
            logger.info(f"Found {len(ground_truth_solutions)} ground truth solutions")
            
            # Compare predictions with ground truth
            correct_predictions = 0
            total_predictions = min(len(arr_responses), len(ground_truth_solutions))
            
            for i in range(total_predictions):
                prediction = arr_responses[i]
                ground_truth = ground_truth_solutions[i]
                
                # Convert prediction to numpy array for comparison if it's not empty
                if prediction and len(prediction) > 0:
                    try:
                        prediction_array = np.array(prediction)
                        ground_truth_array = np.array(ground_truth)
                        
                        if prediction_array.shape == ground_truth_array.shape:
                            if np.array_equal(prediction_array, ground_truth_array):
                                correct_predictions += 1
                                logger.info(f"✓ Prediction {i+1} matches ground truth")
                            else:
                                logger.warning(f"✗ Prediction {i+1} does not match ground truth")
                                logger.warning(f"  Prediction shape: {prediction_array.shape}")
                                logger.warning(f"  Ground truth shape: {ground_truth_array.shape}")
                        else:
                            logger.warning(f"✗ Prediction {i+1} shape mismatch: {prediction_array.shape} vs {ground_truth_array.shape}")
                    except Exception as e:
                        logger.error(f"Error comparing prediction {i+1}: {str(e)}")
                else:
                    logger.warning(f"✗ Prediction {i+1} is empty or invalid")
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            logger.info(f"Accuracy: {correct_predictions}/{total_predictions} = {accuracy:.2%}")
            
        except Exception as e:
            logger.error(f"Error in get_solved_outputs or comparison: {str(e)}")
            return {"task_key": task_key, "status": "FAIL", "error": f"get_solved_outputs failed: {str(e)}"}
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"✓ Solver test for {task_key} completed successfully in {duration:.2f}s")
        finalize_log_file(temp_log_file, f"test_solver_{task_key}", True)
        
        return {
            "task_key": task_key, 
            "status": "PASS", 
            "duration": duration,
            "prompts_generated": len(prompt_arr),
            "ground_truths_generated": len(ground_truths_arr),
            "predictions_generated": len(arr_responses),
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
            "accuracy": accuracy
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logger.error(f"✗ Solver test for {task_key} failed after {duration:.2f}s: {str(e)}")
        finalize_log_file(temp_log_file, f"test_solver_{task_key}", False)
        
        return {
            "task_key": task_key, 
            "status": "FAIL", 
            "error": str(e),
            "duration": duration
        }

@pytest.mark.asyncio
async def test_all_solver_functions_parallel():
    """Test all solver functions in parallel"""
    logger, file_handler, temp_log_file = setup_logger("test_all_solver_parallel")
    
    try:
        logger.info("Starting parallel solver tests")
        
        # Load task keys from hints.json file
        hints_file_path = "testing/samples/solver_data/hints.json"
        try:
            with open(hints_file_path, 'r') as f:
                hints_data = json.load(f)
            logger.info(f"Loaded hints from {hints_file_path}")
        except Exception as e:
            logger.error(f"Failed to load hints from {hints_file_path}: {str(e)}")
            raise
        
        # Get all available task keys from hints
        task_keys = list(hints_data.keys())
        logger.info(f"Testing {len(task_keys)} solver tasks: {task_keys}")
        
        # Run tests in parallel
        start_time = time.time()
        tasks = [run_single_solver_test(task_key) for task_key in task_keys]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_duration = end_time - start_time
        logger.info(f"All parallel tests completed in {total_duration:.2f}s")
        
        # Analyze results
        passed = 0
        failed = 0
        total_prompts = 0
        total_ground_truths = 0
        total_predictions = 0
        total_correct_predictions = 0
        total_accuracy = 0.0
        
        for result in results:
            if result["status"] == "PASS":
                passed += 1
                total_prompts += result.get("prompts_generated", 0)
                total_ground_truths += result.get("ground_truths_generated", 0)
                total_predictions += result.get("predictions_generated", 0)
                total_correct_predictions += result.get("correct_predictions", 0)
                total_accuracy += result.get("accuracy", 0.0)
                
                accuracy = result.get("accuracy", 0.0)
                logger.info(f"✓ {result['task_key']}: PASS ({result.get('duration', 0):.2f}s, Accuracy: {accuracy:.2%})")
            else:
                failed += 1
                logger.error(f"✗ {result['task_key']}: FAIL - {result.get('error', 'Unknown error')}")
        
        overall_accuracy = total_correct_predictions / total_predictions if total_predictions > 0 else 0
        avg_accuracy = total_accuracy / passed if passed > 0 else 0
        
        logger.info(f"Test Summary:")
        logger.info(f"  Total tests: {len(results)}")
        logger.info(f"  Passed: {passed}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total prompts generated: {total_prompts}")
        logger.info(f"  Total ground truths generated: {total_ground_truths}")
        logger.info(f"  Total predictions generated: {total_predictions}")
        logger.info(f"  Total correct predictions: {total_correct_predictions}")
        logger.info(f"  Overall accuracy: {overall_accuracy:.2%}")
        logger.info(f"  Average accuracy per task: {avg_accuracy:.2%}")
        logger.info(f"  Total duration: {total_duration:.2f}s")
        
        # Assert that all tests passed
        assert failed == 0, f"{failed} solver tests failed"
        assert passed == len(task_keys), f"Expected {len(task_keys)} tests to pass, but only {passed} passed"
        
        logger.info("✓ All parallel solver tests passed")
        finalize_log_file(temp_log_file, "test_all_solver_parallel", True)
        
    except Exception as e:
        logger.error(f"✗ Parallel solver tests failed: {str(e)}")
        finalize_log_file(temp_log_file, "test_all_solver_parallel", False)
        raise

def test_solver_integration():
    """Integration test for the complete solver pipeline"""
    logger, file_handler, temp_log_file = setup_logger("test_solver_integration")
    
    try:
        logger.info("Testing complete solver integration")
        
        # Load hints from hints.json file
        hints_file_path = "testing/samples/solver_data/hints.json"
        try:
            with open(hints_file_path, 'r') as f:
                hints_data = json.load(f)
            logger.info(f"Loaded hints from {hints_file_path}")
        except Exception as e:
            logger.error(f"Failed to load hints from {hints_file_path}: {str(e)}")
            raise
        
        # Use the first available task key for integration test
        task_key = list(hints_data.keys())[0]  # Use first task key
        logger.info(f"Using task key {task_key} for integration test")
        
        # Load test data from data directory
        data_file_path = f"data/{task_key}.json"
        try:
            with open(data_file_path, 'r') as f:
                arc_input = json.load(f)
            logger.info(f"Loaded test data from {data_file_path}")
        except Exception as e:
            logger.error(f"Failed to load test data from {data_file_path}: {str(e)}")
            raise
        
        hint = hints_data[task_key]
        
        # Step 1: Format examples
        examples_str = get_formatted_examples(arc_input["train"])
        assert len(examples_str) > 0, "Examples string should not be empty"
        logger.info("✓ Step 1: Examples formatting successful")
        
        # Step 2: Generate prompts
        prompt_arr, ground_truths_arr = get_prompts(arc_input, hint)
        assert len(prompt_arr) > 0, "Should generate at least one prompt"
        assert len(ground_truths_arr) > 0, "Should generate at least one ground truth"
        logger.info("✓ Step 2: Prompt generation successful")
        
        # Step 3: Test matrix extraction
        sample_response = "Here is the matrix:\n```\n1|2|3\n4|5|6\n7|8|9\n```"
        extracted = extract_matrix_from_response(sample_response)
        assert "1|2|3" in extracted, "Matrix extraction should work"
        logger.info("✓ Step 3: Matrix extraction successful")
        
        # Step 4: Test matrix to array conversion
        arr_result = matrix_to_arr(extracted)
        expected = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert arr_result == expected, "Matrix to array conversion should work"
        logger.info("✓ Step 4: Matrix to array conversion successful")
        
        # Step 5: Run get_solved_outputs and compare with ground truth
        logger.info("Running get_solved_outputs to test complete pipeline...")
        responses, arr_responses = get_solved_outputs(arc_input, hint)
        assert len(arr_responses) > 0, "Should generate at least one prediction"
        logger.info("✓ Step 5: get_solved_outputs successful")
        
        # Step 6: Compare predictions with ground truth
        ground_truth_solutions = []
        for test_entry in arc_input['test']:
            if 'output' in test_entry:
                ground_truth_solutions.append(test_entry['output'])
        
        correct_predictions = 0
        total_predictions = min(len(arr_responses), len(ground_truth_solutions))
        
        for i in range(total_predictions):
            prediction = arr_responses[i]
            ground_truth = ground_truth_solutions[i]
            
            if prediction and len(prediction) > 0:
                try:
                    prediction_array = np.array(prediction)
                    ground_truth_array = np.array(ground_truth)
                    
                    if prediction_array.shape == ground_truth_array.shape:
                        if np.array_equal(prediction_array, ground_truth_array):
                            correct_predictions += 1
                            logger.info(f"✓ Prediction {i+1} matches ground truth")
                        else:
                            logger.warning(f"✗ Prediction {i+1} does not match ground truth")
                    else:
                        logger.warning(f"✗ Prediction {i+1} shape mismatch: {prediction_array.shape} vs {ground_truth_array.shape}")
                except Exception as e:
                    logger.error(f"Error comparing prediction {i+1}: {str(e)}")
            else:
                logger.warning(f"✗ Prediction {i+1} is empty or invalid")
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        logger.info(f"✓ Step 6: Accuracy comparison complete - {correct_predictions}/{total_predictions} = {accuracy:.2%}")
        
        logger.info("✓ Solver integration test passed")
        finalize_log_file(temp_log_file, "test_solver_integration", True)
        
    except Exception as e:
        logger.error(f"✗ Solver integration test failed: {str(e)}")
        finalize_log_file(temp_log_file, "test_solver_integration", False)
        raise

def test_single_task_accuracy():
    """Test the accuracy of a single task by running get_solved_outputs and comparing with ground truth"""
    logger, file_handler, temp_log_file = setup_logger("test_single_task_accuracy")
    
    try:
        logger.info("Testing single task accuracy")
        
        # Load hints from hints.json file
        hints_file_path = "testing/samples/solver_data/hints.json"
        try:
            with open(hints_file_path, 'r') as f:
                hints_data = json.load(f)
            logger.info(f"Loaded hints from {hints_file_path}")
        except Exception as e:
            logger.error(f"Failed to load hints from {hints_file_path}: {str(e)}")
            raise
        
        # Use the first available task key
        task_key = list(hints_data.keys())[0]
        logger.info(f"Testing accuracy for task key: {task_key}")
        
        # Load test data from data directory
        data_file_path = f"data/{task_key}.json"
        try:
            with open(data_file_path, 'r') as f:
                arc_input = json.load(f)
            logger.info(f"Loaded test data from {data_file_path}")
        except Exception as e:
            logger.error(f"Failed to load test data from {data_file_path}: {str(e)}")
            raise
        
        hint = hints_data[task_key]
        logger.info(f"Using hint: {hint[:100]}...")
        
        # Run get_solved_outputs to get predictions
        logger.info("Running get_solved_outputs...")
        responses, arr_responses = get_solved_outputs(arc_input, hint)
        logger.info(f"Generated {len(arr_responses)} predictions")
        
        # Get ground truth solutions from test data
        ground_truth_solutions = []
        for test_entry in arc_input['test']:
            if 'output' in test_entry:
                ground_truth_solutions.append(test_entry['output'])
        
        logger.info(f"Found {len(ground_truth_solutions)} ground truth solutions")
        
        # Compare predictions with ground truth
        correct_predictions = 0
        total_predictions = min(len(arr_responses), len(ground_truth_solutions))
        
        logger.info(f"Comparing {total_predictions} predictions with ground truth...")
        
        for i in range(total_predictions):
            prediction = arr_responses[i]
            ground_truth = ground_truth_solutions[i]
            
            logger.info(f"Comparing prediction {i+1}:")
            logger.info(f"  Prediction: {prediction}")
            logger.info(f"  Ground truth: {ground_truth}")
            
            # Convert prediction to numpy array for comparison if it's not empty
            if prediction and len(prediction) > 0:
                try:
                    prediction_array = np.array(prediction)
                    ground_truth_array = np.array(ground_truth)
                    
                    logger.info(f"  Prediction shape: {prediction_array.shape}")
                    logger.info(f"  Ground truth shape: {ground_truth_array.shape}")
                    
                    if prediction_array.shape == ground_truth_array.shape:
                        if np.array_equal(prediction_array, ground_truth_array):
                            correct_predictions += 1
                            logger.info(f"  ✓ MATCH")
                        else:
                            logger.info(f"  ✗ NO MATCH")
                            # Log the differences
                            diff_mask = prediction_array != ground_truth_array
                            diff_count = np.sum(diff_mask)
                            logger.info(f"  Differences: {diff_count} elements differ")
                    else:
                        logger.info(f"  ✗ SHAPE MISMATCH")
                except Exception as e:
                    logger.error(f"  Error comparing prediction {i+1}: {str(e)}")
            else:
                logger.info(f"  ✗ EMPTY PREDICTION")
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        logger.info(f"Final accuracy: {correct_predictions}/{total_predictions} = {accuracy:.2%}")
        
        # Assert that we have at least one prediction
        assert total_predictions > 0, "No predictions were generated"
        
        # Log the accuracy for manual review (don't fail the test based on accuracy)
        if accuracy > 0:
            logger.info(f"✓ Task {task_key} completed with {accuracy:.2%} accuracy")
        else:
            logger.warning(f"⚠ Task {task_key} completed with 0% accuracy - review needed")
        
        finalize_log_file(temp_log_file, "test_single_task_accuracy", True)
        
    except Exception as e:
        logger.error(f"✗ Single task accuracy test failed: {str(e)}")
        finalize_log_file(temp_log_file, "test_single_task_accuracy", False)
        raise

if __name__ == "__main__":
    # Run individual tests
    test_get_formatted_examples()
    test_get_prompts()
    test_extract_matrix_from_response()
    test_matrix_to_arr()
    test_solver_integration()
    test_single_task_accuracy()
    
    # Run parallel tests
    asyncio.run(test_all_solver_functions_parallel())
