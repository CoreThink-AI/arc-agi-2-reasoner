"""
ARC-AGI Core Processing Module
Handles task solving with pattern-based hints and consensus from multiple attempts.
"""

import json, os
import asyncio
import time
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from arc_agi.src.solver.solver import get_solved_outputs_multiple_in_parallel
from arc_agi.src.utils.visualization_utils import plot_grid
import matplotlib.pyplot as plt
from arc_agi.src.objects.base import Grid, BaseObject
from arc_agi.src.patterns.find_patterns import unit_patterns
from arc_agi.src.patterns_intersection.aggregate import intersect
from arc_agi.src.low_hanging.jigsaw import do_jigsaw, check_jigsaw

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

async def process_single_training_example(i, train_example):
    """
    Process a single training example using ThreadPoolExecutor pattern similar to Solver
    
    Args:
        i: Index of the training example
        train_example: Single training example with input/output
        
    Returns:
        tuple: (pattern_params, counts) for this training example
    """
    iteration_start = time.time()
    
    grid_input = train_example["input"]
    grid_output = train_example["output"]
    grid_a = Grid(grid_input)
    input_obj = grid_a.find_objects_in_grid('openai', 'gpt-4.1-mini', 0.0, 4096)
    grid_b = Grid(grid_output)
    output_obj = grid_b.find_objects_in_grid('openai', 'gpt-4.1-mini', 0.0, 4096)
    
    # Create all input object tasks concurrently
    input_tasks = [create_base_object(grid_input, obj) for obj in input_obj]
    before_list = await asyncio.gather(*input_tasks)
    # Create all output object tasks concurrently
    output_tasks = [create_base_object(grid_output, obj) for obj in output_obj]
    after_list = await asyncio.gather(*output_tasks)
    
    pattern_params, counts = await unit_patterns(grid_input, grid_output, before_list, after_list)
    
    iteration_time = time.time() - iteration_start
    
    return pattern_params, counts, iteration_time

async def get_consensus_response(json_data, hint, num_attempts=3,critic=False):
    """
    Get consensus response from multiple solver attempts using majority voting.
    
    Args:
        json_data: ARC task data with train/test examples
        hint: Pattern hint for solving
        num_attempts: Number of solver attempts for consensus
        
    Returns:
        tuple: (consensus_responses, ground_truth_responses)
    """
    print(f"Getting consensus from {num_attempts} attempts...")
    
    # Run solver attempts concurrently
    # tasks = [get_solved_outputs(json_data, hint) for _ in range(num_attempts)]
    # results = await asyncio.gather(*tasks, return_exceptions=True)
    
    results = await get_solved_outputs_multiple_in_parallel(json_data, hint, num_attempts,critic)
    
    # Extract valid responses
    valid_responses = []
    ground_truth = None
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Attempt {i+1} failed: {result}")
            continue
            
        response, gt = result
        if response and all(test_attempts for test_attempts in response):
            valid_responses.append(response)
            if ground_truth is None:
                ground_truth = gt
    
    if not valid_responses:
        print("No valid responses received")
        return [], ground_truth
    
    print(f"Got {len(valid_responses)} valid responses")
    
    # Generate consensus for each test case
    if not valid_responses or not valid_responses[0]:
        return [], ground_truth
        
    num_test_cases = len(valid_responses[0])
    consensus_responses = []
    
    for test_idx in range(num_test_cases):
        # Collect all attempts for this test case from all responses
        test_attempts = []
        for response in valid_responses:
            if test_idx < len(response):
                test_attempts.extend(response[test_idx])  # response[test_idx] is a list of attempts
        
        # Filter valid grids with consistent dimensions
        valid_grids = [grid for grid in test_attempts if _is_valid_grid(grid)]
        
        if not valid_grids:
            consensus_responses.append(None)
            continue
            
        # Use first grid if all attempts agree, otherwise apply majority voting
        if len(set(str(grid) for grid in valid_grids)) == 1:
            consensus_responses.append(valid_grids[0])
        else:
            consensus_grid = _get_majority_vote_grid(valid_grids)
            consensus_responses.append(consensus_grid)
    
    return consensus_responses, ground_truth


def _is_valid_grid(grid):
    """Check if grid is valid (non-empty, rectangular)."""
    return (isinstance(grid, list) and 
            len(grid) > 0 and 
            all(isinstance(row, list) and len(row) > 0 for row in grid) and
            len(set(len(row) for row in grid)) == 1)  # All rows same length


def _get_majority_vote_grid(grids):
    """Create consensus grid using majority voting for each cell."""
    if not grids:
        return None
        
    # Ensure all grids have same dimensions
    height = len(grids[0])
    width = len(grids[0][0])
    
    same_size_grids = [g for g in grids 
                      if len(g) == height and all(len(row) == width for row in g)]
    
    if not same_size_grids:
        return grids[0]  # Fallback to first grid
    
    consensus_grid = []
    for row in range(height):
        consensus_row = []
        for col in range(width):
            # Get majority value for this cell
            cell_values = [grid[row][col] for grid in same_size_grids]
            majority_value = Counter(cell_values).most_common(1)[0][0]
            consensus_row.append(majority_value)
        consensus_grid.append(consensus_row)
    
    return consensus_grid


def visualize_results(responses, ground_truths, title_prefix="Test"):
    """Visualize response and ground truth grids side by side."""
    if not responses and not ground_truths:
        print("No results to visualize")
        return
    
    num_cases = max(len(responses) if responses else 0, 
                   len(ground_truths) if ground_truths else 0)
    
    if num_cases == 0:
        print("No test cases to visualize")
        return
    
    # Create subplot layout: response and ground truth for each test case
    fig, axes = plt.subplots(1, num_cases * 2, figsize=(5 * num_cases * 2, 5))
    
    # Handle single test case
    if num_cases == 1:
        axes = [axes] if num_cases * 2 == 1 else list(axes)
    
    for i in range(num_cases):
        # Plot response
        resp_idx = i * 2
        if i < len(responses) and responses[i] is not None:
            plot_grid(responses[i], f"{title_prefix} {i+1} - Response", axes[resp_idx])
        else:
            axes[resp_idx].set_title(f"{title_prefix} {i+1} - No Response")
            axes[resp_idx].axis('off')
        
        # Plot ground truth
        gt_idx = i * 2 + 1
        if ground_truths and i < len(ground_truths) and ground_truths[i] is not None:
            plot_grid(ground_truths[i], f"{title_prefix} {i+1} - Ground Truth", axes[gt_idx])
        else:
            axes[gt_idx].set_title(f"{title_prefix} {i+1} - No Ground Truth")
            axes[gt_idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def save_results_as_png(responses, ground_truths, task_id, title_prefix="Test"):
    """Save visualization as PNG file in e2e_logs folder."""
    if not responses and not ground_truths:
        print("No results to save")
        return
    
    num_cases = max(len(responses) if responses else 0, 
                   len(ground_truths) if ground_truths else 0)
    
    if num_cases == 0:
        print("No test cases to save")
        return
    
    # Create subplot layout: response and ground truth for each test case
    fig, axes = plt.subplots(1, num_cases * 2, figsize=(5 * num_cases * 2, 5))
    
    # Handle single test case
    if num_cases == 1:
        axes = [axes] if num_cases * 2 == 1 else list(axes)
    
    for i in range(num_cases):
        # Plot response
        resp_idx = i * 2
        if i < len(responses) and responses[i] is not None:
            plot_grid(responses[i], f"{title_prefix} {i+1} - Response", axes[resp_idx])
        else:
            axes[resp_idx].set_title(f"{title_prefix} {i+1} - No Response")
            axes[resp_idx].axis('off')
        
        # Plot ground truth
        gt_idx = i * 2 + 1
        if ground_truths and i < len(ground_truths) and ground_truths[i] is not None:
            plot_grid(ground_truths[i], f"{title_prefix} {i+1} - Ground Truth", axes[gt_idx])
        else:
            axes[gt_idx].set_title(f"{title_prefix} {i+1} - No Ground Truth")
            axes[gt_idx].axis('off')
    
    plt.tight_layout()
    
    # Save as PNG in e2e_logs folder
    png_filename = f"e2e_logs/{task_id}_visualization.png"
    plt.savefig(png_filename, dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Visualization saved to: {png_filename}")


async def solve_arc_task(file_path, hint, num_attempts=3, visualize=True, critic=False, task_id=None):
    """
    Main function to solve an ARC task with given hint.
    
    Args:
        file_path: Path to ARC task JSON file
        hint: Pattern hint for solving
        num_attempts: Number of consensus attempts
        visualize: Whether to show visualization
        critic: Whether to use critic mode
        task_id: Task ID for saving PNG files
        
    Returns:
        tuple: (consensus_responses, ground_truth, execution_time)
    """
    start_time = time.time()
    print(f"Solving ARC task: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None, None, 0
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
        return None, None, 0
    
    print(f"Task has {len(json_data.get('train', []))} training examples and "
          f"{len(json_data.get('test', []))} test examples")
    
    # Get consensus response
    responses, ground_truth = await get_consensus_response(json_data, hint, num_attempts,critic)
    
    execution_time = time.time() - start_time
    print(f"Task solved in {execution_time:.2f} seconds")
    
    # Visualize results or save as PNG
    if responses or ground_truth:
        if visualize:
            visualize_results(responses, ground_truth)
        elif task_id:
            save_results_as_png(responses, ground_truth, task_id)
    
    return responses, ground_truth, execution_time

async def get_hints(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    patterns = []
    all_counts = {}
    
    pattern_detection_start = time.time()
    
    # Process training examples concurrently using asyncio.gather instead of ThreadPoolExecutor
    # This avoids event loop conflicts with asyncio objects
    print(f"Processing {len(json_data['train'])} training examples concurrently...")
    
    try:
        # Create tasks for all training examples
        tasks = [
            process_single_training_example(i, json_data["train"][i]) 
            for i in range(len(json_data["train"]))
        ]
        
        # Process all training examples concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        completed = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Training example {i+1} failed: {result}")
                continue
                
            pattern_params, counts, iteration_time = result
            patterns.extend(pattern_params)
            
            # Accumulate counts from each training example
            for pattern_name, count in counts.items():
                if pattern_name not in all_counts:
                    all_counts[pattern_name] = 0
                all_counts[pattern_name] += count
            
            completed += 1
            print(f"Completed {completed}/{len(json_data['train'])} training examples ({iteration_time:.2f}s)")
            
    except Exception as e:
        print(f"Concurrent processing failed, falling back to sequential: {e}")
        # Fallback to sequential processing if concurrent fails
        for i in range(len(json_data["train"])):
            try:
                pattern_params, counts, iteration_time = await process_single_training_example(i, json_data["train"][i])
                patterns.extend(pattern_params)
                
                # Accumulate counts from each training example
                for pattern_name, count in counts.items():
                    if pattern_name not in all_counts:
                        all_counts[pattern_name] = 0
                    all_counts[pattern_name] += count
                
                print(f"Completed {i+1}/{len(json_data['train'])} training examples ({iteration_time:.2f}s)")
            except Exception as ex:
                print(f"Training example {i+1} failed: {ex}")
    
    pattern_detection_time = time.time() - pattern_detection_start
    print(f"All training examples processed in {pattern_detection_time:.2f}s")    
    # Aggregate patterns after processing all training examples
    aggregation_start = time.time()
    
    # Get top 2 patterns with highest counts
    top_2_patterns = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Filter patterns to only include those in top 2
    top_pattern_names = {pattern[0] for pattern in top_2_patterns}
    filtered_patterns = [p for p in patterns if p.get('name') in top_pattern_names]
    
    restructured_pattern_params, final_counts = await intersect(filtered_patterns)
    aggregation_time = time.time() - aggregation_start
    print("Aggregartion Done")
    # Get solved outputs and visualize
    hint = json.dumps(restructured_pattern_params)
    return hint

def setup_logger_for_id(task_id):
    """Set up a logger for a specific task ID"""
    # Create e2e_logs directory if it doesn't exist
    log_dir = "e2e_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(f"e2e_{task_id}")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create temporary file handler (will rename after getting score)
    temp_log_file = os.path.join(log_dir, f"{task_id}_temp.log")
    file_handler = logging.FileHandler(temp_log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger, temp_log_file

def rename_log_file(temp_log_file, task_id, score_percentage):
    """Rename the temporary log file to include the score"""
    log_dir = "e2e_logs"
    final_log_file = os.path.join(log_dir, f"{task_id}_score_{score_percentage:.1f}%.log")
    
    # Rename the file
    if os.path.exists(temp_log_file):
        os.rename(temp_log_file, final_log_file)
    
    return final_log_file
    
# Example usage
async def main():
    """E2E usage of the ARC solver."""
    ids = [
        #"3e6067c3",
        #"0934a4d8",
        # "135a2760",
        # "1818057f",
        "20a9e565",
        "221dfab4",
        "28a6681f",
        "2ba387bc",
        "2c181942",
        "2d0172a1",
        "3a25b0d8",
        "332f06d7",
        "446ef5d2",
        "45a5af55",
        "4c416de3",
        "4e34c42c",
        "53fb4810",
        "58490d8a",
        "58f5dbd5",
        "5961cc34",
        "64efde09",
        "6e453dd6",
        "71e489b6",
        "7491f3cf",
        "7666fa5d",
        "7b0280bc",
        "7b5033c1"
    ]
    #ids = [f[:-5] for f in os.listdir('data') if f.endswith('.json')]
    overall_score = 0
    overall_count = 0
    
    for task_id in ids:
        # Set up logging for this task
        logger, temp_log_file = setup_logger_for_id(task_id)
        
        try:
            logger.info(f"Starting processing for task {task_id}")
            
            file_path = f"data/{task_id}.json"
            if check_jigsaw(file_path):
                print("Doing Jigsaw")
                responses, ground_truth,solve_time = do_jigsaw(file_path)
                hint_time=0
            else:
                # Get hints
                logger.info("Getting hints...")
                hint_start_time = time.time()
                hint = await get_hints(file_path)
                hint_time = time.time() - hint_start_time
                logger.info(f"Hints completed in {hint_time:.2f}s")
                logger.info(f"Generated hint: {hint}")
                
                # Solve the task
                logger.info("Solving task...")
                solve_start_time = time.time()
                responses, ground_truth, exec_time = await solve_arc_task(
                    file_path=file_path,
                    hint=hint,
                    num_attempts=10,
                    visualize=False,
                    task_id=task_id
                )
                solve_time = time.time() - solve_start_time
            logger.info(f"Task solving completed in {solve_time:.2f}s")
            
            # Calculate score for this task
            task_score = 0
            task_count = 0
            
            if responses:
                solved_count = len([r for r in responses if r is not None])
                logger.info(f"Successfully solved {solved_count} out of {len(responses)} test cases")
                
                for i in range(len(responses)):
                    if responses[i] == ground_truth[i]:
                        task_score += 1
                        logger.info(f"Test case {i+1}: CORRECT")
                    else:
                        logger.info(f"Test case {i+1}: INCORRECT")
                    task_count += 1
            else:
                logger.warning("No valid solutions found")
            
            # Calculate percentage for this task
            task_percentage = (task_score / task_count * 100) if task_count > 0 else 0
            logger.info(f"Task {task_id} Score: {task_score}/{task_count} ({task_percentage:.1f}%)")
            logger.info(f"Total processing time: {hint_time + solve_time:.2f}s")
            
            # Update overall score
            overall_score += task_score
            overall_count += task_count
            
        except Exception as e:
            import traceback
            logger.error(f"Error processing task {task_id}: {str(e)}")
            logger.error(f"Detailed error traceback:\n{traceback.format_exc()}")
            task_percentage = 0
        
        finally:
            # Close logger handlers
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            
            # Rename log file with score
            final_log_file = rename_log_file(temp_log_file, task_id, task_percentage)
            print(f"Log saved to: {final_log_file}")
    
    # Print overall results
    overall_percentage = (overall_score / overall_count * 100) if overall_count > 0 else 0
    print(f"\nOverall Score: {overall_score}/{overall_count} ({overall_percentage:.1f}%)")

if __name__ == "__main__":
    asyncio.run(main())
