"""
ARC-AGI Core Processing Module
Handles task solving with pattern-based hints and consensus from multiple attempts.
"""

import json, os
import asyncio
import time
from collections import Counter
from arc_agi.src.solver.solver import get_solved_outputs_multiple_in_parallel
from arc_agi.src.utils.visualization_utils import plot_grid
import matplotlib.pyplot as plt
from arc_agi.src.objects.base import Grid, BaseObject
from arc_agi.src.patterns.find_patterns import unit_patterns
from arc_agi.src.patterns_intersection.aggregate import intersect

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

async def solve_arc_task(file_path, hint, num_attempts=3, visualize=True,critic=False):
    """
    Main function to solve an ARC task with given hint.
    
    Args:
        file_path: Path to ARC task JSON file
        hint: Pattern hint for solving
        num_attempts: Number of consensus attempts
        visualize: Whether to show visualization
        
    Returns:
        tuple: (consensus_responses, ground_truth, execution_time)
    """
    start_time = time.time()
    print(f"Solving ARC task: {file_path}")
    
    try:
        with open(file_path) as f:
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
    
    # Visualize results
    if visualize and (responses or ground_truth):
        visualize_results(responses, ground_truth)
    
    return responses, ground_truth, execution_time

async def get_hints(file_path):
    with open(file_path) as f:
        json_data = json.load(f)

    patterns = []
    all_counts = {}
    
    pattern_detection_start = time.time()
    for i in range(len(json_data["train"])):
        iteration_start = time.time()
        
        grid_input = json_data["train"][i]["input"]
        grid_output = json_data["train"][i]["output"]
        grid_a = Grid(grid_input)
        input_obj = grid_a.find_objects_in_grid('openai', 'gpt-4.1-mini', 0.0, 4096)
        input_obj = grid_a.objects_concatenator('openai', 'gpt-4.1-mini', 0.0, 4096)
        grid_b = Grid(grid_output)
        output_obj = grid_b.find_objects_in_grid('openai', 'gpt-4.1-mini', 0.0, 4096)
        output_obj = grid_b.objects_concatenator('openai', 'gpt-4.1-mini', 0.0, 4096)
        
        object_creation_start = time.time()
        # Create all input object tasks concurrently
        input_tasks = [create_base_object(grid_input, obj) for obj in input_obj]
        before_list = await asyncio.gather(*input_tasks)
        # Create all output object tasks concurrently
        output_tasks = [create_base_object(grid_output, obj) for obj in output_obj]
        after_list = await asyncio.gather(*output_tasks)
        object_creation_time = time.time() - object_creation_start
        
        pattern_finding_start = time.time()
        pattern_params, counts = await unit_patterns(grid_input, grid_output, before_list, after_list)
        pattern_finding_time = time.time() - pattern_finding_start
        patterns.extend(pattern_params)
        
        # Accumulate counts from each training example
        for pattern_name, count in counts.items():
            if pattern_name not in all_counts:
                all_counts[pattern_name] = 0
            all_counts[pattern_name] += count
        
        iteration_time = time.time() - iteration_start
    
    pattern_detection_time = time.time() - pattern_detection_start    
    # Aggregate patterns after processing all training examples
    aggregation_start = time.time()
    
    # Get top 2 patterns with highest counts
    top_2_patterns = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)[:2]
    
    # Filter patterns to only include those in top 2
    top_pattern_names = {pattern[0] for pattern in top_2_patterns}
    filtered_patterns = [p for p in patterns if p.get('name') in top_pattern_names]
    
    restructured_pattern_params, final_counts = await intersect(filtered_patterns)
    aggregation_time = time.time() - aggregation_start
    
    # Get solved outputs and visualize
    hint = json.dumps(restructured_pattern_params)
    return hint
    
# Example usage
async def main():
    """E2E usage of the ARC solver."""
    ids = []
    ids = [f[:-5] for f in os.listdir('data') if f.endswith('.json')]
    score = 0
    count = 0
    for id in ids:
        file_path = f"data/{id}.json"
        hint = await get_hints(file_path)
        responses, ground_truth, exec_time = await solve_arc_task(
            file_path=file_path,
            hint=hint,
            num_attempts=10,
            visualize=False
        )
        if responses:
            print(f"Successfully solved {len([r for r in responses if r is not None])} out of {len(responses)} test cases")
        else:
            print("No valid solutions found")
        for i in range(len(responses)):
            if responses[i]==ground_truth[i]:
                score+=1
            count+=1
    print("Score: ", score/count*100)

if __name__ == "__main__":
    asyncio.run(main())
