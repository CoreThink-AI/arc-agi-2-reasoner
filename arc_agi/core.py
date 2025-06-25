from collections import Counter
from arc_agi.src.objects.base import Grid, BaseObject
from arc_agi.src.patterns.find_patterns import unit_patterns
import json
import asyncio
from arc_agi.src.patterns_intersection.aggregate import intersect
from arc_agi.src.solver.solver import get_solved_outputs
from arc_agi.src.utils.visualization_utils import plot_grid
import matplotlib.pyplot as plt
import time


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

async def get_consensus_response(json_data, hint, num_attempts=1):
    """
    Call get_solved_outputs multiple times and use majority voting for final response.
    
    Args:
        json_data: The ARC task data
        hint: The hint/pattern information 
        num_attempts: Number of times to call get_solved_outputs (default 3)
    
    Returns:
        consensus_response: List of grids with majority-voted values
        ground_truth: Ground truth grids (same for all attempts)
    """
    print(f"Getting {num_attempts} responses for consensus (concurrently)...")
    
    # Create tasks for concurrent execution
    async def get_single_response():
        # Since get_solved_outputs is already an async function, await it directly
        return await get_solved_outputs(json_data, hint)
    
    # Run all attempts concurrently
    tasks = [get_single_response() for _ in range(num_attempts)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Separate responses and ground truth, handle exceptions
    all_responses = []
    ground_truth = None
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  Attempt {i+1} failed: {result}")
            continue
        
        response, gt = result
        print(f"  Attempt {i+1} - Response type: {type(response)}, Length: {len(response) if hasattr(response, '__len__') else 'no length'}")
        if response and len(response) > 0:
            print(f"    First item type: {type(response[0])}")
            print(f"    First item content: {response[0]}")
        
        all_responses.append(response)
        if ground_truth is None:
            ground_truth = gt  # Ground truth is same for all attempts
        print(f"  Attempt {i+1} completed successfully")
    
    if not all_responses or not all_responses[0]:
        print("No valid responses received")
        return [], ground_truth
    
    print(f"Successfully got {len(all_responses)} valid responses out of {num_attempts} attempts")
    
    # Number of test cases
    num_test_cases = len(all_responses[0])
    consensus_response = []
    
    for test_idx in range(num_test_cases):
        print(f"Processing consensus for test case {test_idx+1}")
        
        # Get all responses for this test case and filter valid ones
        test_responses = []
        for attempt_idx, attempt_responses in enumerate(all_responses):
            if test_idx < len(attempt_responses) and attempt_responses[test_idx] is not None:
                grid = attempt_responses[test_idx]
                print(f"  Attempt {attempt_idx+1} for test case {test_idx+1}: type={type(grid)}, content={grid}")
                
                # Validate grid format before adding
                if isinstance(grid, list) and len(grid) > 0 and all(isinstance(row, list) for row in grid):
                    test_responses.append(grid)
                else:
                    print(f"  Skipping invalid grid format in attempt {attempt_idx+1} for test case {test_idx+1}")
            else:
                print(f"  Attempt {attempt_idx+1} for test case {test_idx+1}: None or missing")
        
        if not test_responses:
            print(f"No valid responses for test case {test_idx+1}")
            consensus_response.append(None)
            continue
        
        print(f"  Using {len(test_responses)} valid responses out of {len(all_responses)} attempts")
        
        # Get grid dimensions from first valid response
        first_grid = test_responses[0]
        height = len(first_grid)
        width = len(first_grid[0]) if height > 0 else 0
        
        # Verify all valid grids have same dimensions, filter if needed
        valid_same_size_responses = []
        for response_grid in test_responses:
            if (len(response_grid) == height and 
                all(len(row) == width for row in response_grid)):
                valid_same_size_responses.append(response_grid)
            else:
                print(f"  Skipping grid with different dimensions: {len(response_grid)}x{len(response_grid[0]) if response_grid else 0} vs expected {height}x{width}")
        
        if not valid_same_size_responses:
            print(f"No valid same-size responses for test case {test_idx+1}")
            consensus_response.append(None)
            continue
        
        print(f"  Using {len(valid_same_size_responses)} same-size valid responses")
        
        # Create consensus grid
        consensus_grid = []
        for row in range(height):
            consensus_row = []
            for col in range(width):
                # Collect values from all valid responses for this position
                cell_values = []
                for response_grid in valid_same_size_responses:
                    cell_values.append(response_grid[row][col])
                
                # Use majority voting
                value_counts = Counter(cell_values)
                majority_value = value_counts.most_common(1)[0][0]
                consensus_row.append(majority_value)
            
            consensus_grid.append(consensus_row)
        
        consensus_response.append(consensus_grid)
        print(f"Consensus grid for test case {test_idx+1}: {len(consensus_grid)}x{len(consensus_grid[0]) if consensus_grid else 0}")
    
    return consensus_response, ground_truth

file_path = "data/1818057f.json"
async def main():
    start_time = time.time()
    print(f"Starting processing at {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    
    with open(file_path) as f:
        json_data = json.load(f)

    patterns = []
    all_counts = {}
    
    pattern_detection_start = time.time()
    for i in range(len(json_data["train"])):
        print(f"Processing training example {i}")
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
        object_creation_time = time.time() - object_creation_start
        print(f"Object creation took: {object_creation_time:.2f} seconds")
        
        pattern_finding_start = time.time()
        print("Finding Patterns")
        pattern_params, counts = await unit_patterns(grid_input, grid_output, before_list, after_list)
        pattern_finding_time = time.time() - pattern_finding_start
        print(f"Pattern finding took: {pattern_finding_time:.2f} seconds")
        
        print(pattern_params)
        print(counts)
        patterns.extend(pattern_params)
        
        # Accumulate counts from each training example
        for pattern_name, count in counts.items():
            if pattern_name not in all_counts:
                all_counts[pattern_name] = 0
            all_counts[pattern_name] += count
        
        iteration_time = time.time() - iteration_start
        print(f"Training example {i} completed in: {iteration_time:.2f} seconds")
        print("-" * 50)
    
    pattern_detection_time = time.time() - pattern_detection_start
    print(f"Total pattern detection time: {pattern_detection_time:.2f} seconds")
    
    # Aggregate patterns after processing all training examples
    aggregation_start = time.time()
    print("Aggregating Patterns")
    print("All counts:", all_counts)
    
    # Get top 2 patterns with highest counts
    top_2_patterns = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)[:2]
    print(f"Top 2 patterns: {top_2_patterns}")
    
    # Filter patterns to only include those in top 2
    top_pattern_names = {pattern[0] for pattern in top_2_patterns}
    filtered_patterns = [p for p in patterns if p.get('name') in top_pattern_names]
    print(f"Filtered to {len(filtered_patterns)} patterns from top 2 categories")
    
    restructured_pattern_params, final_counts = await intersect(filtered_patterns)
    aggregation_time = time.time() - aggregation_start
    print(f"Pattern aggregation took: {aggregation_time:.2f} seconds")
    
    print(restructured_pattern_params)
    print(final_counts)
    
    # Get solved outputs and visualize
    solving_start = time.time()
    #hint = json.dumps(restructured_pattern_params)
    hint = "You have to recolor the plus which can be found at the edges of the objects in the input image."
    # Debug: Check the structure of the JSON data
    print(f"Number of training examples: {len(json_data.get('train', []))}")
    print(f"Number of test examples: {len(json_data.get('test', []))}")
    
    response, ground_truth = await get_consensus_response(json_data, hint)
    solving_time = time.time() - solving_start
    print(f"Solving took: {solving_time:.2f} seconds")
    
    print("Response type:", type(response), "Length:", len(response) if hasattr(response, '__len__') else 'no length')
    print("Ground truth type:", type(ground_truth), "Length:", len(ground_truth) if hasattr(ground_truth, '__len__') else 'no length')
    
    if response:
        print("First response item type:", type(response[0]) if len(response) > 0 else 'empty')
        if len(response) > 0 and response[0] is not None:
            print("First response grid shape:", f"{len(response[0])}x{len(response[0][0]) if response[0] else 0}")
    if ground_truth:
        print("First ground truth item type:", type(ground_truth[0]) if len(ground_truth) > 0 else 'empty')
        if len(ground_truth) > 0 and ground_truth[0] is not None:
            print("First ground truth grid shape:", f"{len(ground_truth[0])}x{len(ground_truth[0][0]) if ground_truth[0] else 0}")
    
    print("Response:", response)
    print("Ground truth:", ground_truth)
    
    # Debug each item in response and ground truth
    print("\n--- Detailed Debug ---")
    print(f"Response array length: {len(response)}")
    for i, resp in enumerate(response):
        if resp is not None:
            print(f"Response[{i}]: {type(resp)}, shape: {len(resp)}x{len(resp[0]) if resp and len(resp) > 0 else 0}")
        else:
            print(f"Response[{i}]: None")
    
    print(f"Ground truth array length: {len(ground_truth) if ground_truth is not None else 0}")
    if ground_truth is not None:
        for i, gt in enumerate(ground_truth):
            if gt is not None:
                print(f"Ground truth[{i}]: {type(gt)}, shape: {len(gt)}x{len(gt[0]) if gt and len(gt) > 0 else 0}")
            else:
                print(f"Ground truth[{i}]: None")
    else:
        print("Ground truth is None")
    print("--- End Debug ---\n")
    
    # Plot response and ground truth grids
    num_grids = max(len(response), len(ground_truth) if ground_truth is not None else 0)
    if num_grids > 0:
        # Create side-by-side layout: each pair (response, ground_truth) is adjacent
        total_subplots = num_grids * 2
        fig, axes = plt.subplots(1, total_subplots, figsize=(5*total_subplots, 5))
        
        # Handle the case where there's only one subplot pair
        if total_subplots == 2:
            # axes is a 1D array with 2 elements
            print(f"Axes type: {type(axes)}, shape: {axes.shape if hasattr(axes, 'shape') else 'no shape'}")
        
        # Ensure axes is iterable even for single subplot
        if total_subplots == 2:
            axes_list = axes
        else:
            axes_list = axes
        
        for i in range(num_grids):
            print(f"Processing grid {i}, accessing axes[{i*2}] and axes[{i*2 + 1}]")
            
            # Plot response grid on the left of each pair
            if i < len(response) and response[i] is not None:
                print(f"  Plotting response {i}")
                plot_grid(response[i], f"Response {i+1}", axes_list[i*2])
            else:
                print(f"  No response for {i}, setting empty")
                axes_list[i*2].set_title(f"Response {i+1} (No Grid)")
                axes_list[i*2].axis('off')
            
            # Plot ground truth grid on the right of each pair
            if i < len(ground_truth) and ground_truth[i] is not None:
                print(f"  Plotting ground truth {i}")
                plot_grid(ground_truth[i], f"Ground Truth {i+1}", axes_list[i*2 + 1])
            else:
                print(f"  No ground truth for {i}, setting empty")
                axes_list[i*2 + 1].set_title(f"Ground Truth {i+1} (No Grid)")
                axes_list[i*2 + 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")

if __name__ == "__main__":
    asyncio.run(main())