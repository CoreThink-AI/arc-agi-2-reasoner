"""
ARC-AGI Core Processing Module
Handles task solving with pattern-based hints and consensus from multiple attempts.
"""

import argparse
import json, os
import asyncio
import time
import logging
from collections import Counter
from arc_agi.src.solver.solver import get_solved_outputs_multiple_in_parallel
from arc_agi.src.utils.visualization_utils import plot_grid
import matplotlib.pyplot as plt
from arc_agi.src.objects.base import Grid, BaseObject
from arc_agi.src.patterns.find_patterns import unit_patterns
from arc_agi.src.patterns_intersection.aggregate import intersect
from arc_agi.src.low_hanging.jigsaw import do_jigsaw, check_jigsaw

# Progress display controls
VERBOSE_PROGRESS = True


def find_task_ids(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)

    def find_all_ids(obj, ids=None):
        """Recursively find all dictionary keys that look like IDs."""
        if ids is None:
            ids = []

        if isinstance(obj, dict):
            for k, v in obj.items():
                ids.append(k)
                find_all_ids(v, ids)
        elif isinstance(obj, list):
            for item in obj:
                find_all_ids(item, ids)

        return ids

    all_ids = find_all_ids(data)

    ids = []
    for id_val in all_ids:
        if id_val not in ["train", "test", "input", "output"]:
            ids.append(id_val)

    return ids


def get_task_data(task_id, data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    return data[task_id]


def create_base_object_sync(grid, coord_tuples):
    """
    Helper: given a 2D list (or array) and a set of (x, y) tuples,
    construct a BaseObject synchronously.
    """
    # BaseObject constructor expects Set[Tuple[int, int]] directly
    # to_dict is async; run it to completion in this thread
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

async def create_base_object_async(grid, coord_tuples):
    """
    Async helper: given a 2D list (or array) and a set of (x, y) tuples,
    construct a BaseObject asynchronously.
    """
    # Await the async to_dict method directly
    data = await BaseObject(grid, coord_tuples).to_dict(
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
    # Measure objects stage (finding objects + creating base objects)
    objects_stage_start = time.time()
    grid_input = train_example["input"]
    grid_output = train_example["output"]
    grid_a = Grid(grid_input)
    grid_b = Grid(grid_output)
    # Run object detection for input and output concurrently
    print("finding objects ....")
    input_obj, output_obj = await asyncio.gather(
        grid_a.find_objects_in_grid('openai', 'gpt-4.1-mini', 0.0, 4096),
        grid_b.find_objects_in_grid('openai', 'gpt-4.1-mini', 0.0, 4096),
    )
    print("creating base objects ...")
    # Create all base object tasks concurrently across input and output
    input_tasks = [create_base_object_async(grid_input, obj) for obj in input_obj]
    output_tasks = [create_base_object_async(grid_output, obj) for obj in output_obj]
    print("combining tasks ...")
    combined_results = await asyncio.gather(*(input_tasks + output_tasks))
    print("task combined ...")
    before_list = combined_results[: len(input_tasks)]
    after_list = combined_results[len(input_tasks) :]

    objects_stage_time = time.time() - objects_stage_start

    # Measure patterns stage
    patterns_stage_start = time.time()
    print("finding patterns ....")
    pattern_params, counts = await unit_patterns(grid_input, grid_output, before_list, after_list)
    patterns_stage_time = time.time() - patterns_stage_start
    print("patterns found .....")

    return pattern_params, counts, objects_stage_time, patterns_stage_time, input_obj, output_obj


async def get_consensus_response(json_data, hint, num_attempts=3, critic=False):
    """
    Get consensus response from multiple solver attempts using majority voting.

    Args:
        json_data: ARC task data with train/test examples
        hint: Pattern hint for solving
        num_attempts: Number of solver attempts for consensus

    Returns:
        list: consensus_responses
    """
    # Run solver attempts concurrently
    # tasks = [get_solved_outputs(json_data, hint) for _ in range(num_attempts)]
    # results = await asyncio.gather(*tasks, return_exceptions=True)

    results = await get_solved_outputs_multiple_in_parallel(json_data, hint, num_attempts, critic)

    # Extract valid responses
    valid_responses = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Attempt {i + 1} failed: {result}")
            continue

        response, _ = result
        if response and all(test_attempts for test_attempts in response):
            valid_responses.append(response)

    if not valid_responses:
        print("No valid responses received")
        return []

    # Single-stage prints handled in caller

    # Generate consensus for each test case
    if not valid_responses or not valid_responses[0]:
        return []

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
            print(f"Test case {test_idx + 1} has no valid grids")
            continue

        # Use first grid if all attempts agree, otherwise apply majority voting
        if len(set(str(grid) for grid in valid_grids)) == 1:
            consensus_responses.append(valid_grids[0])
        else:
            consensus_grid = _get_majority_vote_grid(valid_grids)
            consensus_responses.append(consensus_grid)

    print(f"Consensus responses generated for {len(consensus_responses)} test cases")
    return consensus_responses


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


def visualize_results(responses, title_prefix="Test"):
    """Visualize response grids."""
    if not responses:
        print("No results to visualize")
        return

    num_cases = len(responses)

    if num_cases == 0:
        print("No test cases to visualize")
        return

    fig, axes = plt.subplots(1, num_cases, figsize=(5 * num_cases, 5))

    if num_cases == 1:
        axes = [axes]

    for i in range(num_cases):
        if i < len(responses) and responses[i] is not None:
            plot_grid(responses[i], f"{title_prefix} {i + 1} - Response", axes[i])
        else:
            axes[i].set_title(f"{title_prefix} {i + 1} - No Response")
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def save_results_as_png(responses, task_id, title_prefix="Test"):
    """Save visualization as PNG file in e2e_logs folder."""
    if not responses:
        print("No results to save")
        return

    num_cases = len(responses)

    if num_cases == 0:
        print("No test cases to save")
        return

    fig, axes = plt.subplots(1, num_cases, figsize=(5 * num_cases, 5))

    if num_cases == 1:
        axes = [axes]

    for i in range(num_cases):
        if i < len(responses) and responses[i] is not None:
            plot_grid(responses[i], f"{title_prefix} {i + 1} - Response", axes[i])
        else:
            axes[i].set_title(f"{title_prefix} {i + 1} - No Response")
            axes[i].axis('off')

    plt.tight_layout()

    # Save as PNG in e2e_logs folder
    png_filename = f"e2e_logs/{task_id}_visualization.png"
    plt.savefig(png_filename, dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Visualization saved to: {png_filename}")


async def solve_arc_task(json_data, hint, num_attempts=3, visualize=True, critic=False, task_id=None):
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
        tuple: (consensus_responses, execution_time)
    """
    start_time = time.time()

    if check_jigsaw(json_data):
        print("Doing Jigsaw")
        responses, _, solve_time = do_jigsaw(json_data)
        return responses, solve_time
    if VERBOSE_PROGRESS:
        print(
            f"Task has {len(json_data.get('train', []))} training examples and "
            f"{len(json_data.get('test', []))} test examples"
        )

    # Get consensus response (single-stage print)
    if VERBOSE_PROGRESS:
        print(f"Running Consensus with {num_attempts} attempts...")
    consensus_start = time.time()
    responses = await get_consensus_response(json_data, hint, num_attempts, critic)
    consensus_time = time.time() - consensus_start
    if VERBOSE_PROGRESS:
        print(f"Consensus took {consensus_time:.2f}s")

    execution_time = time.time() - start_time
    # Overall solve time print removed to keep single print per stage

    # Visualize results or save as PNG
    if responses:
        if visualize:
            visualize_results(responses)
        elif task_id:
            save_results_as_png(responses, task_id)

    return responses, execution_time


async def solve_arc_task_2(json_data, objects):
    from arc_agi.src.utils.llm_utils import aget_grok_response_stream
    from arc_agi.src.solver.solver_2_utils import get_prompts

    api_key = os.getenv("XAI_API_KEY_FLOW_2")
    if not api_key:
        raise ValueError("Missing XAI API key for flow 2. Set it as environment variable 'XAI_API_KEY_FLOW_2'.")

    system_prompt = (
        "You are an expert at solving grid-based problems. "
        "You are given a grid and a format to write the solution. "
        "Always use the provided format and enclose your code within triple backticks."
    )

    prompts = get_prompts(json_data, objects)

    # Support both single prompt or list of prompts
    if isinstance(prompts, str):
        return await aget_grok_response_stream(prompts, system_prompt, api_key)

    # Run all prompts in parallel
    results = await asyncio.gather(
        *(aget_grok_response_stream(p, system_prompt, api_key) for p in prompts)
    )
    return results


async def get_hints(json_data):
    objects_train = []
    patterns = []
    all_counts = {}
    total_objects_time = 0.0
    total_patterns_time = 0.0

    # Process training examples concurrently

    try:
        if VERBOSE_PROGRESS:
            print("Finding Objects & Patterns...")
        # Create tasks for all training examples
        # ✅ Keep input/output objects for objects_train in concurrent path
        async def process_and_return_all(i, example):
            # unpack all 6 returned values
            pattern_params, counts, objects_time, patterns_time, input_obj, output_obj = \
                await process_single_training_example(i, example)
            return pattern_params, counts, objects_time, patterns_time, [i, input_obj, output_obj]

        print("starting pattern post processing...")
        tasks = []
        for i, example in enumerate(json_data["train"]):
            # create a task that keeps all required data
            task = asyncio.create_task(process_and_return_all(i, example))
            tasks.append(task)

        # Process all training examples concurrently
        for completed_task in asyncio.as_completed(tasks):
            try:
                pattern_params, counts, objects_time, patterns_time, obj_info = await completed_task
            except Exception as err:
                if VERBOSE_PROGRESS:
                    print(f"A training example failed: {err}")
                continue

            # collect patterns
            patterns.extend(pattern_params)

            # accumulate counts
            for pattern_name, count in counts.items():
                all_counts[pattern_name] = all_counts.get(pattern_name, 0) + count

            # store objects for flow2
            objects_train.append(obj_info)

            # accumulate time
            total_objects_time += objects_time
            total_patterns_time += patterns_time

        print("Finished pattern post processing.")

    except Exception as e:
        print(f"Concurrent processing failed, falling back to sequential: {e}")
        # Fallback to sequential processing if concurrent fails
        for i in range(len(json_data["train"])):
            try:
                pattern_params, counts, objects_time, patterns_time, input_obj, output_obj = \
                    await process_single_training_example(i, json_data["train"][i])

                objects_train.append([i, input_obj, output_obj])
                patterns.extend(pattern_params)

                # Accumulate counts from each training example
                for pattern_name, count in counts.items():
                    if pattern_name not in all_counts:
                        all_counts[pattern_name] = 0
                    all_counts[pattern_name] += count
                total_objects_time += objects_time
                total_patterns_time += patterns_time
            except Exception as ex:
                print(f"Training example {i + 1} failed: {ex}")

    # Single prints for stages
    if VERBOSE_PROGRESS:
        print(f"Finding Objects took {total_objects_time:.2f}s")
        print(f"Finding Patterns took {total_patterns_time:.2f}s")

    # Aggregate patterns after processing all training examples
    if VERBOSE_PROGRESS:
        print("Aggregating Patterns...")
    aggregation_start = time.time()

    # Get top 2 patterns with highest counts
    top_2_patterns = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    # Filter patterns to only include those in top 2
    top_pattern_names = {pattern[0] for pattern in top_2_patterns}
    filtered_patterns = [p for p in patterns if p.get('name') in top_pattern_names]

    restructured_pattern_params, final_counts = await intersect(filtered_patterns)
    aggregation_time = time.time() - aggregation_start
    if VERBOSE_PROGRESS:
        print(f"Aggregation took {aggregation_time:.2f}s")
    # Get solved outputs and visualize
    hint = json.dumps(restructured_pattern_params)
    return hint, objects_train


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

    # Create temporary file handler (will rename after processing)
    temp_log_file = os.path.join(log_dir, f"{task_id}_temp.log")
    file_handler = logging.FileHandler(temp_log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    return logger, temp_log_file


def rename_log_file(temp_log_file, task_id):
    """Rename the temporary log file to final name without score."""
    log_dir = "e2e_logs"
    final_log_file = os.path.join(log_dir, f"{task_id}.log")

    # Rename the file
    if os.path.exists(temp_log_file):
        os.rename(temp_log_file, final_log_file)

    return final_log_file


async def process_task_get_hints(task_id):
    logger, temp_log_file = setup_logger_for_id(task_id)

    try:
        logger.info(f"Starting getting hints for task {task_id}")
        if VERBOSE_PROGRESS:
            print(f"Task {task_id}: Getting hints...")

        task_json_data = get_task_data(
            task_id, "arc-agi_test_challenges.json"
        )

        # Get hints
        hint_start_time = time.time()
        hint, objects_train = await get_hints(task_json_data)
        hint_time = time.time() - hint_start_time
        logger.info(f"Hints completed in {hint_time:.2f}s")
        if VERBOSE_PROGRESS:
            print(f"Task {task_id}: Hints completed in {hint_time:.2f}s")
        logger.info(f"Generated hint: {hint}")

        return task_id, hint, objects_train

    except Exception as e:
        logger.error(f"Error processing {task_id}: {e}")
        return task_id, None


# Example usage
async def main(j):
    # Start time for the overall run
    main_start = time.time()
    print(f"Main start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(main_start))}")
    for i in range(int(240/j)):
        print(f"Running batch {i + 1}")
        ids = find_task_ids("arc-agi_test_challenges.json")
        ids = ids[i*j : i*j + j]
        async def process_and_solve(task_id):
            logger, temp_log_file = setup_logger_for_id(task_id)
            try:
                logger.info(f"Starting processing for task {task_id}")
                task_json_data = get_task_data(
                    task_id, "arc-agi_test_challenges.json"
                )

                # Generate hint
                _, hint, objects_train = await process_task_get_hints(task_id)
                print(task_id)

                # Solve as soon as hint is ready (flow-1)
                async def run_flow1():
                    logger.info(f"Solving task {task_id} - flow 1")
                    start_solve = time.time()
                    responses, _ = await solve_arc_task(
                        json_data=task_json_data,
                        hint=hint,
                        num_attempts=3,
                        visualize=False,
                        task_id=task_id
                    )
                    logger.info(f"Solved in {time.time() - start_solve:.2f}s - flow 1")
                    return responses

                # Solve as soon as hint is ready (flow-2)
                async def run_flow2():
                    logger.info(f"Solving task {task_id} - flow 2")
                    start_solve = time.time()
                    print(len(objects_train))
                    response = await solve_arc_task_2(task_json_data, objects_train)
                    logger.info(f"Solved in {time.time() - start_solve:.2f}s - flow 2")
                    return response

                # Run both flows concurrently and wait for both to complete
                responses, response = await asyncio.gather(run_flow1(), run_flow2())

                # ✅ Write results to a file
                output_file = f"outputs/task_results_{task_id}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    import json
                    json.dump(
                        {
                            "task_id": task_id,
                            "flow1_responses": responses,
                            "flow2_response": response
                        },
                        f,
                        ensure_ascii=False,
                        indent=2
                    )
                logger.info(f"Results saved to {output_file}")

                # No ground truth; return just log info
                return temp_log_file, task_id

            except Exception as e:
                logger.exception(f"Task {task_id} failed")
                return temp_log_file, task_id

        # Create and run all tasks concurrently, solve each immediately after hint
        process_tasks = [asyncio.create_task(process_and_solve(task_id)) for task_id in ids]

        for coro in asyncio.as_completed(process_tasks):
            temp_log_file, task_id = await coro
            # Safe file rename here
            rename_log_file(temp_log_file, task_id)

        # End time for the overall run
        main_end = time.time()
        print(f"Main end: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(main_end))}")
        print(f"Main duration: {main_end - main_start:.2f}s")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=0, help='batch size')
    args = parser.parse_args()
    asyncio.run(main(args.batch_size))
