"""
ARC-AGI Core Processing Module
Handles task solving with pattern-based hints and consensus from multiple attempts.
"""

import json
import asyncio
import time
from collections import Counter
from arc_agi.src.solver.solver import get_solved_outputs, get_solved_outputs_multiple_in_parallel
from arc_agi.src.utils.visualization_utils import plot_grid
import matplotlib.pyplot as plt

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

# Example usage
async def main():
    """Example usage of the ARC solver."""
    file_path = "data/28a6681f.json"
    hint = "### Task: Gravity-Driven Cavity Filling with Blue Cells\n\n#### Color Index Reference\n- **Color 1 (Blue)** — Mobile filler cells that will “fall” into cavities under a gravity effect.\n- **Color 0 (Background/Voids)** — Empty space and cavities to be filled.\n- **Color 2+ (Other Colors)** — Immovable obstacles; define the boundaries of cavities.\n\n---\n\n#### Input  \nYou are given a 2D grid of size *H×W* containing:\n- **Blue cells (1):** A fixed number of filler cells that can move vertically under gravity.\n- **Void cells (0):** Empty spaces that represent cavities.\n- **Obstacle cells (≥2):** Walls or fixed regions that define cavity boundaries.\n\n---\n\n#### Objective  \n1. **Detect all vertical cavities** that are bounded on both left and right by obstacle cells (i.e., each row segment of zeros whose immediate neighbors on left and right are non-zero).\n2. **Simulate gravity** by letting Blue cells “fall” straight down into these cavities from above, filling from the bottom up.\n3. **Preserve the total count** of Blue cells; no Blue cell is created or destroyed.\n4. **Produce an updated grid** where voids within bounded cavities are filled as far as possible by Blue cells under gravity.\n\n---\n\n#### Step-by-Step Instructions\n\n1. **Initialize**  \n   - Read the input grid `G[H][W]`.  \n   - Prepare an output grid `H_grid ← G` for the final state.\n\n2. **Locate Bounded Cavities**  \n   - For each row *r* and each column segment `c_start…c_end` where `G[r][c] == 0` for all `c_start ≤ c ≤ c_end`, check that:  \n     - `G[r][c_start – 1] ≥ 2` (left obstacle) and  \n     - `G[r][c_end + 1] ≥ 2` (right obstacle).  \n   - Record all such row‐segments as “cavity cells.”\n\n3. **Count Blue Cells Above Cavities**  \n   - For each cavity cell `(r, c)` in a bounded segment, look upward in column *c* from row `0` to `r–1` and count all Blue cells (`1`) that are not already assigned to another cavity fill.  \n   - Aggregate these counts per cavity segment.\n\n4. **Simulate Gravity Filling**  \n   - For each cavity segment in bottom‐up order (largest *r* first):  \n     a. Let *k* = number of available Blue cells above that segment.  \n     b. For rows `r` down to `r – k + 1`, set `H_grid[row][c] = 1` to drop Blue cells into the lowest empty spots.  \n     c. Mark those *k* Blue cells in the source columns as “used” (so they won’t fall again).  \n     d. Leave any remaining voids (`0`) if Blue cells are exhausted.\n\n5. **Preserve Remaining Grid**  \n   - All non‐cavity zeros that aren’t bounded or that lie outside the simulated falls remain `0`.  \n   - Obstacle cells (≥2) remain unchanged.  \n   - Any Blue cells not used to fill cavities stay in their original positions in `H_grid`.\n\n6. **Finalize Output**  \n   - Return `H_grid`, now with gravity‐filled bounded cavities and the same total count of Blue cells as the input.\n\n---\n\n#### Constraints\n- Cavities must be strictly horizontally bounded by obstacle cells on both sides in the same row.\n- Gravity acts only downward; Blue cells do not move horizontally or upward.\n- Total number of Blue cells in the output must equal the input count.\n- Obstacle cells (colors ≥2) are fixed and impermeable.\n\n---\n\n#### Output  \nA 2D grid of size *H×W* in which all possible bounded cavities have been filled from the bottom up by Blue cells under gravity, with no change in total Blue‐cell count and all obstacle positions preserved."
    responses, ground_truth, exec_time = await solve_arc_task(
        file_path=file_path,
        hint=hint,
        num_attempts=10,
        visualize=True
    )
    print(exec_time)
    if responses:
        print(f"Successfully solved {len([r for r in responses if r is not None])} out of {len(responses)} test cases")
    else:
        print("No valid solutions found")

if __name__ == "__main__":
    asyncio.run(main())
