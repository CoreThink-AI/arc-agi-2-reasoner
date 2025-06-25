"""
ARC-AGI Core Processing Module
Handles task solving with pattern-based hints and consensus from multiple attempts.
"""

import json
import asyncio
import time
from collections import Counter
from arc_agi.src.solver.solver import get_solved_outputs
from arc_agi.src.utils.visualization_utils import plot_grid
import matplotlib.pyplot as plt


async def get_consensus_response(json_data, hint, num_attempts=3):
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
    tasks = [get_solved_outputs(json_data, hint) for _ in range(num_attempts)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Extract valid responses
    valid_responses = []
    ground_truth = None
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Attempt {i+1} failed: {result}")
            continue
            
        response, gt = result
        if response and all(grid is not None for grid in response):
            valid_responses.append(response)
            if ground_truth is None:
                ground_truth = gt
    
    if not valid_responses:
        print("No valid responses received")
        return [], ground_truth
    
    print(f"Got {len(valid_responses)} valid responses")
    
    # Generate consensus for each test case
    if not valid_responses[0]:
        return [], ground_truth
        
    num_test_cases = len(valid_responses[0])
    consensus_responses = []
    
    for test_idx in range(num_test_cases):
        test_grids = [resp[test_idx] for resp in valid_responses if test_idx < len(resp)]
        
        # Filter valid grids with consistent dimensions
        valid_grids = [grid for grid in test_grids if _is_valid_grid(grid)]
        
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


async def solve_arc_task(file_path, hint, num_attempts=3, visualize=True):
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
    responses, ground_truth = await get_consensus_response(json_data, hint, num_attempts)
    
    execution_time = time.time() - start_time
    print(f"Task solved in {execution_time:.2f} seconds")
    
    # Visualize results
    if visualize and (responses or ground_truth):
        visualize_results(responses, ground_truth)
    
    return responses, ground_truth, execution_time


# Example usage
async def main():
    """Example usage of the ARC solver."""
    file_path = "data/71e489b6.json"
    hint = """
        ### Task: Surround Voids and Clean Blue Noise

#### Color Index Reference
- **Color 1 (Blue)** — Surface area.
- **Color 0 (Black)** — Background and voids.
- **Color 7 (Orange)** — Used to surround voids.

---

#### Input
You are given a grid with three types of cells:
- **Blue (1):** Represents solid surface regions.
- **Black (0):** Represents background and potential voids.
- **Orange (7):** Will be used to surround detected voids.

The grid contains wide horizontal **black bands** which separate the blue zones. Scattered black pixels appearing inside blue areas are considered **voids**. There may also be isolated blue pixels (noise) in black areas.

---

#### Objective
1. **Detect voids** — black cells inside blue zones (not inside black bands).
2. **Surround each valid void** with **Color 7 (orange)** in all 4-connected directions.
3. **Remove stray blue noise** — isolated or out-of-place blue cells not forming a proper region.

---

#### Step-by-Step Instructions

1. **Locate Blue Zones**
   - Focus on areas that contain contiguous horizontal bands of **Color 1 (blue)**.
   - Ignore large solid horizontal **black (0)** bands — they are separators, not zones.

2. **Detect Valid Voids (Black in Blue)**
   - Inside each blue zone, identify **black (0)** pixels.
   - A pixel is a valid void if it is **surrounded on at least 3 sides by blue (1)**.

3. **Add Orange Boundary (Color 7)**
   - For each valid black void:
     - Check its **4-connected neighbors** (up, down, left, right).
     - If any neighbor is blue (1), change it to **orange (7)**.
   - This creates a neat border around the void.

4. **Clean Up Blue Noise**
   - After surrounding voids:
     - Traverse the entire grid.
     - Any **blue (1)** pixel that is isolated or not part of a blue region/void structure should be removed (set to black `0`).

---

#### Constraints

- ❌ Do **not** surround voids in black bands — only those inside blue zones.
- ❌ Do **not** overwrite the void cells — they must remain black (0).
- ❌ Use only **4-connected** directions (no diagonal filling).
- ❌ Do **not** leave leftover or misaligned blue (1) cells in the final grid.

---

#### Output
A refined grid where:
- All voids inside blue zones are surrounded cleanly by **Color 7 (orange)**.
- Unwanted blue pixels are removed, resulting in a clean visual layout.

    """
    responses, ground_truth, exec_time = await solve_arc_task(
        file_path=file_path,
        hint=hint,
        num_attempts=3,
        visualize=True
    )
    print(exec_time)
    if responses:
        print(f"Successfully solved {len([r for r in responses if r is not None])} out of {len(responses)} test cases")
    else:
        print("No valid solutions found")

if __name__ == "__main__":
    asyncio.run(main())
