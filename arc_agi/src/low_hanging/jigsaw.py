import math, time
from collections import Counter
from typing import Any, List
import re, json
from arc_agi.src.low_hanging.jigsaw_prompt import blank_prompt, prompt
from arc_agi.src.utils.llm_utils import get_anthropic_response_stream
from arc_agi.src.utils.visualization_utils import get_arr_viz
from arc_agi.src.solver.solver import get_formatted_examples,extract_matrix_from_response, matrix_to_arr

def find_symmetry(grid, blank_val=None):

    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    def score_axis(is_horizontal):
        length = rows if is_horizontal else cols
        # consider every possible axis position
        half = int(length/2)
        mids = [i for i in range(half-5,half+5)] + [i + 0.5 for i in range(half - 5, half + 5)]
        best = {'score': -1, 'mid': None}

        for mid in mids:
            matches = total = 0
            if is_horizontal:
                max_i = math.floor(mid)
                for i in range(max_i + 1):
                    mi = int(round(2 * mid - i))
                    if mi <= i or mi >= rows:
                        continue
                    for j in range(cols):
                        a, b = grid[i][j], grid[mi][j]
                        total += 1
                        if blank_val is not None and a == blank_val or b == blank_val:
                            continue
                        if a == b:
                            matches += 1
            else:
                max_j = math.floor(mid)
                for j in range(max_j + 1):
                    mj = int(round(2 * mid - j))
                    if mj <= j or mj >= cols:
                        continue
                    for i in range(rows):
                        a, b = grid[i][j], grid[i][mj]
                        total += 1
                        if blank_val is not None and a == blank_val or b == blank_val:
                            continue
                        if a == b:
                            matches += 1
            if total:
                score = (matches / total) * 100
                if score > best['score']:
                    best['score'], best['mid'] = score, mid
        return best['score'], best['mid']

    score_h, mid_h = score_axis(is_horizontal=True)
    score_v, mid_v = score_axis(is_horizontal=False)
    b,f = compute_diagonal_symmetry_scores(grid,blank_val)
    print("horizontal ", score_h)
    print("vertical", score_v)
    return mid_h, mid_v,max(score_h,score_v)

def fill_blanks(grid, axis, mid, blank_val=None):
    rows, cols = len(grid), len(grid[0]) if grid else 0
    new_grid = [row.copy() for row in grid]

    def mirror(i, j):
        # precise mirror index via exact arithmetic: no rounding
        if axis == 'horizontal':
            mi = int(2 * mid - i)
            return mi, j
        else:
            mj = int(2 * mid - j)
            return i, mj

    # Fill loop: repeat until stable
    while True:
        changed = False
        for i in range(rows):
            for j in range(cols):
                a = new_grid[i][j]
                mi, mj = mirror(i, j)
                if not (0 <= mi < rows and 0 <= mj < cols):
                    continue
                b = new_grid[mi][mj]
                # if one side blank and other not, copy value
                if blank_val is not None:
                    if a == blank_val and b != blank_val:
                        new_grid[i][j] = b
                        changed = True
                    elif b == blank_val and a != blank_val:
                        new_grid[mi][mj] = a
                        changed = True
        if not changed:
            break
    return new_grid

def fill_by_backwardslash_symmetry(grid, blank_val):
    n = len(grid)
    for i in range(n):
        for j in range(n):
            if grid[i][j] == blank_val and grid[j][i] != blank_val:
                grid[i][j] = grid[j][i]
    return grid

def fill_by_forwardslash_symmetry(grid, blank_val):
    n = len(grid)
    for i in range(n):
        for j in range(n):
            sym_i, sym_j = n - 1 - j, n - 1 - i
            if grid[i][j] == blank_val and grid[sym_i][sym_j] != blank_val:
                grid[i][j] = grid[sym_i][sym_j]
    return grid

def compute_diagonal_symmetry_scores(grid, blank_val=None):
    """
    Returns (backslash_score, forward_slash_score), each as a percentage:
      backslash_score  = symmetry across the main diagonal (i,j) ↔ (j,i)
      forward_slash_score = symmetry across the anti‐diagonal (i,j) ↔ (n-1-j, m-1-i)
    
    Works with non‐square grids by only pairing cells whose “mirror” actually exists.
    """
    n = len(grid)
    if n == 0:
        return 0.0, 0.0
    m = len(grid[0])
    # for main‐diagonal, only indices 0 ≤ i<j < min(n,m) pair up
    min_nm = min(n, m)

    total_back = match_back = 0
    for i in range(min_nm):
        for j in range(i + 1, min_nm):
            a = grid[i][j]
            b = grid[j][i]
            if a != blank_val and b != blank_val:
                total_back += 1
                if a == b:
                    match_back += 1

    # for anti‐diagonal, map (i,j) → (i2,j2) = (n-1-j, m-1-i)
    total_fwd = match_fwd = 0
    for i in range(n):
        for j in range(m):
            i2 = n - 1 - j
            j2 = m - 1 - i
            # check the target is in‐bounds
            if 0 <= i2 < n and 0 <= j2 < m:
                # only count each pair once: (i,j) < (i2,j2) lexicographically
                if (i, j) < (i2, j2):
                    a = grid[i][j]
                    b = grid[i2][j2]
                    if a != blank_val and b != blank_val:
                        total_fwd += 1
                        if a == b:
                            match_fwd += 1

    score_back = (match_back / total_back * 100) if total_back else 0.0
    score_fwd  = (match_fwd  / total_fwd  * 100) if total_fwd  else 0.0

    return score_back, score_fwd


def infer_blank_color(data) -> Any:

  response = get_anthropic_response_stream(blank_prompt.format(get_formatted_examples(data["train"])))
  match = re.search(r"```[\s]*([0-9]+)[\s]*```",response)
  if match:
    num = int(match.group(1))
    return num
  else:
    print("Match not found")
    return 0
  
def solve(grid,blank_val,data):
  blank_val = infer_blank_color(data)
  axis_h,axis_v,_  = find_symmetry(grid, blank_val)
  filled = fill_blanks(grid, "horizontal",axis_h,blank_val)
  filled = fill_blanks(filled,"vertical",axis_v,blank_val)
  filled = fill_by_backwardslash_symmetry(filled,blank_val)
  #filled = fill_by_forwardslash_symmetry(filled,blank_val)
  return filled,blank_val

def find_background_color_simple(grid):
    """Simple rule-based background color detection without LLM"""
    # Flatten the grid and count color frequencies
    flat_grid = [cell for row in grid for cell in row]
    color_counts = Counter(flat_grid)
    
    # Get edge colors (border cells)
    edge_colors = []
    height, width = len(grid), len(grid[0])
    
    # Top and bottom edges
    for j in range(width):
        edge_colors.append(grid[0][j])  # Top edge
        edge_colors.append(grid[height-1][j])  # Bottom edge
    
    # Left and right edges (excluding corners to avoid double counting)
    for i in range(1, height-1):
        edge_colors.append(grid[i][0])  # Left edge
        edge_colors.append(grid[i][width-1])  # Right edge
    
    edge_color_counts = Counter(edge_colors)
    
    # Find most common color that also appears on edges
    most_common_color = color_counts.most_common(1)[0][0]
    most_common_edge_color = edge_color_counts.most_common(1)[0][0] if edge_color_counts else most_common_color
    
    # If the most common color is also the most common edge color, it's likely the background
    if most_common_color == most_common_edge_color:
        return most_common_color
    
    return most_common_color

from typing import List, Tuple

Grid = List[List[int]]

def extract_corresponding_patch(
    input_grid: Grid,
    output_grid: Grid,
    target_color: int
) -> Grid:
    """
    Finds the bounding box of all cells == target_color in input_grid,
    and returns the subgrid of output_grid covering exactly that same box.
    """
    rows = len(input_grid)
    cols = len(input_grid[0]) if rows else 0

    # Collect coordinates of the patch in input
    patch_coords: List[Tuple[int,int]] = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if input_grid[r][c] == target_color
    ]
    if not patch_coords:
        raise ValueError(f"No cells of color {target_color} found in input grid.")

    # Determine bounding box
    rs, cs = zip(*patch_coords)
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)

    # Slice the output grid
    extracted: Grid = [
        output_grid[r][min_c : max_c + 1]
        for r in range(min_r, max_r + 1)
    ]
    return extracted

def extract_jigsaw_output(file_path,i,full_grid,blank_val):

  with open(file_path, 'r') as f:
    data = json.load(f)
  flag = 0
  input_grid = data["test"][i]["input"]
  for entry in data["train"]:
    if len(entry["input"]) != len(entry["output"]) and len(entry["input"][0]) != len(entry["output"][0]):
      flag=1
  if flag==0:
    return full_grid
  else:
    return extract_corresponding_patch(input_grid,full_grid,blank_val)
  
def check_jigsaw(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    tests = data["test"]
    for test in tests:
        test_input = test["input"]
        blank_val = find_background_color_simple(test_input)
        _, _, score = find_symmetry(test_input,blank_val)
        if score>70:
            return True
        else:
            return False
        
def do_jigsaw(file_path):
    t1 = time.time()
    with open(file_path, 'r') as f:
        data = json.load(f)
    tests = data["test"]
    arrs, gts = [],[]
    for i,test in enumerate(tests):
        print(i)
        test_input = test["input"].copy()
        gts.append(test["output"])
        test_output,blank_val = solve(test_input,8,data)
        arr_response = extract_jigsaw_output(file_path,i,test_output,blank_val)
        arrs.append(arr_response)
    
    return arrs, gts, time.time()-t1