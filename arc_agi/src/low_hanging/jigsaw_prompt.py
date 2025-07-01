blank_prompt = """
You are given a square 2D grid represented as a Python list of lists of integers, where each integer is a colour index. The grid should be perfectly symmetric (vertical, horizontal, rotational, or any combination), but one or more than one contiguous patch of same-coloured cells currently breaks that symmetry.

Your task:
1. Identify which colour patch is wrong.
2. Provide a detailed, step-by-step reasoning of how you located the patch and determined its correct colour.
3. Finally, on a line by itself, output *only* the integer colour of the wrong patch, enclosed in triple backticks.

Input Grid
{}
"""

prompt = """You are given:

- **Sample Input–Output Pairs**
  One or more example grids showing how inputs map to outputs.
- **Test Input**
  A new input grid.
- **Blank-Space Color**
  The color code used to mark cells you must fill.
- **Complete Test Output Grid**
  The fully solved grid for the Test Input—with all blanks already filled.

Your Task:
1. Analyze the Sample Input–Output Pairs to learn the transformation pattern(s).
2. Locate **which region** (either the entire grid or a specific sub-grid) of the Complete Test Output Grid corresponds to the Test Input’s solution.
3. **Extract** that region, preserving the filled-in colors.

Prompt Template:
**Sample Examples:**
{}

**Test Input:**
{}

**Blank-Space Color:**
{}

**Complete Test Output Grid:**
{}

## Output Instructions

1. Show your **full, step-by-step reasoning** for how you identified the correct region.
2. Embed the Final output in ```  ``` and use \n for new row and | as column separator.

**Test Output:**

"""