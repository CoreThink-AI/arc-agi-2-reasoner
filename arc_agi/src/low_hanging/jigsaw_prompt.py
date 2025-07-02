blank_prompt = """
You are given sample train examples which consist of Input and Output Grids. The transformation occuring between Input to Output is completing the symmetry of some patches in the input.

Your task:
1. Compare Input and Output Grid and let me know which color patch needs to be completed from the input grid.
2. Provide a detailed, step-by-step reasoning of how you located the patch and determined its correct colour.
3. Finally, on a line by itself, output *only* the integer colour of the wrong patch, enclosed in triple backticks.

Training Examples
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