example_template = """Example {i}:
Input:
{input_viz}

Output:
{output_viz}
"""

solver_prompt_template = """Consider the following examples:

{examples}

Based on the pattern in the examples, what would the output for the following test input be?

Test input:
{test_input_viz}

Test output:

(Hint: {hint}) 
"""

new_solver_prompt = """
You are given a set of **Input–Output Examples** and a detailed list of **Transformation Steps** (the *Hint*).

## Your Task

1. **Examine** the examples to understand exactly how the transformations are applied.
2. **Follow** the hint steps **in order**, **exactly as described**, with **no additions or omissions**.
3. **Use** the examples to resolve any remaining ambiguities in the hint.
4. **Apply** the same sequence of transformations to the **Test Input**.

> **Note:** The Test question is slightly more challenging than the training examples—it may involve **rotation invariance** and **color invariance**.

---

## Inputs Provided

* **Examples:** `{examples}`
* **Hint (Transformation Steps):** `{hint}`
* **Test Input Visualization:** `{test_input_viz}`
---

## Output Instructions

1. Present your **full reasoning**, detailing how each step from the hint maps to the transformation operations you perform.
2. Do **not** invent any new rules or skip any hint steps. Ensure the Test Output reflects the same behavior demonstrated by the examples.
3. Embed the Final output in triple backticks ``` ``` and use \n for new row and | as column separator.
- Use the character `\n` (a literal backslash followed by n) to separate rows.  
- Use the pipe character `|` to separate columns within each row.  
- Represent each cell with an integer digit only (0, 1, 2, 3, 4, 5, 6, 7, 8, 9). 
- Do NOT include spaces, dots, letters, or any other characters inside the matrix output.  
- This exact formatting will ensure your output can be parsed without errors.
- Here is an example of the Final output which is a 4x4 matrix within triple backticks:
```0|1|2|3\n4|5|6|7\n8|9|1|2\n3|4|5|6```
4. I want you to solve this puzzle **step by step**. First, restate the problem. Then outline your plan. Then execute each step, numbering them, and finally give your answer.
**Test Output:**

"""

critic_prompt = """
You are now acting as both an **Executor** and a **Critic**.  

---

## Original Task for the Executor
`{}`

##Below is the **Assistant’s previous attempt** at solving the task (the “Executor” output).  
`{}`

**Critic’s Tasks**  
1. **Evaluate** the above output against the Original Task and Hint:
   - Did it follow **every** hint step in the correct order?
   - Did it resolve all ambiguities using the examples?
   - Does the final Test Output match exactly what the transformations dictate?
2. **List** any mistakes or omissions (e.g., skipped steps, extra rules invented, misapplied rotation/color invariance).
3. **Produce** an **improved Test Output** that **correctly** applies all the hint steps, embedding it in triple back-ticks with `\n` and `|` separators.
4. Do not **overdo** a pattern. Understand the exact places where changes are necessary.

**Critic Output** should be structured as:
1. **Error Analysis:** (bullet list of issues)  
2. **Corrected Test Output:**  
    ```
   … your improved grid here …
    ```
**Embedd the output in ``` ``` and use \n for new row and | as column separator.

**Begin your Critic review now.**

"""

new_critic_template = """
You are now acting as both an **Executor** and a **Critic**.  

---
Use this template when you want the LLM to both re-execute a transformation task and critique its own output. It clearly separates inputs for the Executor and previous output, then defines the Critic’s evaluation.

---

## Original Task for the Executor

```
{ORIGINAL_TASK}
```

## Assistant’s Previous Attempt (Executor Output)

```
{PREVIOUS_ATTEMPT}
```
---

## Critic’s Instructions

1. **Evaluate** the previous Executor output against the Original Task:

   * Did it follow **every** hint step in the correct order?
   * Did it resolve ambiguities using only the provided examples?
   * Does the final Test Output match exactly what the transformations dictate?
2. **Identify** mistakes or omissions:

   * Skipped or re-ordered steps
   * Applied extra or missing colorings or moves
   * Misinterpreted object shapes, positions, or rules
   * Invented rules not supported by the hints or examples
3. **Produce** an **Improved Test Output** that correctly applies all hint steps, formatted exactly as:

   ```
   <row1>\n
   <row2>\n
   …
   ```

   * Use triple backticks ``` for the block
   * Separate rows with \n
   * Separate columns with |
   * No extra commentary or whitespace

---

**Output Format (Critic):**

1. **Error Analysis:**

   * Bullet-list each issue
2. **Corrected Test Output:**

   ```
   …improved grid…  
   ```

**Notes:**

* Do not generalize beyond the examples.
* Only apply color or transformation to the exact cells/components illustrated.
* Maintain precise formatting.

"""