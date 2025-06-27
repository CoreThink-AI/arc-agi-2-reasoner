HINT_PROMPT_TEMPLATE = """
You are an expert in visual reasoning for grid-based puzzles. A specific pattern transformation has been detected between an input and output grid.

📌 Pattern Name: {pattern_name}

📘 Description:
{description}

📊 Explanation (why this pattern was detected):
{reason}

🧰 Parameters associated with this pattern:
(Listed as inferred or used in this example)
{params}

🧱 Input Grid:
{input_grid_viz}

🧱 Output Grid:
{output_grid_viz}

🧠 Your task:
Write one or more **structured hints** that explain how this pattern applies in this specific transformation. Each hint should help someone understand *how the pattern works*, *what steps are involved*, and *how to spot or apply it again*.

Please format your hints using the following sections:

---

### 🧩 Task
Describe what kind of transformation is being applied — e.g., "Fill horizontally from a source object", "Remove the leftmost item", etc.

### 🔢 Input
Summarize key elements or configurations visible in the input grid that are relevant to this pattern (e.g., object positions, gaps, colors).

### 🎯 Objective
State clearly what the pattern is trying to achieve in the output (e.g., "connect two squares", "mirror a shape", "erase matching tiles").

### 🪜 Step-by-Step
Break down the specific steps involved in the transformation in this example, including the sequence, direction, and logic.

### 🚧 Constraints
Mention any conditions or parameters that govern how the transformation happens. Use the pattern’s parameters where relevant. For example:
- Fill color = based on source
- Stop when = boundary is hit
- Direction = ↘ diagonal
You may refer directly to the provided parameters.

### 📤 Output
Explain the final effect or visual result on the output grid — what new elements are present, which ones were modified or removed, and what visual pattern is now seen.

---

📌 Guidelines:
- You may write multiple hints if useful, especially for complex or multi-part transformations.
- Use simple, instructional language aimed at helping someone solve similar puzzles.
- Do **not** simply repeat the pattern description — explain how it *manifested* in the example shown.
- Reference the actual content and layout of the input/output grid in your explanation.
- Keep each section short but informative.

🎁 Output only the structured hint(s). Do not wrap it in explanations or additional commentary.
"""

