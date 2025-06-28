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

🎁 Output only the structured hint(s). Do not wrap it in explanations or additional commentary. Make sure it is very very detailed.
"""

HINT_SUMMARY_PROMPT = """
You are an expert in visual reasoning for grid-based puzzles. You have just generated multiple structured hints for how a particular pattern transformation applies in a given Input→Output example. Now your task is to **summarize** those hints into a single, **concise** structured hint that retains all the key sections and essential information, eliminating redundancy.

📥 Inputs:
{}

🧠 Your Task:
Combine and condense the above hints into one structured hint. Preserve the following sections exactly as labeled, but merge overlapping points and shorten wording where possible:

---

### 🧩 Task
(One clear description of the transformation being applied)

### 🔢 Input
(Brief summary of only the most critical input features)

### 🎯 Objective
(What the pattern accomplishes in the output)

### 🪜 Step-by-Step
(A streamlined sequence of the steps, in order)

### 🚧 Constraints
(Only the core parameters that govern the transformation)

### 📤 Output
(What the final grid looks like and what changed)

---

📌 Guidelines:
- Do **not** add new technical details—only distill what’s already there.
- Use instructional language.
- Output **only** the unified structured hint, with no extra commentary.

 Make sure it is very very detailed.
"""
