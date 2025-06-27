HINT_PROMPT_TEMPLATE = """
You are an expert in visual reasoning for grid-based puzzles. A specific pattern transformation has been detected between an input and output grid.

📌 Pattern Name: {pattern_name}

📘 Description:
{description}

📊 Explanation (why this pattern was detected):
{reason}

🧱 Input Grid:
{input_grid_viz}

🧱 Output Grid:
{output_grid_viz}

🛠️ Your task:
Using the description and the specific grid example above, write a clear and detailed instructional hint that explains this pattern *in the context of this specific transformation*.

Your hint must include:
1. **What visibly changed in this example** — Describe the actual grid-level transformation (e.g., "new cells appeared", "objects extended", "parts were erased").
2. **How the change happened** — Was it row-wise, column-wise, around a shape, diagonally, etc.
3. **What clues indicate the pattern** — Are there alignments, symmetry, color repetition, movement, etc.
4. **How to recognize this pattern in future grids** — Give practical rules of thumb or visual cues.
5. *(Optional)* Mention relevant parameter categories from the pattern if helpful (e.g., "fill color was based on object", "stop condition was a boundary").

🎓 Guidelines:
- Focus on helping someone *understand and detect* this pattern in similar puzzles.
- Be grounded in the actual grid content shown above.
- Don’t repeat the definition — build on it with concrete insights.

Output only the instructional hint, in plain language.
"""
