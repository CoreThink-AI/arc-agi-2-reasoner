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
Explain the final effect or visual result on the output grid — what new elements are present, which ones were modified or removed, and what visual pattern is now seen. Give output in markdown syntax.

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

Example Output:
### Task: Gravity-Driven Cavity Filling with Blue Cells\n\n#### Color Index Reference\n- **Color 1 (Blue)** — Mobile filler cells that will “fall” into cavities under a gravity effect.\n- **Color 0 (Background/Voids)** — Empty space and cavities to be filled.\n- **Color 2+ (Other Colors)** — Immovable obstacles; define the boundaries of cavities.\n\n---\n\n#### Input  \nYou are given a 2D grid of size *H×W* containing:\n- **Blue cells (1):** A fixed number of filler cells that can move vertically under gravity.\n- **Void cells (0):** Empty spaces that represent cavities.\n- **Obstacle cells (≥2):** Walls or fixed regions that define cavity boundaries.\n\n---\n\n#### Objective  \n1. **Detect all vertical cavities** that are bounded on both left and right by obstacle cells (i.e., each row segment of zeros whose immediate neighbors on left and right are non-zero).\n2. **Simulate gravity** by letting Blue cells “fall” straight down into these cavities from above, filling from the bottom up.\n3. **Preserve the total count** of Blue cells; no Blue cell is created or destroyed.\n4. **Produce an updated grid** where voids within bounded cavities are filled as far as possible by Blue cells under gravity.\n\n---\n\n#### Step-by-Step Instructions\n\n1. **Initialize**  \n   - Read the input grid `G[H][W]`.  \n   - Prepare an output grid `H_grid ← G` for the final state.\n\n2. **Locate Bounded Cavities**  \n   - For each row *r* and each column segment `c_start…c_end` where `G[r][c] == 0` for all `c_start ≤ c ≤ c_end`, check that:  \n     - `G[r][c_start – 1] ≥ 2` (left obstacle) and  \n     - `G[r][c_end + 1] ≥ 2` (right obstacle).  \n   - Record all such row‐segments as “cavity cells.”\n\n3. **Count Blue Cells Above Cavities**  \n   - For each cavity cell `(r, c)` in a bounded segment, look upward in column *c* from row `0` to `r–1` and count all Blue cells (`1`) that are not already assigned to another cavity fill.  \n   - Aggregate these counts per cavity segment.\n\n4. **Simulate Gravity Filling**  \n   - For each cavity segment in bottom‐up order (largest *r* first):  \n     a. Let *k* = number of available Blue cells above that segment.  \n     b. For rows `r` down to `r – k + 1`, set `H_grid[row][c] = 1` to drop Blue cells into the lowest empty spots.  \n     c. Mark those *k* Blue cells in the source columns as “used” (so they won’t fall again).  \n     d. Leave any remaining voids (`0`) if Blue cells are exhausted.\n\n5. **Preserve Remaining Grid**  \n   - All non‐cavity zeros that aren’t bounded or that lie outside the simulated falls remain `0`.  \n   - Obstacle cells (≥2) remain unchanged.  \n   - Any Blue cells not used to fill cavities stay in their original positions in `H_grid`.\n\n6. **Finalize Output**  \n   - Return `H_grid`, now with gravity‐filled bounded cavities and the same total count of Blue cells as the input.\n\n---\n\n#### Constraints\n- Cavities must be strictly horizontally bounded by obstacle cells on both sides in the same row.\n- Gravity acts only downward; Blue cells do not move horizontally or upward.\n- Total number of Blue cells in the output must equal the input count.\n- Obstacle cells (colors ≥2) are fixed and impermeable.\n\n---\n\n#### Output  \nA 2D grid of size *H×W* in which all possible bounded cavities have been filled from the bottom up by Blue cells under gravity, with no change in total Blue‐cell count and all obstacle positions preserved."
 
Make sure it is very very detailed. Give output in markdown syntax.
"""
