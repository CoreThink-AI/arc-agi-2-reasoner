PROMPT = """
**Coordinate System:**

* Top-left cell is (0, 0). x increases down (rows), y increases right (columns).

**Given:**

* Input Grid: `{}`
* Output Grid: `{}`
* Added Objects: `{}`
* Removed Objects: `{}`
* Retained Objects: `{}`
* Pattern Specifications: `{}`

**Task:** For *each* pattern in the given list:

1. Compare Added, Removed and Retained objects to identify moves, removals, additions, rotations, shifts, duplications, or color changes.
2. Decide if the pattern applies.
3. Provide a concise **reason** for your decision **even if** `pattern_detected` is `false`.
4. List only the matched parameter values under `params` (use an empty object if none).

**Output:** Return *only* this JSON array (no extra text):

```json
[
  {{
    "reason": "<detailed explanation>",
    "pattern_detected": <true|false>,
    "pattern_name": "<Pattern Name>",
    "pattern_description": "<Pattern Specification.description>",
    "params": {{ /* matched values or {{}} */ }}
  }},
  ...
]
```
"""