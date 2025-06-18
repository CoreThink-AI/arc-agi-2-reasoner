PROMPT = """
**Coordinate System:**

* Top-left cell is (0, 0). x increases down (rows), y increases right (columns).

**Given:**

* Input Grid: 
`{}`
* Output Grid: 
`{}`
* Input Objects: `{}`
* Output Objects: `{}`
* Pattern Specifications: `{}`

**Task:** For *each* pattern in the given list:

1. Compare Input vs Output objects to identify moves, removals, additions, rotations, shifts, duplications, or color changes.
2. Do note that some objects might combine to form a multi color bigger object.
3. Decide if the pattern applies.
4. Provide a concise **reason** for your decision **even if** `pattern_detected` is `false`. Include the precise reasons for object movements, additions, removals, retention. There exist some logic, your task is to find it using the help of patterns and params. 
5. List only the matched parameter values under `params` (use an empty object if none).

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