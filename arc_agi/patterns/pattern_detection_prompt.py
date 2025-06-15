PROMPT = """
**Coordinate System:**  
Treat the grid like a matrix:  
- The top-left cell is (0, 0).  
- x increases downwards (rows), y increases right (columns).  
- The bottom-right cell is (height − 1, width − 1).
---

You are an expert pattern analyst for ARC-AGI (Abstraction and Reasoning Corpus) puzzles. Your task is to determine if a specific transformation pattern exists between an input grid and an output grid.

## INPUT DATA:
**Input Grid:** {}
**Output Grid:** {}
**Input Objects:** {}
**Output Objects:** {}
**Pattern Name** {}
**Pattern Specification:** {}

**Task**  
1. Analyze the difference between INPUT_DICT and OUTPUT_DICT.  
2. Determine if the transformation matches PATTERN_SPEC.description.  
3. If it matches, for each param in PATTERN_SPEC.params, select all values that apply.  
4. Return exactly one JSON object with:
```json
{{
  "pattern_detected": <true|false>,
  "matched_params": {{
    "<param1>": [<matched values>],
    "<param2>": [<matched values>],
    …
  }}
}}
```
– If no match: "pattern_detected": false and "matched_params": {{}}.
– Output only valid JSON, no commentary or extra fields.
"""
