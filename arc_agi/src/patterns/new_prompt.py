PROMPT = """
**Coordinate System:**  
Treat the grid like a matrix:  
- The top-left cell is (0, 0).  
- x increases downwards (rows), y increases right (columns).  
- The bottom-right cell is (height − 1, width − 1).
---
You are given two grids (an Input image and an Output image) and the full set of identified objects in each. Every object is represented as a JSON‐style dictionary with these exact fields:

  • colors: a list of all pixel-colors in the object  
  • color: a single color if the object is monochrome; otherwise null  
  • size: [height, width] in grid‐cells  
  • grid_size: the overall grid dimensions (height*width) of the image  
  • type: the class name of the object (e.g. “Rectangle”, “Circle”, etc.)  
  • coordinates: a list of [x,y] coordinates for every pixel in the object  
  • shape: the high‐level shape_type (e.g. “filled_rectangle”, “hollow_shape”, etc.)  
  • x1, x2, y1, y2: the bounding‐box corners in grid coordinates  
  • placement: the [x,y] position of the object’s center point  
  • centroid: the [x,y] coordinate of its geometric center  
  • cavities: a list of lists, where each inner list describes one hole (as [x,y] tuples) inside the object  
  • bottom_left: the [x,y] coordinate of its bottom‐left corner  
  • top_right: the [x,y] coordinate of its top‐right corner  
----
You are an expert pattern analyst for puzzles. Your task is to determine if a given transformation patterns exists between an input grid and an output grid.

## INPUT DATA:
**Input Grid:** 
{}
**Output Grid:** 
{}
**Input Objects:** {}
**Output Objects:** {}
**List of Pattern with Specification:** {}

###Concentrate on:
1. **Removal Criteria**  
   - Which objects get removed, and what relationships they had in the Input image?  
     - Are removed objects adjacent to (or overlapping with) a retained object?  
     - Do they share a color, shape, or size with another object that was retained?  
     - Are they part of a cluster or alignment that disappears?  
     - Were any removed objects originally in a rotated or shifted arrangement relative to a retained set?

2. **Addition Criteria**  
   - Which objects are added, and how they relate to retained or removed objects? Find the Reason for Object addition at that very specific place.   
     - Do added objects appear at the midpoint or intersection of two retained objects?  
     - Are they introduced only if two removed objects satisfied some spatial relation (e.g., collinear)?  
     - Do they “replace” a removed object in the same location or adopt its color/shape?  
     - Are added objects rotated versions of removed objects (e.g., a horizontal line becomes a vertical line)?  
     - Are added objects simply shifted copies of retained objects (e.g., same shape/color but moved one cell right)?

3. **Relationships Among Removed/Added**  
   - Sometimes removed objects themselves form patterns (e.g., all removed objects lie on a diagonal), or new objects appear in the same geometric arrangement. Look for:  
     - Pairwise or group relationships among removed objects (e.g., they formed a square that vanished).  
     - Pairwise or group relationships among added objects (e.g., two new circles appear symmetrically around a retained rectangle).  
     - Rotational symmetry: were any removed objects rotated 90° or 180° and then re-inserted as added objects?  
     - Translational symmetry: were added objects simply the same shape/color but shifted by a constant vector?

4. **Retained Objects as Anchors**  
   - Since retained objects have no property changes (except possibly a rotation or pure shift), treat them as fixed reference points. Ask:  
     - Does the status (removed/added) of an object depend on its distance from a retained object?  
     - Does an added object appear only if it completes a row/column with retained ones?  
     - Were any retained objects themselves rotated or shifted between Input and Output? (If so, note that as part of the transformation.)

5. **Color/Shape Constraints**  
   - Check if color or shape similarity between objects triggers removal/addition:  
     - Remove all objects of the same color as a retained object if they lie outside a certain radius.  
     - Add new objects whose color matches the two nearest retained ones.  
     - If a removed object is exactly the same shape as a retained one but rotated, is it considered “removed + added” or “retained with rotation”?

6. **Collective Behavior**  
   - If a group (cluster) of objects is removed, check how that group was defined—perhaps by adjacency, by bounding-box overlap, or by sharing a color. Similarly, if multiple new objects appear in a symmetric arrangement, note that.  
   - Look for combined rotations or shifts of entire sub-clusters: e.g., “All objects that formed a 2×2 block were shifted right by 2 cells and then rotated 90° collectively.”
---
**Steps:**
1. Read the entire INPUT_GRID to build a mental model of all objects, their shapes, colors, and positions.
2. Read the entire OUTPUT_GRID to see how those objects have changed.
3. Compare object-by-object:
   - Which objects moved, rotated, resized, duplicated, or changed color?
   - Which new objects appeared or old objects disappeared?
   - How did each object’s coordinates (x, y), dimensions, and properties differ between input and output?
4. Check if these observed changes exactly match the Pattern Specification’s textual description.
5. Give a very detailed reason for each instance, and if you are very, very confident then only say pattern_detected is true.
6. For each parameter in Pattern_Specification.params, pick **only** those listed values that truly occur in this transformation.
7. Return List of **only** this final JSON (no extra text):
[
```json
{{
  "reason":"<why this pattern, found or not found>"
  "pattern_detected": <true|false>,
  "pattern_name": "<Pattern Name>",
  "pattern_description": "<Pattern Specification.description>",
  "params": {{
    "<param1>": [<matched values>],
    "<param2>": [<matched values>],
    …
  }}
}}
```
,..]
8. It is possible to have multiple matching patterns. The output list should include all the observed patterns in a comma separated format.
– If no match: "pattern_detected": false and "params": {{}}.
– Do not add any fields beyond those five.
"""