# KiVA Benchmark — Worked Solutions
**Prepared by:** Aryan Chindalur, Corethink AI  
**Dataset:** KiVA-easy | **Trials used:** Trial 0 per concept  
**Format:** Neurosymbolic worked solutions modelled on ARC-AGI-2 style

---

> **How to read each solution**  
> Each solution shows the **training demonstration** (input → output pair the model learns from), then the **test question** (new object + two options), followed by a detailed transformation analysis and formal IF-THEN rules.  
> `mc_0` is always the **correct** answer. `mc_1` is the **distractor**.

---

---

# Solution 1 — Colour / Red (Trial 0)

**Task ID:** ColourRed_0  
**Concept:** Colour  
**Parameter:** Red  
**Difficulty:** KiVA-easy

---

## Training Demonstration

| | Image | Description |
|---|---|---|
| **Train Input** | `ColourRed_0_train_0_input.png` | A sitting dog rendered entirely in **blue/purple** — same pose, white background |
| **Train Output** | `ColourRed_0_train_0_output.png` | The **identical** dog in the **identical** pose, now rendered entirely in **red** |

**What changed:** The colour of every pixel belonging to the object shifted from blue → red.  
**What stayed the same:** Shape, pose, size, position, background (white), object identity (dog).

---

## Test Question

| | Image | Description |
|---|---|---|
| **Test Input** | `ColourRed_0_test_0_input.png` | A halved citrus fruit (lemon/lime cross-section) rendered in **blue/purple** |
| **Option A — mc_0 ✅** | `ColourRed_0_test_mc_0_input.png` | The same citrus slice now rendered in **red/orange** |
| **Option B — mc_1 ❌** | `ColourRed_0_test_mc_1_input.png` | The same citrus slice rendered in **green** |

---

## #Transformation Pattern

The transformation recolours every pixel of the object from its current colour to the target colour **Red**, leaving everything else — shape, size, orientation, background — completely unchanged. This is a pure colour substitution applied uniformly to the entire object.

### 1. What stays the same
- **Object identity** — the type of object (dog, citrus slice, etc.) does not change between training and test.
- **Shape and silhouette** — the precise outline and internal structure of the object is pixel-for-pixel identical.
- **Size and position** — the object occupies the same region of the frame before and after transformation.
- **Background** — always white; background pixels are never recoloured.
- **Pose and orientation** — the object faces the same direction, holds the same position.

### 2. What changes
- **Pixel colour** — every non-background pixel of the object is uniformly remapped to **Red**. In the training example, blue (#6B6BDB range) → red (#E84040 range). The transformation is applied globally to all object pixels, not selectively.

### 3. Why the object changes between training and test
- KiVA tests **cross-domain generalisation** — the training demonstration uses one object class (dog), and the test uses a completely different object class (citrus slice). The rule must be understood abstractly ("change colour to Red") and applied to a novel object.

### 4. Why Option A (mc_0) is correct
- Option A shows the citrus slice in red — the same colour the dog was transformed to in training. Shape, size, and internal detail (citrus segments visible) are preserved exactly. This is the correct application of the Colour→Red rule to the new object.

### 5. Why Option B (mc_1) is a distractor
- Option B shows the citrus slice in **green** — a valid colour within the KiVA Colour domain (Green is another parameter). The distractor is deliberately another KiVA colour parameter, testing whether the model correctly identifies Red (not just "a different colour"). A system relying purely on "change colour" without identifying the specific target colour would fail here.

### 6. Corner cases
- The internal texture detail of the object (fur patterns on dog, citrus segment lines) remains visible after recolouring — it is not erased. Only the hue shifts.
- All object pixels receive the same target colour uniformly — there is no gradient or partial recolouring.

**In short:** "Whatever colour the object currently is, recolour it entirely to Red. Keep everything else identical."

---

## #Formal Rules

```
Rule_Colour_Red: ApplyColourTransform

IF   Object(id=0, colour=ANY, shape=S, size=Z, pose=P)
AND  Transform = Colour
AND  Parameter = Red
THEN Object(id=0, colour=Red, shape=S, size=Z, pose=P)
AND  Background(unchanged)
AND  ObjectCount(unchanged)
```

**Distractor rule (what mc_1 incorrectly applies):**
```
Rule_Colour_Green_WRONG: ApplyColourTransform
IF   Parameter = Green   ← incorrect parameter
THEN Object(colour=Green) ← wrong target colour
```

---

---

# Solution 2 — Counting / +1 (Trial 0)

**Task ID:** Counting+1_0  
**Concept:** Counting  
**Parameter:** +1  
**Difficulty:** KiVA-easy

---

## Training Demonstration

| | Image | Description |
|---|---|---|
| **Train Input** | `Counting+1_0_train_0_input.png` | **2 balloon-like toys** (grey spheres each with a small loop/ring at the base), evenly spaced in a row across the top of the frame |
| **Train Output** | `Counting+1_0_train_0_output.png` | **3 balloon-like toys** — identical objects, same size and style, now evenly spaced across the top of the frame |

**What changed:** The count increased by exactly 1 (2 → 3).  
**What stayed the same:** Object type, size, colour, style of each individual object; background; spatial arrangement logic (evenly spaced row).

---

## Test Question

| | Image | Description |
|---|---|---|
| **Test Input** | `Counting+1_0_test_0_input.png` | **2 badminton rackets**, side by side, same size, white background |
| **Option A — mc_0 ✅** | `Counting+1_0_test_mc_0_input.png` | **3 badminton rackets**, evenly spaced, same size and style |
| **Option B — mc_1 ❌** | `Counting+1_0_test_mc_1_input.png` | **1 badminton racket**, alone in the frame |

---

## #Transformation Pattern

The transformation adds exactly **one more copy** of the existing object to the scene. The new copy is identical in appearance (same size, colour, orientation) to the originals. The spatial arrangement is maintained — objects remain evenly distributed across the frame.

### 1. What stays the same
- **Object identity** — the type of object stays the same within each task (all balloons, all rackets).
- **Individual object appearance** — each object's size, colour, and orientation is unchanged.
- **Arrangement logic** — objects are arranged in a row; adding one more extends the row evenly rather than stacking or overlapping.
- **Background** — always white.

### 2. What changes
- **Count** — the number of visible objects increases by exactly **+1**. Input has N objects → Output has N+1 objects.

### 3. Why the object changes between training and test
- Training uses balloon toys (N=2 → N=3). Test uses badminton rackets (N=2 → N=3). The concept is purely numerical: the model must abstract "add one more" regardless of what the object is.

### 4. Why Option A (mc_0) is correct
- Option A shows 3 rackets — exactly one more than the test input (2). The objects are identically sized, same orientation, evenly spaced. This correctly applies the +1 rule to the new object class.

### 5. Why Option B (mc_1) is a distractor
- Option B shows **1 racket** — this corresponds to the **−1** rule (subtracting one), another valid KiVA Counting parameter. This is the most natural confusion: both options differ from the input by one object, but in opposite directions. A system that detects "count changes" without determining the direction would fail here.

### 6. Corner cases
- The newly added object is always a **copy** of the existing objects — no new object type is introduced.
- Objects do not overlap after addition; layout expands horizontally to accommodate the extra item.
- The rule applies to the count at the object level, not pixel level.

**In short:** "Count the objects, add one identical copy, redistribute evenly."

---

## #Formal Rules

```
Rule_Counting_PlusOne: IncrementObjectCount

IF   Objects(count=N, type=T, size=Z, colour=C, orientation=O)
AND  Transform = Counting
AND  Parameter = +1
THEN Objects(count=N+1, type=T, size=Z, colour=C, orientation=O)
AND  Layout(evenly_spaced, same_axis_as_input)
AND  NewObject(appearance=copy_of_existing)
```

**Distractor rule (what mc_1 incorrectly applies):**
```
Rule_Counting_MinusOne_WRONG: DecrementObjectCount
IF   Parameter = -1    ← wrong direction
THEN Objects(count=N-1) ← removes rather than adds
```

---

---

# Solution 3 — Resize / 0.5XY (Trial 0)

**Task ID:** Resize0.5XY_0  
**Concept:** Resize  
**Parameter:** 0.5XY (shrink to 50% in both width and height)  
**Difficulty:** KiVA-easy

---

## Training Demonstration

| | Image | Description |
|---|---|---|
| **Train Input** | `Resize0.5XY_0_train_0_input.png` | A brown Labrador puppy, medium size, centred in the frame, facing right |
| **Train Output** | `Resize0.5XY_0_train_0_output.png` | The **identical** dog, same colour and pose, now visibly **smaller** — approximately half the width and half the height of the input, repositioned toward the lower-centre of the frame |

**What changed:** The object's dimensions reduced to 50% in both the X (width) and Y (height) axes. Resulting area = 25% of original.  
**What stayed the same:** Object identity, colour, pose, detail/texture, background.

---

## Test Question

| | Image | Description |
|---|---|---|
| **Test Input** | `Resize0.5XY_0_test_0_input.png` | A grey chess king piece, medium size, centred vertically in the frame |
| **Option A — mc_0 ✅** | `Resize0.5XY_0_test_mc_0_input.png` | The same chess king piece at **half the size** — noticeably smaller, same grey colour and details |
| **Option B — mc_1 ❌** | `Resize0.5XY_0_test_mc_1_input.png` | The same chess king piece at **larger than input size** — clearly bigger |

---

## #Transformation Pattern

The transformation uniformly scales the object down to **50% of its original dimensions in both axes simultaneously**. The object's aspect ratio is perfectly preserved (no stretching or squashing). The object remains centred in the frame at its new, smaller size.

### 1. What stays the same
- **Object identity** — the chess king remains a chess king; no new objects are introduced.
- **Colour and texture** — every surface detail, colour, and material appearance is preserved at the new scale.
- **Aspect ratio** — width-to-height ratio is unchanged; the object looks identical but smaller, not distorted.
- **Background** — always white.
- **Orientation** — the object faces the same direction.

### 2. What changes
- **Width** — reduced to 50% of input width.
- **Height** — reduced to 50% of input height.
- **Apparent area** — because both axes scale equally, the visible area becomes **25%** of the original (0.5 × 0.5 = 0.25).

### 3. Why the object changes between training and test
- Training demonstrates the rule on a dog; test applies it to a chess piece. The model must generalise "shrink to 50%" to an entirely different object category with different aspect ratio and texture.

### 4. Why Option A (mc_0) is correct
- Option A shows the chess king at clearly half the size of the test input — same grey colour, same crown/cross details, same proportions, just smaller. This is the correct application of the 0.5XY rule.

### 5. Why Option B (mc_1) is a distractor
- Option B shows the chess king **larger** than the test input — this corresponds to the **2× (scale up)** rule, another KiVA Resize parameter. The distractor exploits the fact that both options differ from the input in size; the model must identify the correct **direction** (shrink, not grow) and **magnitude** (50%, not 200%).

### 6. Corner cases
- The 0.5XY parameter applies scaling to both axes simultaneously and equally — it is distinct from 0.5X (shrink width only) or 0.5Y (shrink height only).
- Internal detail is preserved at the smaller scale — the crown of the chess piece is still visible in mc_0.
- The object is never clipped or cropped; the frame always contains the full object.

**In short:** "Keep the object identical in every way, but shrink both its width and height to exactly half."

---

## #Formal Rules

```
Rule_Resize_0.5XY: UniformScaleDown

IF   Object(id=0, width=W, height=H, colour=C, pose=P, texture=T)
AND  Transform = Resize
AND  Parameter = 0.5XY
THEN Object(id=0, width=0.5×W, height=0.5×H, colour=C, pose=P, texture=T)
AND  AspectRatio(preserved)
AND  Position(centred_in_frame)
AND  ObjectCount(unchanged = 1)
```

**Distractor rule (what mc_1 incorrectly applies):**
```
Rule_Resize_2XY_WRONG: UniformScaleUp
IF   Parameter = 2XY    ← scale-up instead of scale-down
THEN Object(width=2×W, height=2×H) ← grows instead of shrinks
```

---

---

# Solution 4 — Reflect / X (Trial 0)

**Task ID:** ReflectX_0  
**Concept:** Reflect  
**Parameter:** X (reflection over the horizontal/X-axis — a vertical flip)  
**Difficulty:** KiVA-easy

---

## Training Demonstration

| | Image | Description |
|---|---|---|
| **Train Input** | `ReflectX_0_train_0_input.png` | A whippet/greyhound dog sitting upright, facing left — head at top, paws at bottom, white background |
| **Train Output** | `ReflectX_0_train_0_output.png` | The **same dog**, now **upside-down** — paws pointing upward, head facing downward-left. The image is flipped top-to-bottom as if reflected in a horizontal mirror |

**What changed:** The object's vertical orientation is inverted — top becomes bottom, bottom becomes top.  
**What stayed the same:** Shape, colour, size, left-right orientation (the dog still faces left — no horizontal flip occurred), background.

---

## Test Question

| | Image | Description |
|---|---|---|
| **Test Input** | `ReflectX_0_test_0_input.png` | A tan/beige blimp (airship) in horizontal orientation — body pointing right, tail fin on the upper-left, gondola hanging below centre |
| **Option A — mc_0 ✅** | `ReflectX_0_test_mc_0_input.png` | The same blimp, now **flipped upside-down** — body still pointing right but the gondola is now on top, tail fin now on the lower-left |
| **Option B — mc_1 ❌** | `ReflectX_0_test_mc_1_input.png` | The same blimp in its **original orientation** (or a Y-axis mirror) — appears similar to the test input |

---

## #Transformation Pattern

The transformation reflects the object over its **horizontal (X) axis** — equivalently, it flips the image **vertically** (top ↔ bottom). The result looks as if you held the original image up to a mirror placed flat underneath it. Left-right relationships are perfectly preserved; only top-bottom relationships invert.

### 1. What stays the same
- **Object identity** — the blimp remains a blimp, the dog remains a dog.
- **Colour and texture** — all surface details, colours, and materials are preserved.
- **Size** — the object occupies the same amount of space in the frame.
- **Left-right orientation** — the blimp still points right after the flip; horizontal position relationships are unaffected.
- **Background** — always white.

### 2. What changes
- **Vertical orientation** — the top of the object becomes the bottom and vice versa. For the blimp: the gondola (originally below) is now above; the tail fin (originally at upper-left) is now at lower-left.
- **The transformation can be described as:** `pixel(x, y) → pixel(x, H−y)` where H is the image height.

### 3. Why the object changes between training and test
- The dog demonstrates the abstract rule "flip top-to-bottom." The blimp test requires applying this same operation to a rigid, asymmetric object with clearly distinguishable top and bottom features (gondola below vs. above), making the flip visually unambiguous.

### 4. Why Option A (mc_0) is correct
- mc_0 shows the blimp inverted vertically: what was the underside (gondola) is now visible on top, and the tail fin has moved to the lower region. This precisely matches a ReflectX transformation.

### 5. Why Option B (mc_1) is a distractor
- mc_1 shows the blimp in an orientation that appears the same as — or very close to — the test input. It is designed to test whether the model recognises that a true vertical flip produces a visually distinct result. A model that does not carefully track which features are on top vs. bottom may select the unchanged version as "correct."
- Note: mc_1 for ReflectX looks similar to mc_0 for 2DRotation (+90°) — deliberate cross-concept distractor design.

### 6. Corner cases
- ReflectX (vertical flip) must not be confused with **ReflectY** (horizontal/left-right flip). For an asymmetric object like the blimp, these produce different results: ReflectX keeps the blimp pointing right (flips gondola up), while ReflectY would make the blimp point left.
- For objects that have top-bottom symmetry (e.g., a perfect sphere), ReflectX would produce an image visually identical to the input — but KiVA chooses asymmetric objects to make the transformation visible.

**In short:** "Flip the object upside-down — swap top and bottom while keeping left and right exactly as they are."

---

## #Formal Rules

```
Rule_Reflect_X: VerticalFlip

IF   Object(id=0, pixel_map=P[x,y], width=W, height=H)
AND  Transform = Reflect
AND  Parameter = X
THEN Object(id=0, pixel_map=P[x, H−y])
AND  LeftRightOrientation(unchanged)
AND  TopBottomOrientation(inverted)
AND  Size(unchanged)
AND  Colour(unchanged)
```

**Distractor rule (what mc_1 shows — no transformation):**
```
Rule_NoTransform_WRONG:
IF   Parameter = None / Identity  ← no operation applied
THEN Object(pixel_map=P[x,y])     ← same as input, unchanged
```

---

---

# Solution 5 — 2DRotation / +90° (Trial 0)

**Task ID:** 2DRotation+90_0  
**Concept:** 2DRotation  
**Parameter:** +90° (90-degree clockwise rotation)  
**Difficulty:** KiVA-easy

---

## Training Demonstration

| | Image | Description |
|---|---|---|
| **Train Input** | `2DRotation+90_0_train_0_input.png` | A whippet/greyhound dog sitting upright — head at top-left, paws at bottom, facing left |
| **Train Output** | `2DRotation+90_0_train_0_output.png` | The **same dog**, now rotated **90° clockwise** — the dog is on its back with legs pointing to the upper-left and head now pointing to the lower-right. The "up" direction of the original image now faces right |

**What changed:** The entire object was rotated 90° clockwise about its centre.  
**What stayed the same:** Shape, colour, size, internal details, background.

---

## Test Question

| | Image | Description |
|---|---|---|
| **Test Input** | `2DRotation+90_0_test_0_input.png` | A tan/beige blimp (airship) in horizontal orientation — body pointing right, tail fin on upper-left, gondola below centre |
| **Option A — mc_0 ✅** | `2DRotation+90_0_test_mc_0_input.png` | The same blimp now **vertical** — nose pointing upward, tail at the bottom-right, gondola on the left side. A clear 90° clockwise rotation of the original |
| **Option B — mc_1 ❌** | `2DRotation+90_0_test_mc_1_input.png` | The blimp appears **flipped upside-down** (a vertical reflection / ReflectX result) — not a rotation |

---

## #Transformation Pattern

The transformation rotates the entire object **90° clockwise** about its centre. What was at the top of the image moves to the right; what was on the right moves to the bottom; what was at the bottom moves to the left; what was on the left moves to the top. The object's size and colour are perfectly preserved.

### 1. What stays the same
- **Object identity** — the blimp remains a blimp.
- **Colour and texture** — all surface detail is preserved.
- **Size** — the object does not grow or shrink.
- **Background** — always white.

### 2. What changes
- **Orientation** — the object is rotated 90° clockwise. The mapping is: `top → right`, `right → bottom`, `bottom → left`, `left → top`.
- **For the blimp specifically:** The horizontal body becomes vertical; what was the left end (nose) is now at the top; what was the right end (tail fin area) is now at the bottom; the gondola which hung below now protrudes to the left.
- **Mathematically:** `pixel(x, y) → pixel(H−y, x)` (for a square image of side H).

### 3. Why the object changes between training and test
- The dog demonstrates the clockwise rotation rule. The blimp — an elongated, asymmetric object with a clearly identifiable front (nose), back (tail fin), and underside (gondola) — makes the rotation result visually unambiguous and different from a reflection.

### 4. Why Option A (mc_0) is correct
- mc_0 shows the blimp rotated to a vertical orientation, nose pointing up, which is the exact result of a 90° clockwise rotation of the horizontal input blimp. The tail fin and gondola positions are consistent with this rotation direction.

### 5. Why Option B (mc_1) is a distractor
- mc_1 shows the blimp **flipped upside-down** — this is the result of a **ReflectX** (vertical flip), not a +90° rotation. This is the most sophisticated distractor in KiVA: rotation and reflection both change the object's apparent orientation, and for some objects and angles they can look superficially similar. The key distinguishing feature:
  - **Rotation (+90°):** the nose (left end) moves to the **top** of the frame.
  - **ReflectX:** the nose stays on the **left** (only top/bottom flip).
- A model that merely detects "orientation change" without distinguishing rotation from reflection will fail.

### 6. Corner cases
- **+90° vs −90° vs +180°:** These are three separate KiVA parameters and produce visually distinct results for asymmetric objects. +90° (clockwise) ≠ −90° (counter-clockwise); for the blimp, +90° puts the nose at the top, −90° puts the nose at the bottom.
- **Rotation vs Reflection (the critical distinction):** A rotation preserves **chirality** (handedness) — a right-pointing arrow rotated 90° clockwise points downward, but if you then reflect it, it points upward. These produce different results and KiVA deliberately uses this as a cross-concept distractor.
- The object is always rotated about its own centre — it does not translate (move) to a different position in the frame.

**In short:** "Rotate the entire object 90° clockwise about its centre — top goes right, right goes down, down goes left, left goes up. Size and colour unchanged."

---

## #Formal Rules

```
Rule_2DRotation_Plus90: RotateClockwise90

IF   Object(id=0, pixel_map=P[x,y], width=W, height=H)
AND  Transform = 2DRotation
AND  Parameter = +90
THEN Object(id=0, pixel_map=P[H−y, x])
AND  Top_becomes(Right)
AND  Right_becomes(Bottom)
AND  Bottom_becomes(Left)
AND  Left_becomes(Top)
AND  Size(unchanged)
AND  Colour(unchanged)
AND  Chirality(preserved)   ← key: rotation preserves handedness, reflection does not
```

**Distractor rule (what mc_1 incorrectly applies — ReflectX):**
```
Rule_ReflectX_WRONG: VerticalFlip
IF   Transform = Reflect, Parameter = X  ← wrong concept entirely
THEN Object(pixel_map=P[x, H−y])        ← flips top↔bottom, does not rotate
AND  Chirality(inverted)                 ← reflection inverts handedness
```

---

---

# Summary Table

| Concept | Parameter | Train Object | Test Object | Correct (mc_0) | Distractor (mc_1) | Distractor Logic |
|---------|-----------|-------------|------------|----------------|-------------------|-----------------|
| **Colour** | Red | Blue dog → Red dog | Blue citrus | Red citrus | Green citrus | Another valid Colour parameter (Green) |
| **Counting** | +1 | 2 balloons → 3 balloons | 2 rackets | 3 rackets | 1 racket | Opposite direction (−1 instead of +1) |
| **Resize** | 0.5XY | Medium dog → Small dog | Medium chess king | Small chess king | Large chess king | Opposite direction (2× instead of 0.5×) |
| **Reflect** | X (vertical flip) | Dog upright → Dog upside-down | Blimp horizontal | Blimp upside-down | Blimp unchanged | No transformation applied |
| **2DRotation** | +90° (clockwise) | Dog upright → Dog rotated | Blimp horizontal | Blimp vertical (nose up) | Blimp upside-down | ReflectX result, not a rotation |

---

## Key Cross-Concept Insight

The KiVA benchmark is carefully designed so that **distractors for one concept are often the correct answers for another**:

- The **Reflect/X mc_0** (upside-down blimp) = the **2DRotation mc_1** distractor
- The **Counting +1 mc_1** (1 object) = the **Counting −1 mc_0** correct answer
- The **Colour/Green mc_0** = the **Colour/Red mc_1** distractor

This means a model cannot succeed by simply detecting "something changed" — it must correctly identify **which specific transformation** occurred and **in which direction/to which target**.

---

*Document prepared for internal use at Corethink AI. For questions contact aryan@corethink.ai*
