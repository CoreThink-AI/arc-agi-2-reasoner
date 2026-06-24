"""
KiVA Pattern Detector — Phase 2

Mirrors the ARC pipeline's unit_patterns → intersect flow, but adapted for
image-based tasks with a known finite set of transformations.

Pipeline per task:
  1. Run REPEAT_COUNT vision calls on the training pair (input + output image)
  2. Each call returns a structured guess: concept + parameter + reasoning
  3. Majority-vote on concept and parameter separately (mirrors ARC's Counter)
  4. Summarise reasoning into a natural-language hint for Phase 3

Output format mirrors ARC's restructured_pattern_params:
  [{
      'name': 'Colour',
      'description': '...',
      'reason': '...',
      'params': {'parameter': ['Red']},
      'detailed_hint': '...'
  }]
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections import Counter
from typing import List, Optional

from pydantic import BaseModel

from kiva.llm_client import call_vision, call_text

# ── Config ─────────────────────────────────────────────────────────────────────

REPEAT_COUNT = int(os.getenv("KIVA_DETECTION_REPETITIONS", "3"))
CONCURRENT_REQUESTS = int(os.getenv("KIVA_DETECTION_CONCURRENCY", "3"))
_semaphore: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    return _semaphore


# ── Known transformations ──────────────────────────────────────────────────────

KNOWN_CONCEPTS = ["Counting", "Resize", "Colour", "Reflect", "2DRotation"]

KNOWN_PARAMETERS = {
    "Counting":   ["+1", "+2", "-1", "-2"],
    "Resize":     ["0.5XY", "2XY"],
    "Colour":     ["Red", "Green", "Blue"],
    "Reflect":    ["X", "Y"],
    "2DRotation": ["+90", "-90", "180"],
}

CONCEPT_DESCRIPTIONS = {
    "Counting":   "the number of objects changes (one more or fewer object appears)",
    "Resize":     "the object's physical size changes (grows larger or shrinks smaller)",
    "Colour":     "the object's color changes to a specific color (Red, Green, or Blue)",
    "Reflect":    "the object is mirror-flipped across a horizontal (X) or vertical (Y) axis",
    "2DRotation": "the object rotates by a fixed angle (+90°, -90°, or 180°)",
}


# ── Pydantic schema for one detection call ─────────────────────────────────────

class TransformationDetection(BaseModel):
    concept: str        # one of KNOWN_CONCEPTS
    parameter: str      # one of KNOWN_PARAMETERS[concept]
    reasoning: str      # brief natural-language explanation


# ── Prompts ────────────────────────────────────────────────────────────────────

_DETECTION_SYSTEM = (
    "You are an expert visual reasoning system. "
    "You will be shown two images: the LEFT image is the INPUT object and the "
    "RIGHT image is the OUTPUT object after a transformation has been applied. "
    "Your job is to identify exactly what transformation was applied."
)

_DETECTION_USER = """You are given two images (left = input, right = output).

The transformation belongs to exactly ONE of these five types:
- Counting   : {Counting}
- Resize     : {Resize}
- Colour     : {Colour}
- Reflect    : {Reflect}
- 2DRotation : {2DRotation}

And the specific parameter is one of a small fixed set per type:
- Counting   → +1, +2, -1, -2
- Resize     → 0.5XY (half size), 2XY (double size)
- Colour     → Red, Green, Blue
- Reflect    → X (horizontal flip), Y (vertical flip)
- 2DRotation → +90 (clockwise 90°), -90 (counter-clockwise 90°), 180 (half turn)

Carefully examine both images and respond with ONLY a JSON object in this exact format:
{{
  "concept": "<one of: Counting, Resize, Colour, Reflect, 2DRotation>",
  "parameter": "<exact parameter value from the list above>",
  "reasoning": "<one sentence explaining what visually changed>"
}}

Do not include any text before or after the JSON.""".format(**CONCEPT_DESCRIPTIONS)


_HINT_SYSTEM = "You are an expert visual reasoning assistant."

_HINT_USER = """You are given several explanations of the same visual transformation.
Synthesise them into one clear, concise description that:
1. Names the transformation type
2. Specifies the exact parameter value
3. Describes the visual change step by step

Explanations:
{explanations}

Write a single paragraph (2–4 sentences). Be precise and concrete."""


# ── Single detection call ──────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from model response, handling markdown code fences."""
    # Try to find a JSON block inside triple backticks
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        # Try to find a bare JSON object
        obj_match = re.search(r"\{.*\}", text, re.DOTALL)
        if obj_match:
            text = obj_match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _normalise_concept(raw: str) -> Optional[str]:
    """Case-insensitive match to a known concept."""
    raw = raw.strip()
    for c in KNOWN_CONCEPTS:
        if c.lower() == raw.lower():
            return c
    return None


def _normalise_parameter(concept: str, raw: str) -> Optional[str]:
    """Case-insensitive match to a known parameter for the given concept."""
    raw = raw.strip()
    known = KNOWN_PARAMETERS.get(concept, [])
    for p in known:
        if p.lower() == raw.lower():
            return p
    return None


async def _single_detection(
    train_input_path: str,
    train_output_path: str,
) -> Optional[TransformationDetection]:
    """Run one vision call and return a parsed TransformationDetection, or None."""
    async with _get_semaphore():
        response = await call_vision(
            user_text=_DETECTION_USER,
            image_paths=[train_input_path, train_output_path],
            system=_DETECTION_SYSTEM,
            max_tokens=512,
            temperature=0.0,
        )

    parsed = _extract_json(response)
    if not parsed:
        print(f"[KiVA detector] JSON parse failed — raw: {response[:120]}")
        return None

    concept = _normalise_concept(parsed.get("concept", ""))
    if not concept:
        print(f"[KiVA detector] Unknown concept: {parsed.get('concept')}")
        return None

    parameter = _normalise_parameter(concept, parsed.get("parameter", ""))
    if not parameter:
        print(f"[KiVA detector] Unknown parameter '{parsed.get('parameter')}' for concept '{concept}'")
        return None

    return TransformationDetection(
        concept=concept,
        parameter=parameter,
        reasoning=parsed.get("reasoning", ""),
    )


# ── Aggregation (mirrors ARC's intersect) ─────────────────────────────────────

async def _summarise_hint(concept: str, parameter: str, reasonings: List[str]) -> str:
    """Combine repeated reasonings into one clean hint string."""
    if not reasonings:
        return f"The transformation is {concept} with parameter {parameter}."
    if len(reasonings) == 1:
        return reasonings[0]

    numbered = "\n".join(f"{i+1}. {r}" for i, r in enumerate(reasonings))
    prompt = _HINT_USER.format(explanations=numbered)
    result = await call_text(user_text=prompt, system=_HINT_SYSTEM, max_tokens=512)
    return result.strip() or reasonings[0]


def _majority_vote(values: List[str]) -> Optional[str]:
    """Return the most common value; None if the list is empty."""
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


# ── Public API ─────────────────────────────────────────────────────────────────

async def detect_kiva_pattern(
    train_input_path: str,
    train_output_path: str,
    repeat: int = REPEAT_COUNT,
) -> dict:
    """
    Run repeated vision detections on a single training pair and aggregate.

    Mirrors ARC's unit_patterns() → intersect() flow.

    Args:
        train_input_path:  Path to the training input image.
        train_output_path: Path to the training output image.
        repeat:            Number of independent detection calls (default REPEAT_COUNT).

    Returns:
        A dict in the same shape as ARC's restructured_pattern_params[0]:
        {
            'name': 'Colour',
            'description': '...',
            'reason': '...',
            'params': {'parameter': ['Red']},
            'detailed_hint': '...',
            'concept_votes': {'Colour': 3},
            'param_votes': {'Red': 3},
            'confidence': 1.0,
        }
    """
    # Run all repetitions concurrently
    tasks = [
        _single_detection(train_input_path, train_output_path)
        for _ in range(repeat)
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect valid detections
    detections: List[TransformationDetection] = []
    for r in raw_results:
        if isinstance(r, Exception):
            print(f"[KiVA detector] Detection call raised: {r}")
        elif r is not None:
            detections.append(r)

    if not detections:
        return {
            'name': 'Unknown',
            'description': 'Detection failed for all attempts.',
            'reason': '',
            'params': {},
            'detailed_hint': 'Could not determine the transformation.',
            'concept_votes': {},
            'param_votes': {},
            'confidence': 0.0,
        }

    # Majority vote on concept and parameter (mirrors ARC's Counter)
    concept_votes = Counter(d.concept for d in detections)
    winning_concept = concept_votes.most_common(1)[0][0]

    # Vote only among detections that agree on the winning concept
    param_votes = Counter(
        d.parameter for d in detections if d.concept == winning_concept
    )
    winning_param = param_votes.most_common(1)[0][0]

    # Collect reasonings that agree with the winning (concept, param)
    reasonings = [
        d.reasoning for d in detections
        if d.concept == winning_concept and d.parameter == winning_param
    ]

    confidence = len([d for d in detections if d.concept == winning_concept]) / len(detections)

    # Summarise into a hint (mirrors ARC's summarize_reasons + generate_pattern_hint)
    hint = await _summarise_hint(winning_concept, winning_param, reasonings)

    description = CONCEPT_DESCRIPTIONS.get(winning_concept, "")

    return {
        'name': winning_concept,
        'description': description,
        'reason': reasonings[0] if reasonings else "",
        'params': {'parameter': [winning_param]},
        'detailed_hint': hint,
        'concept_votes': dict(concept_votes),
        'param_votes': dict(param_votes),
        'confidence': round(confidence, 2),
    }


async def get_kiva_hints(task) -> dict:
    """
    Top-level entry point for a KiVATask — matches the ARC pipeline's get_hints().

    Runs detect_kiva_pattern() on the task's single training pair.

    Args:
        task: A KiVATask dataclass (from kiva.loader).

    Returns:
        hint dict (same shape as detect_kiva_pattern output).
    """
    if not task.train:
        return {
            'name': 'Unknown',
            'description': '',
            'reason': '',
            'params': {},
            'detailed_hint': 'No training examples available.',
            'concept_votes': {},
            'param_votes': {},
            'confidence': 0.0,
        }

    pair = task.train[0]
    print(f"[KiVA detector] Detecting pattern for task {task.task_id}...")
    result = await detect_kiva_pattern(pair.input_path, pair.output_path)
    print(
        f"[KiVA detector] {task.task_id}: "
        f"concept={result['name']} param={result['params']} "
        f"confidence={result['confidence']}"
    )
    return result
