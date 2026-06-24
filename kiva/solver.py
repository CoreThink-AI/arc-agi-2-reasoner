"""
KiVA MCQ Solver — Phase 3

Mirrors the ARC pipeline's get_consensus_response() / get_solved_outputs() flow,
adapted for multiple-choice instead of grid generation.

KiVA evaluation has three sequential stages per task:
  1. Cross-domain  — "What type of transformation is shown?"
                      (Counting / Resize / Colour / Reflect / 2DRotation)
  2. Within-domain — "What is the specific parameter?"
                      (e.g. Red, +1, 0.5XY, X, +90)
  3. Extrapolation — "Which image (A/B/C) shows the correct transformation
                      applied to the test input?"

Each stage is run with majority voting across NUM_SOLVER_ATTEMPTS calls,
matching the ARC solver's self-consistency approach.

The hint from Phase 2 (pattern_detector.get_kiva_hints) is used to ground
the cross-domain and within-domain answers; the extrapolation stage is a
full vision call so the model must visually verify the correct option.
"""

from __future__ import annotations

import asyncio
import os
import random
import re
from collections import Counter
from typing import List, Optional, Tuple

from kiva.llm_client import call_vision, call_text
from kiva.loader import KiVATask

# ── Config ─────────────────────────────────────────────────────────────────────

NUM_SOLVER_ATTEMPTS = int(os.getenv("KIVA_SOLVER_ATTEMPTS", "3"))
SOLVER_CONCURRENCY  = int(os.getenv("KIVA_SOLVER_CONCURRENCY", "3"))

_semaphore: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(SOLVER_CONCURRENCY)
    return _semaphore


# ── Cross-domain label mapping ─────────────────────────────────────────────────
# Matches KiVA's original evaluation script (correct_cross_domain())

_CONCEPT_TO_CROSS = {
    "Counting":   "Number of objects",
    "Resize":     "Size of objects",
    "Colour":     "Color of objects",
    "2DRotation": "Orientation of objects",
    "Reflect":    "Orientation of objects",
}

_ALL_CROSS_LABELS = [
    "Number of objects",
    "Size of objects",
    "Color of objects",
    "Orientation of objects",
]

# ── Letter labels ──────────────────────────────────────────────────────────────

_LETTERS = ["A", "B", "C", "D", "E"]


# ── Prompt templates ───────────────────────────────────────────────────────────

_CROSS_SYSTEM = "You are an expert visual reasoning assistant."

_CROSS_USER = """You are shown two images: LEFT = input object, RIGHT = output object after a transformation.

Which of the following best describes what changed?
{choices}

Reply with ONLY the letter in parentheses of your chosen answer, e.g. (A). Nothing else."""


_WITHIN_SYSTEM = "You are an expert visual reasoning assistant."

_WITHIN_USER = """You are shown two images: LEFT = input object, RIGHT = output object.

The transformation type is: {cross_label}
Which specific rule applies?
{choices}

Reply with ONLY the letter in parentheses of your chosen answer, e.g. (A). Nothing else."""


_EXTRAP_SYSTEM = (
    "You are an expert visual reasoning assistant solving a visual analogy puzzle."
)

_EXTRAP_USER = """You are given a visual analogy task.

The FIRST image shows the TRAINING INPUT (before transformation).
The SECOND image shows the TRAINING OUTPUT (after transformation).
The THIRD image shows the TEST INPUT (a new object to transform).
The remaining images are multiple-choice options labeled {labels}.

Transformation hint: {hint}

Which option shows the same transformation correctly applied to the TEST INPUT?
Reply with ONLY the letter in parentheses of the correct option, e.g. (A). Nothing else."""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_letter(response: str, valid: List[str]) -> Optional[str]:
    """
    Extract the first valid letter from a model response.
    Looks for patterns like (A), A), (A, or bare A.
    """
    response = response.strip().upper()
    # Prefer parenthesised form first
    for pattern in [r'\(([A-Z])\)', r'([A-Z])\)', r'\(([A-Z])', r'\b([A-Z])\b']:
        m = re.search(pattern, response)
        if m and m.group(1) in valid:
            return m.group(1)
    return None


def _majority(values: List[str]) -> Optional[str]:
    """Return most common value, None if empty."""
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _make_labeled_choices(options: List[str]) -> Tuple[str, List[str]]:
    """
    Turn a list of option strings into a labeled choices block and the letter list.
    e.g. ["Red", "Blue"] → "(A) Red\n(B) Blue", ["A", "B"]
    """
    letters = _LETTERS[: len(options)]
    block = "\n".join(f"({l}) {opt}" for l, opt in zip(letters, options))
    return block, letters


# ── Stage 1: Cross-domain ──────────────────────────────────────────────────────

async def _single_cross_domain(
    train_input_path: str,
    train_output_path: str,
    correct_label: str,
) -> Optional[str]:
    """
    One vision call for the cross-domain stage.
    Returns the predicted cross-domain label string, or None on failure.
    """
    # Build a shuffled choices list that always includes the correct label
    distractors = [l for l in _ALL_CROSS_LABELS if l != correct_label]
    chosen_distractors = random.sample(distractors, min(2, len(distractors)))
    options = chosen_distractors + [correct_label]
    random.shuffle(options)

    choices_block, letters = _make_labeled_choices(options)
    letter_to_label = dict(zip(letters, options))

    async with _get_semaphore():
        response = await call_vision(
            user_text=_CROSS_USER.format(choices=choices_block),
            image_paths=[train_input_path, train_output_path],
            system=_CROSS_SYSTEM,
            max_tokens=64,
            temperature=0.0,
        )

    chosen_letter = _extract_letter(response, letters)
    if not chosen_letter:
        return None
    return letter_to_label[chosen_letter]


async def solve_cross_domain(
    task: KiVATask,
    hint: dict,
    num_attempts: int = NUM_SOLVER_ATTEMPTS,
) -> dict:
    """
    Stage 1: Cross-domain classification.

    Uses both the hint (from Phase 2) and repeated vision calls with majority voting.
    The hint provides a strong prior; vision calls verify visually.

    Returns:
        {
            'predicted':  'Color of objects',   # majority-voted label
            'correct':    True,
            'gt_label':   'Color of objects',
            'votes':      {'Color of objects': 3},
            'raw_responses': [...],
        }
    """
    concept = hint.get('name', '')
    gt_label = _CONCEPT_TO_CROSS.get(concept, '')
    pair = task.train[0]

    tasks = [
        _single_cross_domain(pair.input_path, pair.output_path, gt_label)
        for _ in range(num_attempts)
    ]
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    predictions = [r for r in raw if isinstance(r, str)]
    votes = dict(Counter(predictions))
    predicted = _majority(predictions) or ''

    return {
        'predicted':  predicted,
        'correct':    predicted == gt_label,
        'gt_label':   gt_label,
        'votes':      votes,
    }


# ── Stage 2: Within-domain ─────────────────────────────────────────────────────

def _get_within_options(concept: str, correct_param: str) -> List[str]:
    """Return shuffled within-domain options including the correct parameter."""
    from kiva.pattern_detector import KNOWN_PARAMETERS
    all_params = KNOWN_PARAMETERS.get(concept, [correct_param])
    distractors = [p for p in all_params if p != correct_param]
    chosen = random.sample(distractors, min(2, len(distractors)))
    options = chosen + [correct_param]
    random.shuffle(options)
    return options


async def _single_within_domain(
    train_input_path: str,
    train_output_path: str,
    cross_label: str,
    correct_param: str,
    concept: str,
) -> Optional[str]:
    """One vision call for the within-domain stage."""
    options = _get_within_options(concept, correct_param)
    choices_block, letters = _make_labeled_choices(options)
    letter_to_param = dict(zip(letters, options))

    async with _get_semaphore():
        response = await call_vision(
            user_text=_WITHIN_USER.format(
                cross_label=cross_label,
                choices=choices_block,
            ),
            image_paths=[train_input_path, train_output_path],
            system=_WITHIN_SYSTEM,
            max_tokens=64,
            temperature=0.0,
        )

    chosen_letter = _extract_letter(response, letters)
    if not chosen_letter:
        return None
    return letter_to_param[chosen_letter]


async def solve_within_domain(
    task: KiVATask,
    hint: dict,
    num_attempts: int = NUM_SOLVER_ATTEMPTS,
) -> dict:
    """
    Stage 2: Within-domain parameter identification.

    Returns:
        {
            'predicted':  '+1',
            'correct':    True,
            'gt_param':   '+1',
            'votes':      {'+1': 3},
        }
    """
    concept   = hint.get('name', '')
    params    = hint.get('params', {})
    gt_param  = (params.get('parameter') or [task.parameter])[0]
    cross_label = _CONCEPT_TO_CROSS.get(concept, '')
    pair = task.train[0]

    tasks = [
        _single_within_domain(
            pair.input_path, pair.output_path,
            cross_label, gt_param, concept,
        )
        for _ in range(num_attempts)
    ]
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    predictions = [r for r in raw if isinstance(r, str)]
    votes = dict(Counter(predictions))
    predicted = _majority(predictions) or ''

    return {
        'predicted': predicted,
        'correct':   predicted == gt_param,
        'gt_param':  gt_param,
        'votes':     votes,
    }


# ── Stage 3: Extrapolation (core MCQ) ─────────────────────────────────────────

async def _single_extrapolation(
    train_input_path: str,
    train_output_path: str,
    test_input_path: str,
    shuffled_option_paths: List[str],
    shuffled_letters: List[str],
    hint_text: str,
) -> Optional[str]:
    """
    One vision call for the extrapolation stage.

    Images passed (in order):
      [train_input, train_output, test_input, option_0, option_1, ...]
    Returns the chosen letter, or None on failure.
    """
    image_paths = (
        [train_input_path, train_output_path, test_input_path]
        + shuffled_option_paths
    )
    labels_str = ", ".join(f"({l})" for l in shuffled_letters)

    async with _get_semaphore():
        response = await call_vision(
            user_text=_EXTRAP_USER.format(
                labels=labels_str,
                hint=hint_text,
            ),
            image_paths=image_paths,
            system=_EXTRAP_SYSTEM,
            max_tokens=64,
            temperature=0.0,
        )

    return _extract_letter(response, shuffled_letters)


async def solve_extrapolation(
    task: KiVATask,
    hint: dict,
    num_attempts: int = NUM_SOLVER_ATTEMPTS,
) -> dict:
    """
    Stage 3: Extrapolation — pick the correct MC option.

    The correct answer is always task.test_correct_path (mc_0).
    Options are shuffled each attempt so the model cannot exploit position.
    Majority vote across attempts selects the final answer.

    Returns:
        {
            'predicted_letter':  'B',      # majority-voted letter
            'correct':           True,     # did the chosen option == mc_0?
            'chosen_path':       '...png', # which image path was chosen
            'correct_path':      '...png', # the ground-truth correct path
            'option_map':        {'A': '...', 'B': '...', 'C': '...'},
            'votes':             {'B': 2, 'A': 1},
            'confidence':        0.67,
        }
    """
    pair        = task.train[0]
    all_options = [task.test_correct_path] + task.test_incorrect_paths
    hint_text   = hint.get('detailed_hint', '') or hint.get('description', '')

    # Run all attempts, each with an independent shuffle
    async def one_attempt() -> Tuple[Optional[str], dict]:
        shuffled = all_options[:]
        random.shuffle(shuffled)
        letters = _LETTERS[: len(shuffled)]
        letter_to_path = dict(zip(letters, shuffled))

        chosen_letter = await _single_extrapolation(
            pair.input_path,
            pair.output_path,
            task.test_input_path,
            shuffled,
            letters,
            hint_text,
        )
        return chosen_letter, letter_to_path

    raw = await asyncio.gather(
        *[one_attempt() for _ in range(num_attempts)],
        return_exceptions=True,
    )

    # Collect (chosen_path, letter) pairs
    chosen_paths: List[str] = []
    all_letters:  List[str] = []

    for r in raw:
        if isinstance(r, Exception):
            print(f"[KiVA solver] Extrapolation attempt raised: {r}")
            continue
        letter, lmap = r
        if letter and letter in lmap:
            chosen_paths.append(lmap[letter])
            all_letters.append(letter)

    # Majority vote on the chosen PATH (not letter, since shuffle differs per attempt)
    predicted_path = _majority(chosen_paths)
    is_correct = predicted_path == task.test_correct_path

    # For reporting: what letter did the majority pick in its own run?
    # Use the most common raw letter as an approximation
    voted_letter = _majority(all_letters) or ''
    path_votes = dict(Counter(chosen_paths))

    return {
        'predicted_letter': voted_letter,
        'correct':          is_correct,
        'chosen_path':      predicted_path or '',
        'correct_path':     task.test_correct_path,
        'path_votes':       path_votes,
        'votes':            dict(Counter(all_letters)),
        'confidence':       round(
            Counter(chosen_paths).most_common(1)[0][1] / len(chosen_paths), 2
        ) if chosen_paths else 0.0,
    }


# ── Top-level orchestration ────────────────────────────────────────────────────

async def solve_kiva_task(
    task: KiVATask,
    hint: dict,
    num_attempts: int = NUM_SOLVER_ATTEMPTS,
    run_all_stages: bool = True,
) -> dict:
    """
    Full 3-stage KiVA evaluation for one task. Mirrors ARC's solve_arc_task().

    KiVA's original protocol is sequential (within-domain only asked if
    cross-domain is correct; extrapolation always asked). We always run all
    stages for complete paper numbers, but record which stage each answer
    came from.

    Args:
        task:           KiVATask from the loader.
        hint:           Output of get_kiva_hints() from Phase 2.
        num_attempts:   Consensus attempts per stage (default NUM_SOLVER_ATTEMPTS).
        run_all_stages: If False, stops after first wrong stage (KiVA-original
                        protocol). If True (default), runs all 3 regardless.

    Returns:
        {
            'task_id':      'Colour_Red_0',
            'concept':      'Colour',
            'parameter':    'Red',
            'cross_domain':  { ... solve_cross_domain result ... },
            'within_domain': { ... solve_within_domain result ... },
            'extrapolation': { ... solve_extrapolation result ... },
            'all_correct':  True,   # all 3 stages correct
        }
    """
    print(f"[KiVA solver] Solving {task.task_id}...")

    cross  = await solve_cross_domain(task, hint, num_attempts)
    print(f"  cross_domain:  predicted={cross['predicted']!r}  correct={cross['correct']}")

    if not run_all_stages and not cross['correct']:
        within = {'predicted': '', 'correct': False, 'gt_param': task.parameter, 'votes': {}, 'skipped': True}
        extrap = {'predicted_letter': '', 'correct': False, 'chosen_path': '', 'correct_path': task.test_correct_path, 'votes': {}, 'confidence': 0.0, 'skipped': True}
    else:
        within = await solve_within_domain(task, hint, num_attempts)
        print(f"  within_domain: predicted={within['predicted']!r}  correct={within['correct']}")

        extrap = await solve_extrapolation(task, hint, num_attempts)
        print(f"  extrapolation: correct={extrap['correct']}  confidence={extrap['confidence']}")

    return {
        'task_id':       task.task_id,
        'concept':       task.concept,
        'parameter':     task.parameter,
        'cross_domain':  cross,
        'within_domain': within,
        'extrapolation': extrap,
        'all_correct':   cross['correct'] and within['correct'] and extrap['correct'],
    }


async def solve_kiva_batch(
    tasks: List[KiVATask],
    hints: List[dict],
    num_attempts: int = NUM_SOLVER_ATTEMPTS,
    run_all_stages: bool = True,
) -> List[dict]:
    """
    Solve a batch of KiVA tasks concurrently. Mirrors ARC's process_batch().

    Args:
        tasks:        List of KiVATask objects.
        hints:        Corresponding hints (one per task, same order).
        num_attempts: Consensus attempts per stage.
        run_all_stages: See solve_kiva_task().

    Returns:
        List of result dicts, one per task.
    """
    coroutines = [
        solve_kiva_task(task, hint, num_attempts, run_all_stages)
        for task, hint in zip(tasks, hints)
    ]
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    clean = []
    for task, r in zip(tasks, results):
        if isinstance(r, Exception):
            print(f"[KiVA solver] Task {task.task_id} raised: {r}")
            clean.append({'task_id': task.task_id, 'error': str(r)})
        else:
            clean.append(r)
    return clean
