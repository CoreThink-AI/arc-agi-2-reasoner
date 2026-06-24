"""
KiVA Dataset Loader
Reads the KiVA benchmark from disk into structured task objects.

Actual dataset layout (verified from cloned repo):
    <root>/
    └── transformed objects/
        └── KiVA/
            ├── trial_tracker/
            │   ├── output_ColourRed.txt        # 4 lines × n_trials
            │   ├── output_Counting+1.txt
            │   └── ...
            ├── Colour/                         # flat dir, all Colour params mixed
            │   ├── ColourRed_0_train_0_input.png
            │   ├── ColourRed_0_train_0_output.png
            │   ├── ColourRed_0_test_0_input.png
            │   ├── ColourRed_0_test_mc_0_input.png   ← CORRECT answer
            │   ├── ColourRed_0_test_mc_1_input.png   ← distractor
            │   ├── ColourBlue_0_train_0_input.png
            │   └── ...
            ├── Counting/
            ├── Resize/
            ├── Reflect/
            └── 2DRotation/

Metadata format (4 lines per trial, repeated):
    train_0_input: <value>
    train_0_output: <value>
    test_input: <value>
    mc: <distractor_value>
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class KiVATrainPair:
    input_path: str
    output_path: str


@dataclass
class KiVATask:
    task_id: str                      # e.g. "Colour_Red_0"
    concept: str                      # e.g. "Colour"
    parameter: str                    # e.g. "Red"
    trial: int                        # e.g. 0
    train: List[KiVATrainPair]        # exactly 1 training pair in KiVA-easy
    test_input_path: str              # the test input image
    test_correct_path: str            # mc_0 — always the correct answer
    test_incorrect_paths: List[str]   # mc_1+ — distractors
    metadata: dict = field(default_factory=dict)  # raw parsed values for this trial

    def all_image_paths(self) -> List[str]:
        paths = [p.input_path for p in self.train] + [p.output_path for p in self.train]
        paths += [self.test_input_path, self.test_correct_path] + self.test_incorrect_paths
        return paths

    def missing_images(self) -> List[str]:
        return [p for p in self.all_image_paths() if not os.path.isfile(p)]


# ── KiVA-easy concept/parameter map ───────────────────────────────────────────

KIVA_EASY_CONCEPTS: Dict[str, List[str]] = {
    "Counting":   ["+1", "+2", "-1", "-2"],
    "Resize":     ["0.5XY", "2XY"],
    "Colour":     ["Red", "Green", "Blue"],
    "Reflect":    ["X", "Y"],
    "2DRotation": ["+90", "-90", "180"],
}

# Canonical directory name per concept
_CONCEPT_DIR: Dict[str, str] = {
    "Counting":   "Counting",
    "Resize":     "Resize",
    "Colour":     "Colour",
    "Reflect":    "Reflect",
    "2DRotation": "2DRotation",
}


# ── Metadata parsing ───────────────────────────────────────────────────────────

def _parse_trial_metadata(txt_path: Path) -> List[dict]:
    """
    Parse a trial_tracker metadata file into one dict per trial.

    Format (4 lines × n_trials, no blank separators):
        train_0_input: <val>
        train_0_output: <val>
        test_input: <val>
        mc: <distractor_val>
    """
    trials = []
    if not txt_path.is_file():
        return trials

    lines = [l.strip() for l in txt_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Group into chunks of 4
    for i in range(0, len(lines) - 3, 4):
        chunk = lines[i : i + 4]
        entry = {}
        for line in chunk:
            if ": " in line:
                key, _, val = line.partition(": ")
                entry[key.strip()] = val.strip()
        if entry:
            trials.append(entry)
    return trials


# ── File discovery ─────────────────────────────────────────────────────────────

def _build_prefix(concept: str, parameter: str) -> str:
    """Return the filename prefix, e.g. 'ColourRed', 'Counting+1', 'Resize0.5XY'."""
    return f"{concept}{parameter}"


def _find_files_for_trial(
    concept_dir: Path,
    prefix: str,
    trial: int,
) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
    """
    Locate all images for one trial.

    Returns:
        (train_input, train_output, test_input, [test_mc_0, test_mc_1, ...])
        mc_0 is the CORRECT answer; mc_1+ are distractors.
    """
    base = f"{prefix}_{trial}_"

    def find(pattern: str) -> Optional[str]:
        matches = sorted(concept_dir.glob(pattern))
        return str(matches[0]) if matches else None

    def find_all(pattern: str) -> List[str]:
        return [str(p) for p in sorted(concept_dir.glob(pattern))]

    train_input  = find(f"{base}train_0_input.png")
    train_output = find(f"{base}train_0_output.png")
    test_input   = find(f"{base}test_0_input.png")
    test_mc_all  = find_all(f"{base}test_mc_*.png")

    return train_input, train_output, test_input, test_mc_all


# ── Public API ─────────────────────────────────────────────────────────────────

def load_kiva_tasks(
    kiva_root: Optional[str] = None,
    difficulty: str = "KiVA",
    concepts: Optional[List[str]] = None,
) -> List[KiVATask]:
    """
    Load all KiVA tasks from the dataset directory.

    Args:
        kiva_root:  Path to the root of the cloned KiVA repo.
                    Falls back to KIVA_DATA_DIR env var.
        difficulty: "KiVA" (easy) or "KiVA-adults".
        concepts:   Optional list of concept names to filter, e.g. ["Colour"].
                    None loads all 5 concepts.

    Returns:
        List of KiVATask, one per trial.
    """
    root = Path(kiva_root or os.environ.get("KIVA_DATA_DIR", ""))
    if not root.exists():
        raise FileNotFoundError(
            f"KiVA data root not found: {root}\n"
            "Set KIVA_DATA_DIR in your .env file or pass kiva_root=.\n"
            "Run  python -m kiva.dataset_setup  for instructions."
        )

    data_dir = root / "transformed objects" / difficulty
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Could not find '{difficulty}' directory.\nExpected: {data_dir}"
        )

    tracker_dir = data_dir / "trial_tracker"
    target_concepts = concepts or list(KIVA_EASY_CONCEPTS.keys())
    all_tasks: List[KiVATask] = []

    for concept in target_concepts:
        params = KIVA_EASY_CONCEPTS.get(concept, [])
        concept_dir = data_dir / _CONCEPT_DIR.get(concept, concept)

        if not concept_dir.exists():
            print(f"[KiVA loader] Concept directory missing: {concept_dir} — skipping")
            continue

        for param in params:
            prefix = _build_prefix(concept, param)
            txt_path = tracker_dir / f"output_{prefix}.txt"
            trial_metas = _parse_trial_metadata(txt_path)

            if not trial_metas:
                print(f"[KiVA loader] No metadata for {concept}/{param} — skipping")
                continue

            loaded = 0
            for trial_idx, meta in enumerate(trial_metas):
                train_inp, train_out, test_inp, test_mc = _find_files_for_trial(
                    concept_dir, prefix, trial_idx
                )

                if not train_inp or not train_out:
                    continue  # skip if training images missing
                if not test_inp or len(test_mc) < 1:
                    continue  # skip if test images missing

                task = KiVATask(
                    task_id=f"{concept}_{param}_{trial_idx}",
                    concept=concept,
                    parameter=param,
                    trial=trial_idx,
                    train=[KiVATrainPair(input_path=train_inp, output_path=train_out)],
                    test_input_path=test_inp,
                    test_correct_path=test_mc[0],       # mc_0 is always correct
                    test_incorrect_paths=test_mc[1:],   # mc_1+ are distractors
                    metadata=meta,
                )
                all_tasks.append(task)
                loaded += 1

            print(f"[KiVA loader] {concept}/{param}: {loaded} trials loaded")

    print(f"[KiVA loader] Total: {len(all_tasks)} tasks loaded")
    return all_tasks


def load_kiva_tasks_by_concept(
    concept: str,
    kiva_root: Optional[str] = None,
    difficulty: str = "KiVA",
) -> List[KiVATask]:
    """Convenience wrapper to load tasks for a single concept."""
    return load_kiva_tasks(kiva_root=kiva_root, difficulty=difficulty, concepts=[concept])
