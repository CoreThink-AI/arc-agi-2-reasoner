"""
KiVA Evaluation Harness — Phase 4

Runs the full pipeline (Phase 2 pattern detection + Phase 3 MCQ solving)
across all 700 KiVA-easy tasks and writes results to CSV.

Optimisation notes
------------------
* All tasks are independent → fired concurrently via asyncio.gather.
  The semaphores inside llm_client / pattern_detector / solver act as the
  only rate-limiters; no artificial batching needed.
* Hint caching: after Phase 2 each hint is saved to kiva_results/hints/
  as JSON.  If the run is resumed, existing hints are loaded from disk
  instead of re-running vision calls.
* Incremental CSV writes: results are pushed to an asyncio.Queue and
  written by a dedicated writer coroutine so no progress is lost on crash.
* Resume: tasks whose task_id already appears in the output CSV are skipped.

Usage
-----
    # Full eval (all concepts, all 700 tasks)
    python -m kiva.eval

    # Single concept
    python -m kiva.eval --concepts Colour

    # Multiple concepts, limit trials, more solver attempts
    python -m kiva.eval --concepts Counting Resize --trials 10 --attempts 5

    # Disable resume (re-run everything)
    python -m kiva.eval --no-resume
"""

from __future__ import annotations

# Load .env BEFORE any kiva imports so that KIVA_VISION_MODEL / KIVA_TEXT_MODEL
# env vars are set before llm_client.py evaluates its module-level defaults.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except ImportError:
    pass

import argparse
import asyncio
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from kiva.loader import KiVATask, load_kiva_tasks, KIVA_EASY_CONCEPTS
from kiva.pattern_detector import get_kiva_hints
from kiva.solver import solve_kiva_task
import kiva.llm_client as _llm_client

# ── Output layout ──────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = Path("kiva_results")

CSV_COLUMNS = [
    "task_id", "concept", "parameter", "trial",
    # Phase 2 — pattern detection
    "hint_concept", "hint_param", "hint_confidence",
    "hint_concept_correct", "hint_param_correct",
    # Phase 3 — cross-domain
    "cross_predicted", "cross_gt", "cross_correct",
    # Phase 3 — within-domain
    "within_predicted", "within_gt", "within_correct",
    # Phase 3 — extrapolation
    "extrap_correct", "extrap_confidence",
    # Composite
    "all_correct",
    # Timing
    "elapsed_s",
]

# Sentinel pushed to the queue to signal the writer to stop
_DONE = object()


# ── Hint cache helpers ─────────────────────────────────────────────────────────

def _hint_cache_path(output_dir: Path, task_id: str) -> Path:
    return output_dir / "hints" / f"{task_id}.json"


def _load_cached_hint(output_dir: Path, task_id: str) -> Optional[dict]:
    p = _hint_cache_path(output_dir, task_id)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_hint(output_dir: Path, task_id: str, hint: dict) -> None:
    p = _hint_cache_path(output_dir, task_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(hint, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Completed-task index (for resume) ─────────────────────────────────────────

def _load_completed_ids(csv_path: Path) -> set:
    """Return the set of task_ids already written to the CSV."""
    if not csv_path.is_file():
        return set()
    done = set()
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row.get("task_id", "").strip()
                if tid:
                    done.add(tid)
    except Exception:
        pass
    return done


# ── Per-task pipeline ──────────────────────────────────────────────────────────

async def _run_task(
    task: KiVATask,
    output_dir: Path,
    num_attempts: int,
    queue: asyncio.Queue,
) -> None:
    """
    Full pipeline for one task:
      1. Load or detect hint (Phase 2)
      2. Solve all 3 MCQ stages (Phase 3)
      3. Push one CSV row dict to the queue
    """
    t0 = time.monotonic()

    # ── Phase 2: hint ──────────────────────────────────────────────────────
    hint = _load_cached_hint(output_dir, task.task_id)
    if hint is None:
        try:
            hint = await get_kiva_hints(task)
            _save_hint(output_dir, task.task_id, hint)
        except Exception as e:
            print(f"[eval] Hint failed for {task.task_id}: {e}")
            hint = {
                'name': '', 'params': {}, 'detailed_hint': '',
                'confidence': 0.0, 'concept_votes': {}, 'param_votes': {},
            }

    hint_concept = hint.get('name', '')
    hint_params  = hint.get('params', {})
    hint_param   = (hint_params.get('parameter') or [''])[0]
    hint_conf    = hint.get('confidence', 0.0)

    # ── Phase 3: solve ─────────────────────────────────────────────────────
    try:
        result = await solve_kiva_task(task, hint, num_attempts, run_all_stages=True)
    except Exception as e:
        print(f"[eval] Solve failed for {task.task_id}: {e}")
        result = {
            'cross_domain':  {'predicted': '', 'correct': False, 'gt_label': ''},
            'within_domain': {'predicted': '', 'correct': False, 'gt_param': task.parameter},
            'extrapolation': {'correct': False, 'confidence': 0.0},
            'all_correct':   False,
        }

    cross  = result['cross_domain']
    within = result['within_domain']
    extrap = result['extrapolation']
    elapsed = round(time.monotonic() - t0, 2)

    row = {
        "task_id":    task.task_id,
        "concept":    task.concept,
        "parameter":  task.parameter,
        "trial":      task.trial,
        # Phase 2
        "hint_concept":         hint_concept,
        "hint_param":           hint_param,
        "hint_confidence":      hint_conf,
        "hint_concept_correct": hint_concept == task.concept,
        "hint_param_correct":   hint_param   == task.parameter,
        # Cross-domain
        "cross_predicted": cross.get('predicted', ''),
        "cross_gt":        cross.get('gt_label',  ''),
        "cross_correct":   cross.get('correct',   False),
        # Within-domain
        "within_predicted": within.get('predicted', ''),
        "within_gt":        within.get('gt_param',  task.parameter),
        "within_correct":   within.get('correct',   False),
        # Extrapolation
        "extrap_correct":    extrap.get('correct',    False),
        "extrap_confidence": extrap.get('confidence', 0.0),
        # Composite
        "all_correct": result.get('all_correct', False),
        "elapsed_s":   elapsed,
    }
    await queue.put(row)


# ── CSV writer coroutine ───────────────────────────────────────────────────────

async def _csv_writer(csv_path: Path, queue: asyncio.Queue, append: bool) -> None:
    """
    Drain the queue and append rows to the CSV file.
    Runs as a concurrent coroutine alongside all task pipelines.
    """
    mode = "a" if append else "w"
    write_header = not csv_path.is_file() or not append

    with csv_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        while True:
            item = await queue.get()
            if item is _DONE:
                break
            writer.writerow(item)
            f.flush()  # ensure partial progress survives a crash


# ── Accuracy summary ───────────────────────────────────────────────────────────

def _print_summary(csv_path: Path) -> None:
    """Read the completed CSV and print per-concept accuracy tables."""
    if not csv_path.is_file():
        print("No results file found.")
        return

    rows: List[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Results file is empty.")
        return

    def pct(num: int, den: int) -> str:
        return f"{100 * num / den:.1f}%" if den else "—"

    def acc(subset, key: str) -> str:
        n = len(subset)
        c = sum(1 for r in subset if r.get(key) in ("True", True, "true"))
        return f"{pct(c, n)} ({c}/{n})"

    concepts = sorted(set(r["concept"] for r in rows))

    print("\n" + "=" * 72)
    print(f"{'KiVA-easy Accuracy Summary':^72}")
    print("=" * 72)
    print(f"{'Concept':<14} {'N':>5}  {'Phase2':>8}  {'Cross':>8}  {'Within':>8}  {'Extrap':>8}  {'All':>8}")
    print("-" * 72)

    total_stats: Dict[str, list] = {k: [] for k in ["n", "p2", "cross", "within", "extrap", "all"]}

    for concept in concepts:
        subset = [r for r in rows if r["concept"] == concept]
        n = len(subset)
        p2 = sum(1 for r in subset if r.get("hint_concept_correct") in ("True", True, "true"))
        cr = sum(1 for r in subset if r.get("cross_correct")         in ("True", True, "true"))
        wi = sum(1 for r in subset if r.get("within_correct")        in ("True", True, "true"))
        ex = sum(1 for r in subset if r.get("extrap_correct")        in ("True", True, "true"))
        al = sum(1 for r in subset if r.get("all_correct")           in ("True", True, "true"))
        print(
            f"{concept:<14} {n:>5}  "
            f"{pct(p2,n):>8}  {pct(cr,n):>8}  {pct(wi,n):>8}  {pct(ex,n):>8}  {pct(al,n):>8}"
        )
        for key, val in zip(["n","p2","cross","within","extrap","all"], [n,p2,cr,wi,ex,al]):
            total_stats[key].append(val)

    print("-" * 72)
    tn = sum(total_stats["n"])
    print(
        f"{'TOTAL':<14} {tn:>5}  "
        + "  ".join(
            f"{pct(sum(total_stats[k]), tn):>8}"
            for k in ["p2", "cross", "within", "extrap", "all"]
        )
    )
    print("=" * 72)
    print()
    print("Columns: Phase2=hint concept accuracy, Cross=cross-domain, Within=within-domain,")
    print("         Extrap=extrapolation (core MCQ), All=all 3 stages correct")
    print()


# ── Main entry point ───────────────────────────────────────────────────────────

async def run_eval(
    concepts: Optional[List[str]] = None,
    trials: Optional[int] = None,
    num_attempts: int = 3,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    resume: bool = True,
    kiva_root: Optional[str] = None,
) -> Path:
    """
    Run the full KiVA evaluation pipeline.

    Args:
        concepts:    Concept filter, e.g. ["Colour", "Counting"]. None = all 5.
        trials:      Limit trials per subdomain (None = all 50).
        num_attempts: Consensus attempts per solver stage.
        output_dir:  Directory to write CSV and hint cache.
        resume:      Skip tasks already present in the output CSV.
        kiva_root:   Path to KiVA repo root (falls back to KIVA_DATA_DIR env var).

    Returns:
        Path to the written CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hints").mkdir(exist_ok=True)

    csv_path = output_dir / "results.csv"

    # Load tasks
    all_tasks = load_kiva_tasks(
        kiva_root=kiva_root,
        difficulty="KiVA",
        concepts=concepts,
    )
    if trials is not None:
        # Keep first `trials` trials per concept-parameter pair
        seen: Dict[str, int] = {}
        filtered = []
        for t in all_tasks:
            key = f"{t.concept}_{t.parameter}"
            seen[key] = seen.get(key, 0) + 1
            if seen[key] <= trials:
                filtered.append(t)
        all_tasks = filtered

    # Resume: skip already-done tasks
    completed = _load_completed_ids(csv_path) if resume else set()
    tasks_to_run = [t for t in all_tasks if t.task_id not in completed]

    print(f"\n[eval] Output dir   : {output_dir.resolve()}")
    print(f"[eval] Total tasks  : {len(all_tasks)}")
    print(f"[eval] Already done : {len(completed)}")
    print(f"[eval] To run       : {len(tasks_to_run)}")
    print(f"[eval] Attempts/stage: {num_attempts}")
    print()

    if not tasks_to_run:
        print("[eval] Nothing to run — all tasks already complete.")
        _print_summary(csv_path)
        return csv_path

    # Queue for incremental CSV writes
    queue: asyncio.Queue = asyncio.Queue()

    # Writer coroutine (appends if resuming, creates fresh otherwise)
    writer_coro = _csv_writer(csv_path, queue, append=resume and csv_path.is_file())

    # Task coroutines — all fired concurrently; semaphores inside each module
    # throttle the actual LLM calls
    task_coros = [
        _run_task(task, output_dir, num_attempts, queue)
        for task in tasks_to_run
    ]

    t_start = time.monotonic()

    # Run writer + all task pipelines concurrently.
    # Tasks coroutines are gathered first; sentinel is pushed only after all
    # tasks have enqueued their rows, so the writer never stops early.
    writer_task = asyncio.create_task(writer_coro)
    await asyncio.gather(*task_coros)  # wait for ALL tasks to finish
    await queue.put(_DONE)             # signal writer to stop
    await writer_task                  # drain remaining queue items

    elapsed = time.monotonic() - t_start
    print(f"\n[eval] Finished {len(tasks_to_run)} tasks in {elapsed:.1f}s "
          f"({elapsed / max(len(tasks_to_run), 1):.1f}s/task avg)")
    print(f"[eval] Results saved to: {csv_path.resolve()}")

    # Write token usage summary
    token_stats = _llm_client.get_token_stats()
    token_stats["model"]            = os.getenv("KIVA_VISION_MODEL", "unknown")
    token_stats["tasks_completed"]  = len(tasks_to_run)
    token_stats["elapsed_s"]        = round(elapsed, 1)
    token_path = output_dir / "token_usage.json"
    token_path.write_text(json.dumps(token_stats, indent=2), encoding="utf-8")
    print(f"[eval] Token usage      : {token_stats['total_tokens']:,} total "
          f"({token_stats['prompt_tokens']:,} prompt + {token_stats['completion_tokens']:,} completion)")
    print(f"[eval] Token usage saved: {token_path.resolve()}")

    _print_summary(csv_path)
    return csv_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the KiVA neurosymbolic evaluation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--concepts", nargs="+",
        choices=list(KIVA_EASY_CONCEPTS.keys()),
        default=None,
        help="Concepts to evaluate (default: all 5)",
    )
    parser.add_argument(
        "--trials", type=int, default=None,
        help="Max trials per concept-parameter subdomain (default: all 50)",
    )
    parser.add_argument(
        "--attempts", type=int, default=3,
        help="Consensus attempts per solver stage (default: 3)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV and hint cache (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Re-run all tasks even if results already exist",
    )
    parser.add_argument(
        "--kiva-root", type=str, default=None,
        help="Path to KiVA repo root (overrides KIVA_DATA_DIR env var)",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Skip evaluation and just print summary from existing CSV",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=32000,
        help="Max output tokens per LLM call — benchmark cap (default: 32000)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Vision+text model override (e.g. anthropic/claude-3.5-sonnet, openai/gpt-4o, gemini-2.5-flash)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Apply global token cap before any LLM calls
    _llm_client.set_max_tokens_cap(args.max_tokens)
    print(f"[eval] Max tokens/call: {args.max_tokens}")

    # Override model via env vars if --model is passed
    if args.model:
        os.environ["KIVA_VISION_MODEL"] = args.model
        os.environ["KIVA_TEXT_MODEL"]   = args.model
        _llm_client.DEFAULT_VISION_MODEL = args.model
        _llm_client.DEFAULT_TEXT_MODEL   = args.model
        print(f"[eval] Model          : {args.model}")

    if args.summary_only:
        csv_path = Path(args.output_dir) / "results.csv"
        _print_summary(csv_path)
    else:
        asyncio.run(run_eval(
            concepts=args.concepts,
            trials=args.trials,
            num_attempts=args.attempts,
            output_dir=args.output_dir,
            resume=not args.no_resume,
            kiva_root=args.kiva_root,
        ))
