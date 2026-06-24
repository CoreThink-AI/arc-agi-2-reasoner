"""
ARC-AGI Baseline Evaluation Harness

Directly prompts frontier models on ARC-AGI tasks with no neurosymbolic pipeline.
Measures raw model performance as a baseline to compare against the full pipeline.

Usage:
    python -m arc_baseline.eval --model openai/gpt-4o --output-dir arc_baseline_results/gpt-4o
    python -m arc_baseline.eval --model anthropic/claude-sonnet-4-5 --output-dir arc_baseline_results/claude-sonnet-4-5
    python -m arc_baseline.eval --model gemini-2.5-flash --output-dir arc_baseline_results/gemini-2-5-flash
    python -m arc_baseline.eval --summary-only --output-dir arc_baseline_results/gpt-4o
"""

from __future__ import annotations

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
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import kiva.llm_client as _llm_client

# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are solving ARC (Abstraction and Reasoning Corpus) puzzles. "
    "Each puzzle shows input→output grid pairs that share a hidden transformation rule. "
    "Study the examples carefully, then apply the exact same rule to the test input. "
    "Respond with ONLY the output grid — one row per line, integers separated by spaces. "
    "No explanation, no markdown, no extra text."
)

_EXAMPLE_TMPL = "Example {i}:\nInput:\n{inp}\nOutput:\n{out}"
_USER_TMPL = (
    "{examples}\n\n"
    "Now apply the same rule to this test input:\n\n"
    "Input:\n{test_inp}\n\n"
    "Output grid:"
)

# ── Grid helpers ───────────────────────────────────────────────────────────────

def _grid_to_str(grid: List[List[int]]) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in grid)


def _parse_grid(text: str) -> List[List[int]]:
    rows = []
    for line in text.strip().splitlines():
        nums = re.findall(r'-?\d+', line)
        if nums:
            rows.append([int(n) for n in nums])
    return rows

# ── Task loading ───────────────────────────────────────────────────────────────

DEFAULT_DATA_PATH = Path("arc-agi_test_challenges.json")


def load_tasks(data_path: Path) -> List[Dict]:
    with data_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    tasks = []
    if isinstance(raw, dict):
        for task_id, task in raw.items():
            for i, test_case in enumerate(task["test"]):
                uid = f"{task_id}_{i}" if len(task["test"]) > 1 else task_id
                tasks.append({
                    "task_id":     uid,
                    "train":       task["train"],
                    "test_input":  test_case["input"],
                    "test_output": test_case.get("output"),
                })
    return tasks

# ── Completed-task index ───────────────────────────────────────────────────────

def _load_completed(csv_path: Path) -> set:
    if not csv_path.is_file():
        return set()
    done = set()
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("task_id"):
                    done.add(row["task_id"])
    except Exception:
        pass
    return done

# ── Per-task solver ────────────────────────────────────────────────────────────

async def _solve_task(task: Dict, sem: asyncio.Semaphore) -> Dict:
    t0 = time.monotonic()

    examples_str = "\n\n".join(
        _EXAMPLE_TMPL.format(
            i=i + 1,
            inp=_grid_to_str(ex["input"]),
            out=_grid_to_str(ex["output"]),
        )
        for i, ex in enumerate(task["train"])
    )
    user_text = _USER_TMPL.format(
        examples=examples_str,
        test_inp=_grid_to_str(task["test_input"]),
    )

    async with sem:
        response = await _llm_client.call_text(
            user_text=user_text,
            system=_SYSTEM,
            max_tokens=2048,
        )

    predicted = _parse_grid(response)
    gt = task["test_output"]
    correct = (predicted == gt) if gt is not None else None

    return {
        "task_id":        task["task_id"],
        "correct":        correct,
        "predicted_rows": len(predicted),
        "gt_rows":        len(gt) if gt is not None else "",
        "elapsed_s":      round(time.monotonic() - t0, 2),
        "raw_response":   response[:500],
    }

# ── CSV writer ─────────────────────────────────────────────────────────────────

CSV_COLUMNS = ["task_id", "correct", "predicted_rows", "gt_rows", "elapsed_s", "raw_response"]
_DONE = object()


async def _csv_writer(csv_path: Path, queue: asyncio.Queue, append: bool) -> None:
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
            f.flush()

# ── Summary ────────────────────────────────────────────────────────────────────

def _print_summary(csv_path: Path) -> None:
    if not csv_path.is_file():
        print("No results file found.")
        return
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("Results file is empty.")
        return

    n = len(rows)
    c = sum(1 for r in rows if r.get("correct") in ("True", True, "true"))
    print(f"\n{'=' * 50}")
    print(f"  ARC-AGI Baseline Results")
    print(f"{'=' * 50}")
    print(f"  Model    : {rows[0].get('model', 'unknown') if 'model' in CSV_COLUMNS else 'see token_usage.json'}")
    print(f"  Tasks    : {n}")
    print(f"  Correct  : {c}")
    print(f"  Accuracy : {100 * c / n:.1f}%")
    print(f"{'=' * 50}\n")

# ── Main eval ──────────────────────────────────────────────────────────────────

async def run_eval(
    data_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = Path("arc_baseline_results/default"),
    resume: bool = True,
    concurrency: int = 10,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"

    all_tasks = load_tasks(data_path)
    completed = _load_completed(csv_path) if resume else set()
    tasks_to_run = [t for t in all_tasks if t["task_id"] not in completed]

    print(f"\n[eval] Data          : {data_path}")
    print(f"[eval] Output dir    : {output_dir.resolve()}")
    print(f"[eval] Model         : {_llm_client.DEFAULT_TEXT_MODEL}")
    print(f"[eval] Total tasks   : {len(all_tasks)}")
    print(f"[eval] Already done  : {len(completed)}")
    print(f"[eval] To run        : {len(tasks_to_run)}")
    print()

    if not tasks_to_run:
        print("[eval] Nothing to run — all tasks already complete.")
        _print_summary(csv_path)
        return csv_path

    sem = asyncio.Semaphore(concurrency)
    queue: asyncio.Queue = asyncio.Queue()

    async def _run_and_push(task: Dict) -> None:
        result = await _solve_task(task, sem)
        await queue.put(result)

    t_start = time.monotonic()
    writer_task = asyncio.create_task(
        _csv_writer(csv_path, queue, append=resume and csv_path.is_file())
    )
    await asyncio.gather(*[_run_and_push(t) for t in tasks_to_run])
    await queue.put(_DONE)
    await writer_task
    elapsed = time.monotonic() - t_start

    print(f"\n[eval] Finished {len(tasks_to_run)} tasks in {elapsed:.1f}s "
          f"({elapsed / max(len(tasks_to_run), 1):.1f}s/task avg)")
    print(f"[eval] Results saved : {csv_path.resolve()}")

    stats = _llm_client.get_token_stats()
    stats["model"]           = _llm_client.DEFAULT_TEXT_MODEL
    stats["tasks_completed"] = len(tasks_to_run)
    stats["elapsed_s"]       = round(elapsed, 1)
    token_path = output_dir / "token_usage.json"
    token_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[eval] Token usage   : {stats['total_tokens']:,} total "
          f"({stats['prompt_tokens']:,} prompt + {stats['completion_tokens']:,} completion)")
    print(f"[eval] Token file    : {token_path.resolve()}")

    _print_summary(csv_path)
    return csv_path

# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ARC-AGI baseline evaluation — direct model prompting, no pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", type=str, default=None,
                   help="Model slug (e.g. openai/gpt-4o, anthropic/claude-sonnet-4-5, gemini-2.5-flash)")
    p.add_argument("--max-tokens", type=int, default=32000,
                   help="Global max output tokens cap per call (default: 32000)")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH,
                   help=f"Path to tasks JSON (default: {DEFAULT_DATA_PATH})")
    p.add_argument("--output-dir", type=Path, default=Path("arc_baseline_results/default"),
                   help="Directory for results CSV and token_usage.json")
    p.add_argument("--no-resume", action="store_true",
                   help="Re-run all tasks, ignoring any existing results")
    p.add_argument("--summary-only", action="store_true",
                   help="Print summary from existing CSV without running anything")
    p.add_argument("--concurrency", type=int, default=10,
                   help="Max concurrent LLM calls (default: 10)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    _llm_client.set_max_tokens_cap(args.max_tokens)
    print(f"[eval] Max tokens/call: {args.max_tokens}")

    if args.model:
        os.environ["KIVA_VISION_MODEL"] = args.model
        os.environ["KIVA_TEXT_MODEL"]   = args.model
        _llm_client.DEFAULT_VISION_MODEL = args.model
        _llm_client.DEFAULT_TEXT_MODEL   = args.model
        print(f"[eval] Model          : {args.model}")

    if args.summary_only:
        _print_summary(args.output_dir / "results.csv")
    else:
        asyncio.run(run_eval(
            data_path=args.data,
            output_dir=args.output_dir,
            resume=not args.no_resume,
            concurrency=args.concurrency,
        ))
