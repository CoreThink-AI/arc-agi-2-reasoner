"""
ARC-AGI Parallel Orchestrator for Test Challenges

Reads tasks from `arc-agi_test_challenges.json` and runs a staged pipeline:
- Stage A: Per-train example pattern detection (async client calls)
- Stage B: Aggregate top patterns into final hint (async client calls)
- Stage C: Solve test inputs with multi-attempt async client calls and majority vote

Each stage persists its outputs under `logs_parallel/<task_id>/` so later
stages can reuse cached results without recomputation.

Parallelism is achieved by leveraging async APIs of the clients (AsyncOpenAI),
not via thread pools.
"""

import os
import json
import time
import asyncio
from typing import Any, Dict, List, Tuple, Optional

from arc_agi.src.objects.base import Grid, BaseObject
from arc_agi.src.patterns.find_patterns import unit_patterns
from arc_agi.src.patterns_intersection.aggregate import intersect
from arc_agi.src.utils.visualization_utils import get_arr_viz
from arc_agi.src.utils.llm_utils import openai_client
from arc_agi.src.solver.solver_prompts import new_solver_prompt, critic_prompt
from arc_agi.src.solver.solver import extract_matrix_from_response, matrix_to_arr


# ----------------------------
# Filesystem helpers (caching)
# ----------------------------

def ensure_task_dir(task_id: str) -> str:
    base = os.path.join("logs_parallel", task_id)
    os.makedirs(base, exist_ok=True)
    return base


def read_json_if_exists(path: str) -> Optional[Any]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ----------------------------
# Stage A: Pattern detection
# ----------------------------

async def create_base_object_dict(grid: List[List[int]], coords: List[Tuple[int, int]]) -> Dict[str, Any]:
    """Create BaseObject dict. This uses BaseObject internals which may make sync LLM calls.
    It is intentionally kept in a thread to avoid blocking the event loop.
    """
    def _sync_to_dict() -> Dict[str, Any]:
        data = asyncio.run(
            BaseObject(grid, coords).to_dict(
                provider="openai",
                model="gpt-4.1-mini",
                temperature=0.0,
                max_tokens=4096,
            )
        )
        # The object dictionaries passed to pattern comparison don't need the full grid
        if "grid" in data:
            del data["grid"]
        return data

    return await asyncio.to_thread(_sync_to_dict)


async def process_single_training_example_for_patterns(task_dir: str, idx: int, train_example: Dict[str, Any]):
    """Runs object extraction and pattern detection for one train example. Caches results."""
    patt_path = os.path.join(task_dir, f"train_{idx:02d}_patterns.json")
    counts_path = os.path.join(task_dir, f"train_{idx:02d}_counts.json")

    cached_patterns = read_json_if_exists(patt_path)
    cached_counts = read_json_if_exists(counts_path)
    if cached_patterns is not None and cached_counts is not None:
        return cached_patterns, cached_counts

    grid_input = train_example["input"]
    grid_output = train_example["output"]

    # Object discovery (rule-based + minimal LLM calls inside BaseObject)
    grid_a = Grid(grid_input)
    input_obj = grid_a.find_objects_in_grid("openai", "gpt-4.1-mini", 0.0, 4096)
    grid_b = Grid(grid_output)
    output_obj = grid_b.find_objects_in_grid("openai", "gpt-4.1-mini", 0.0, 4096)

    # Convert objects to BaseObject dicts concurrently
    input_tasks = [create_base_object_dict(grid_input, obj) for obj in input_obj]
    output_tasks = [create_base_object_dict(grid_output, obj) for obj in output_obj]
    before_list, after_list = await asyncio.gather(
        asyncio.gather(*input_tasks), asyncio.gather(*output_tasks)
    )

    # Pattern detection via AsyncOpenAI
    pattern_params, counts = await unit_patterns(grid_input, grid_output, before_list, after_list)

    write_json(patt_path, pattern_params)
    write_json(counts_path, counts)
    return pattern_params, counts


async def stage_a_detect_patterns(task_id: str, arc_task: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Run Stage A across all training examples for a task. Returns combined patterns and total counts."""
    task_dir = ensure_task_dir(task_id)
    train = arc_task.get("train", [])

    tasks = [process_single_training_example_for_patterns(task_dir, i, t) for i, t in enumerate(train)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_patterns: List[Dict[str, Any]] = []
    total_counts: Dict[str, int] = {}
    for res in results:
        if isinstance(res, Exception):
            continue
        pattern_params, counts = res
        all_patterns.extend(pattern_params)
        for k, v in counts.items():
            total_counts[k] = total_counts.get(k, 0) + int(v)

    # Persist combined outputs
    write_json(os.path.join(task_dir, "stage_a_patterns_all.json"), all_patterns)
    write_json(os.path.join(task_dir, "stage_a_counts_total.json"), total_counts)
    return all_patterns, total_counts


# --------------------------------------------
# Stage B: Aggregate patterns into final hints
# --------------------------------------------

async def stage_b_aggregate_hints(task_id: str, all_patterns: List[Dict[str, Any]], total_counts: Dict[str, int]) -> str:
    task_dir = ensure_task_dir(task_id)
    hint_path = os.path.join(task_dir, "stage_b_hint.json")
    cached = read_json_if_exists(hint_path)
    if cached is not None and isinstance(cached, str):
        return cached

    # Top-N patterns by counts
    top_n = 3
    top_patterns = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_names = {name for name, _ in top_patterns}
    filtered = [p for p in all_patterns if p.get("name") in top_names]

    restructured_pattern_params, _ = await intersect(filtered)
    hint = json.dumps(restructured_pattern_params)

    write_json(hint_path, hint)
    return hint


# ------------------------------------------------------
# Stage C: Solve tests via async client calls and voting
# ------------------------------------------------------

def build_examples_str(train_examples: List[Dict[str, Any]]) -> str:
    from arc_agi.src.solver.solver_prompts import example_template
    lines: List[str] = []
    for i, ex in enumerate(train_examples, start=1):
        lines.append(
            example_template.format(
                i=i,
                input_viz=get_arr_viz(ex["input"]),
                output_viz=get_arr_viz(ex.get("output", [])),
            )
        )
    return "\n".join(lines).strip()


def build_test_prompts(arc_task: Dict[str, Any], hint: str) -> Tuple[List[str], List[List[List[int]]]]:
    examples = build_examples_str(arc_task.get("train", []))
    test_viz = [get_arr_viz(entry["input"]) for entry in arc_task.get("test", [])]
    prompts = [new_solver_prompt.format(examples=examples, test_input_viz=viz, hint=hint) for viz in test_viz]
    ground_truths = [entry.get("output", []) for entry in arc_task.get("test", [])]
    return prompts, ground_truths


async def async_llm_execute(prompt: str) -> str:
    resp = await openai_client.chat.completions.create(
        model="o4-mini",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=2000,
    )
    return resp.choices[0].message.content or ""


async def solve_single_test_case(prompt: str, num_attempts: int, max_concurrent_attempts: int = 5) -> List[List[List[int]]]:
    sem = asyncio.Semaphore(max_concurrent_attempts)

    async def _attempt_once() -> List[List[int]]:
        async with sem:
            raw = await async_llm_execute(prompt)
            extracted = extract_matrix_from_response(raw)
            return matrix_to_arr(extracted)

    tasks = [_attempt_once() for _ in range(num_attempts)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    parsed: List[List[List[int]]] = []
    for r in results:
        if isinstance(r, Exception):
            continue
        if r is not None:
            parsed.append(r)
    return parsed


def majority_vote_grid(grids: List[List[List[int]]]) -> Optional[List[List[int]]]:
    if not grids:
        return None
    h, w = len(grids[0]), len(grids[0][0])
    same_size = [g for g in grids if len(g) == h and all(len(row) == w for row in g)]
    if not same_size:
        return grids[0]
    from collections import Counter
    final: List[List[int]] = []
    for r in range(h):
        row: List[int] = []
        for c in range(w):
            vals = [g[r][c] for g in same_size]
            row.append(Counter(vals).most_common(1)[0][0])
        final.append(row)
    return final


async def stage_c_solve_tests(task_id: str, arc_task: Dict[str, Any], hint: str, num_attempts: int = 3) -> Tuple[List[Optional[List[List[int]]]], List[List[List[int]]]]:
    task_dir = ensure_task_dir(task_id)
    prompts_path = os.path.join(task_dir, "stage_c_prompts.json")
    raw_path = os.path.join(task_dir, "stage_c_raw_attempts.json")
    final_path = os.path.join(task_dir, "stage_c_consensus.json")

    prompts, ground_truths = build_test_prompts(arc_task, hint)
    write_json(prompts_path, prompts)

    # Solve each test case with multiple attempts using async client calls
    test_tasks = [solve_single_test_case(p, num_attempts) for p in prompts]
    all_attempts = await asyncio.gather(*test_tasks, return_exceptions=True)

    # Persist raw attempts
    attempts_serializable: List[List[List[List[int]]]] = []
    for tr in all_attempts:
        if isinstance(tr, Exception):
            attempts_serializable.append([])
        else:
            attempts_serializable.append(tr)
    write_json(raw_path, attempts_serializable)

    # Majority vote per test
    consensus: List[Optional[List[List[int]]]] = []
    for tr in all_attempts:
        if isinstance(tr, Exception) or not tr:
            consensus.append(None)
        else:
            consensus.append(majority_vote_grid(tr))

    write_json(final_path, consensus)
    return consensus, ground_truths


# ---------------------------------
# Task Orchestration for challenges
# ---------------------------------

async def process_single_task(task_id: str, arc_task: Dict[str, Any], num_attempts: int = 3) -> Dict[str, Any]:
    t0 = time.time()
    task_dir = ensure_task_dir(task_id)

    # Stage A
    patterns_all, counts_total = await stage_a_detect_patterns(task_id, arc_task)

    # Stage B
    hint = await stage_b_aggregate_hints(task_id, patterns_all, counts_total)

    # Stage C (solve)
    consensus, ground_truth = await stage_c_solve_tests(task_id, arc_task, hint, num_attempts=num_attempts)

    result = {
        "task_id": task_id,
        "hint": hint,
        "consensus": consensus,
        "ground_truth": ground_truth,
        "elapsed_sec": time.time() - t0,
    }
    write_json(os.path.join(task_dir, "summary.json"), result)
    return result


async def run_all_from_test_challenges(json_path: str, limit: Optional[int] = None, num_attempts: int = 3) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        all_tasks: Dict[str, Dict[str, Any]] = json.load(f)

    items = list(all_tasks.items())
    if limit is not None:
        items = items[:limit]

    results: List[Dict[str, Any]] = []
    # Process tasks sequentially to keep memory usage bounded; within each task
    # all heavy calls are parallelized via async clients
    for task_id, arc_task in items:
        res = await process_single_task(task_id, arc_task, num_attempts=num_attempts)
        results.append(res)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ARC-AGI test challenges in stages with async client parallelism")
    parser.add_argument("--path", default="arc-agi_test_challenges.json", help="Path to test challenges JSON map")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--attempts", type=int, default=3, help="Number of attempts per test case")

    args = parser.parse_args()

    asyncio.run(run_all_from_test_challenges(args.path, limit=args.limit, num_attempts=args.attempts))


