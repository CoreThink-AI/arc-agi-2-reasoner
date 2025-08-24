import os
import asyncio
from dotenv import load_dotenv
from arc_agi.src.patterns.pattern_detection_prompt import PROMPT
from arc_agi.src.patterns.detailed_hint_prompt import HINT_PROMPT_TEMPLATE
import json
from typing import List, Dict, Optional
from pydantic import BaseModel
from collections import Counter
from arc_agi.src.utils.visualization_utils import get_arr_viz
from arc_agi.src.utils.llm_utils import get_completion, summarize_reasons, get_completion_grok, get_completion_groq, get_completion_together
from arc_agi.src.patterns.object_comparison import compare_object_lists

load_dotenv()
# Make concurrency configurable to reduce OpenAI timeouts under load
CONCURRENT_REQUESTS = int(os.getenv("OPENAI_CONCURRENCY", "5"))
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
REPEAT_COUNT = int(os.getenv("PATTERN_DETECTION_REPETITIONS", "5"))
OPENAI_TASK_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TASK_TIMEOUT_SECONDS", "72000"))
SUMMARY_CONCURRENT_REQUESTS = int(os.getenv("OPENAI_SUMMARY_CONCURRENCY", "60"))


class PatternDetectionResult(BaseModel):
    reason: str
    pattern_detected: bool
    pattern_name: str
    pattern_description: str
    params: Optional[Dict[str, List[str]]] = None


class PatternDetectionResponse(BaseModel):
    result: List[PatternDetectionResult]


def format_params_for_prompt(params: Dict[str, List[str]]) -> str:
    return "\n".join([f"- **{key}**: {', '.join(values)}" for key, values in params.items()])


async def generate_pattern_hint(input_grid, output_grid, input_grid_viz, output_grid_viz, name, description, reason, params):
    formatted_params = format_params_for_prompt(params)
    prompt = HINT_PROMPT_TEMPLATE.format(
        pattern_name=name,
        description=description,
        reason=reason,
        params=formatted_params,
        input_grid_viz=input_grid_viz,
        output_grid_viz=output_grid_viz
    )
    try:
        response = await asyncio.wait_for(
            get_completion_groq(input_grid, output_grid, semaphore, PatternDetectionResponse, prompt),
            timeout=OPENAI_TASK_TIMEOUT_SECONDS,
        )
        if response and hasattr(response, 'result') and response.result:
            # Extract text from the structured response
            return str(response.result[0].pattern_description if response.result[0].pattern_description else "Hint unavailable.")
        else:
            return "Hint unavailable."
    except Exception as e:
        print(f"Hint generation failed for {name}: {e}")
        return "Hint unavailable."


async def unit_patterns(input_grid, output_grid, before_list: List, after_list: List):
    with open("arc_agi/src/patterns/unit_patterns.json", "r") as f:
        pattern_data = json.load(f)
        prompts = []
        input_grid_viz = get_arr_viz(input_grid)
        output_grid_viz = get_arr_viz(output_grid)
        comparison = compare_object_lists(before_list, after_list)

        add_json = json.dumps(comparison.added)
        remove_json = json.dumps(comparison.removed)
        retain_json = json.dumps(comparison.retained)
        prompts.append(PROMPT.format(
            input_grid_viz,
            output_grid_viz,
            add_json,
            remove_json,
            retain_json,
            pattern_data
        ))
        prompts = prompts * REPEAT_COUNT
        print(f"Processing {len(prompts)} patterns with {CONCURRENT_REQUESTS} concurrent requests...")
        tasks = [
            asyncio.wait_for(
                get_completion_groq(input_grid, output_grid, semaphore, PatternDetectionResponse, p),
                timeout=OPENAI_TASK_TIMEOUT_SECONDS,
            )
            for p in prompts
        ]
        print("preparing pattern results......")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        print("pattern results prepared...")

        counts = Counter()
        all_detected_patterns = []
        pattern_params = {}
        pattern_reasons = {}
        pattern_descriptions = {}

        failed_requests = 0
        for r in results:
            if isinstance(r, Exception):
                failed_requests += 1
                print(f"Pattern detection request failed: {r}")
                continue
            if r is not None and hasattr(r, 'result'):
                for pattern_result in r.result:
                    if pattern_result.pattern_detected:
                        pattern_name = pattern_result.pattern_name
                        counts[pattern_name] += 1
                        all_detected_patterns.append(pattern_result.model_dump())

                        if pattern_name not in pattern_reasons:
                            pattern_reasons[pattern_name] = []
                        pattern_reasons[pattern_name].append(pattern_result.reason)

                        if pattern_name not in pattern_descriptions:
                            pattern_descriptions[pattern_name] = pattern_result.pattern_description

                        if pattern_name not in pattern_params:
                            pattern_params[pattern_name] = {}

                        if pattern_result.params:
                            for param_key, param_values in pattern_result.params.items():
                                if param_key not in pattern_params[pattern_name]:
                                    pattern_params[pattern_name][param_key] = set()
                                if isinstance(param_values, list):
                                    pattern_params[pattern_name][param_key].update(param_values)
                                else:
                                    pattern_params[pattern_name][param_key].add(param_values)

        for pattern_name in pattern_params:
            for param_key in pattern_params[pattern_name]:
                pattern_params[pattern_name][param_key] = list(pattern_params[pattern_name][param_key])

        print("starting summarization...")
        # Summarize reasons for all detected patterns concurrently (bounded)
        summarized_reasons = {}
        if counts:
            summary_semaphore = asyncio.Semaphore(SUMMARY_CONCURRENT_REQUESTS)

            async def summarize_with_sem(reasons_list: List[str]):
                async with summary_semaphore:
                    return await summarize_reasons(reasons_list)

            pattern_names_for_summary = list(counts.keys())
            summary_tasks = [
                asyncio.create_task(summarize_with_sem(pattern_reasons.get(name, [])))
                for name in pattern_names_for_summary
            ]
            summary_results = await asyncio.gather(*summary_tasks, return_exceptions=True)
            for name, res in zip(pattern_names_for_summary, summary_results):
                summarized_reasons[name] = res if not isinstance(res, Exception) else ""

        print("finished summarizing...")
        print("starting hints generation...")
        # Generate detailed hints for all patterns concurrently (bounded by global semaphore inside LLM call)
        restructured_pattern_params = []
        if pattern_params:
            pattern_names_for_hints = list(pattern_params.keys())

            hint_tasks = [
                asyncio.create_task(
                    generate_pattern_hint(
                        input_grid,
                        output_grid,
                        input_grid_viz,
                        output_grid_viz,
                        name,
                        pattern_descriptions.get(name, ""),
                        summarized_reasons.get(name, ""),
                        pattern_params[name],
                    )
                )
                for name in pattern_names_for_hints
            ]

            hint_results = await asyncio.gather(*hint_tasks, return_exceptions=True)
            print("finished hints generation...")

            for name, hint_text in zip(pattern_names_for_hints, hint_results):
                pattern_description = pattern_descriptions.get(name, "")
                reason = summarized_reasons.get(name, "")
                params = pattern_params[name]
                detailed_hint = hint_text if not isinstance(hint_text, Exception) else "Hint unavailable."
                restructured_pattern_params.append({
                    'name': name,
                    'description': pattern_description,
                    'reason': reason,
                    'params': params,
                    'detailed_hint': detailed_hint
                })

        if failed_requests:
            print(f"Pattern detection requests failed: {failed_requests}/{len(results)}")

        return restructured_pattern_params, counts
