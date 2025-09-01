from collections import Counter
from arc_agi.src.utils.llm_utils import summarize_reasons, summarize_hints
from arc_agi.src.utils.visualization_utils import get_arr_viz
import asyncio

CONCURRENT_REQUESTS = 5
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)


def format_params_for_prompt(params):
    """
    Convert params dictionary to markdown-style bullet list.
    """
    if not params:
        return "No parameters detected."
    lines = []
    for key, values in params.items():
        values_list = ', '.join(values) if isinstance(values, list) else str(values)
        lines.append(f"- **{key}**: {values_list}")
    return "\n".join(lines)

async def intersect(results):
    counts = Counter()
    pattern_params = {}
    pattern_reasons = {}
    pattern_descriptions = {}
    pattern_hints = {}

    failed_results = 0
    for pattern_result in results:
        if isinstance(pattern_result, Exception):
            failed_results += 1
            continue

        pattern_name = pattern_result.get('name')
        if not pattern_name:
            continue

        counts[pattern_name] += 1

        if pattern_name not in pattern_reasons:
            pattern_reasons[pattern_name] = []
        if pattern_result.get('reason'):
            pattern_reasons[pattern_name].append(pattern_result['reason'])

        if pattern_name not in pattern_descriptions:
            pattern_descriptions[pattern_name] = pattern_result.get('description', '')

        if pattern_name not in pattern_hints:
            pattern_hints[pattern_name] = []
        if pattern_result.get('detailed_hint'):
            pattern_hints[pattern_name].append(pattern_result['detailed_hint'])

        if pattern_name not in pattern_params:
            pattern_params[pattern_name] = {}

        if pattern_result.get('params'):
            for param_key, param_values in pattern_result['params'].items():
                if param_key not in pattern_params[pattern_name]:
                    pattern_params[pattern_name][param_key] = set()
                if isinstance(param_values, list):
                    pattern_params[pattern_name][param_key].update(param_values)
                else:
                    pattern_params[pattern_name][param_key].add(param_values)

    # Convert sets to lists
    for pattern_name in pattern_params:
        for param_key in pattern_params[pattern_name]:
            pattern_params[pattern_name][param_key] = list(pattern_params[pattern_name][param_key])

    # Summarize reasons
    summarized_reasons = {}
    for pattern_name in counts:
        reasons = pattern_reasons.get(pattern_name, [])
        summarized_reasons[pattern_name] = await summarize_reasons(reasons)

    summarized_hints = {}
    for pattern_name in counts:
        hints = pattern_hints.get(pattern_name, [])
        summarized_hints[pattern_name] = await summarize_hints(hints)

    # Final restructure with hints
    restructured_pattern_params = []
    for pattern_name in counts:
        pattern_description = pattern_descriptions[pattern_name]
        reason = summarized_reasons[pattern_name]
        params = pattern_params[pattern_name]
        detailed_hint = summarized_hints[pattern_name]
        restructured_pattern_params.append({
            'name': pattern_name,
            'description': pattern_description,
            'reason': reason,
            'params': params,
            'detailed_hint': detailed_hint
        })

    if failed_results:
        print(f"Aggregation inputs failed: {failed_results}/{len(results)}")

    return restructured_pattern_params, counts
