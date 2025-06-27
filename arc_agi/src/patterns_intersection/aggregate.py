from collections import Counter
from arc_agi.src.utils.llm_utils import summarize_reasons, get_completion
from arc_agi.src.utils.visualization_utils import get_arr_viz
from arc_agi.src.patterns.pattern_hint_prompt import HINT_PROMPT_TEMPLATE
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


async def generate_pattern_hint(input_grid, output_grid, input_grid_viz, output_grid_viz, name, description, reason, params):
    params_formatted = format_params_for_prompt(params)

    prompt = HINT_PROMPT_TEMPLATE.format(
        pattern_name=name,
        description=description,
        reason=reason,
        input_grid_viz=input_grid_viz,
        output_grid_viz=output_grid_viz,
        params=params_formatted  # ← now passed into the prompt
    )

    try:
        response = await get_completion(input_grid, output_grid, semaphore, str, prompt)
        return response.strip()
    except Exception as e:
        print(f"Hint generation failed for {name}: {e}")
        return "Hint unavailable."


async def intersect(results, input_grid, output_grid):
    counts = Counter()
    pattern_params = {}
    pattern_reasons = {}
    pattern_descriptions = {}

    for pattern_result in results:
        if isinstance(pattern_result, Exception):
            print(f"Request failed: {pattern_result}")
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

    # Visualizations for hint generation
    input_grid_viz = get_arr_viz(input_grid)
    output_grid_viz = get_arr_viz(output_grid)

    # Final restructure with hints
    restructured_pattern_params = []
    for pattern_name in pattern_params:
        pattern_description = pattern_descriptions.get(pattern_name, "")
        reason = summarized_reasons.get(pattern_name, "")
        params = pattern_params[pattern_name]

        detailed_hint = await generate_pattern_hint(
            input_grid, output_grid,
            input_grid_viz, output_grid_viz,
            pattern_name, pattern_description, reason, params
        )

        restructured_pattern_params.append({
            'name': pattern_name,
            'description': pattern_description,
            'reason': reason,
            'params': params,
            'detailed_hint': detailed_hint
        })

    return restructured_pattern_params, counts
