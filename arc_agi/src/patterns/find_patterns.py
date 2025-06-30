import asyncio
from dotenv import load_dotenv
from arc_agi.src.patterns.pattern_detection_prompt import PROMPT
from arc_agi.src.patterns.detailed_hint_prompt import HINT_PROMPT_TEMPLATE
import json
from typing import List, Dict, Optional
from pydantic import BaseModel
from collections import Counter
from arc_agi.src.utils.visualization_utils import get_arr_viz
from arc_agi.src.utils.llm_utils import get_completion, summarize_reasons
from arc_agi.src.patterns.object_comparison import compare_object_lists

load_dotenv()
CONCURRENT_REQUESTS = 5
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)


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
        response = await get_completion(input_grid, output_grid, semaphore, PatternDetectionResponse, prompt)
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
        prompts = prompts*10
        print(f"Processing {len(prompts)} patterns with {CONCURRENT_REQUESTS} concurrent requests...")
        tasks = [get_completion(input_grid, output_grid, semaphore, PatternDetectionResponse, p) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        counts = Counter()
        all_detected_patterns = []
        pattern_params = {}
        pattern_reasons = {}
        pattern_descriptions = {}

        for r in results:
            if isinstance(r, Exception):
                print(f"Request failed: {r}")
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

        summarized_reasons = {}
        for pattern_name in counts:
            reasons = pattern_reasons.get(pattern_name, [])
            summarized_reasons[pattern_name] = await summarize_reasons(reasons)

        restructured_pattern_params = []
        for pattern_name in pattern_params:
            pattern_description = pattern_descriptions.get(pattern_name, "")
            reason = summarized_reasons.get(pattern_name, "")
            params = pattern_params[pattern_name]
            detailed_hint = await generate_pattern_hint(input_grid, output_grid, input_grid_viz, output_grid_viz, pattern_name, pattern_description, reason, params)
            restructured_pattern_params.append({
                'name': pattern_name,
                'description': pattern_description,
                'reason': reason,
                'params': params,
                'detailed_hint': detailed_hint
            })

        return restructured_pattern_params, counts
