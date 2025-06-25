from collections import Counter
from arc_agi.src.utils.llm_utils import summarize_reasons

async def intersect(results):
    """
    Process an array of pattern results to aggregate similar patterns.
    results: array of dicts with keys: name, description, reason, params
    """
    counts = Counter()
    pattern_params = {}  # Will store unique params for each pattern name
    pattern_reasons = {}  # Will store reasons for each pattern name
    pattern_descriptions = {}  # Will store descriptions for each pattern name
    
    for pattern_result in results:
        if isinstance(pattern_result, Exception):
            print(f"Request failed: {pattern_result}")
            continue
            
        pattern_name = pattern_result.get('name')
        if not pattern_name:
            continue
            
        counts[pattern_name] += 1
        
        # Collect reasons for summarization
        if pattern_name not in pattern_reasons:
            pattern_reasons[pattern_name] = []
        if pattern_result.get('reason'):
            pattern_reasons[pattern_name].append(pattern_result['reason'])
        
        # Store description (should be same for all instances of this pattern)
        if pattern_name not in pattern_descriptions:
            pattern_descriptions[pattern_name] = pattern_result.get('description', '')
        
        # Collect unique params for this pattern
        if pattern_name not in pattern_params:
            pattern_params[pattern_name] = {}
        
        if pattern_result.get('params'):
            for param_key, param_values in pattern_result['params'].items():
                if param_key not in pattern_params[pattern_name]:
                    pattern_params[pattern_name][param_key] = set()
                # Add all values to the set for this parameter
                if isinstance(param_values, list):
                    pattern_params[pattern_name][param_key].update(param_values)
                else:
                    pattern_params[pattern_name][param_key].add(param_values)
    
    # Convert sets to lists for JSON serialization if needed
    for pattern_name in pattern_params:
        for param_key in pattern_params[pattern_name]:
            pattern_params[pattern_name][param_key] = list(pattern_params[pattern_name][param_key])
    
    # Summarize reasons for each pattern
    summarized_reasons = {}
    for pattern_name in counts:
        reasons = pattern_reasons.get(pattern_name, [])
        summarized_reasons[pattern_name] = await summarize_reasons(reasons)
    
    # Restructure pattern_params to include name, description, reason, and params
    restructured_pattern_params = []
    for pattern_name in pattern_params:
        pattern_description = pattern_descriptions.get(pattern_name, "")
        reason = summarized_reasons.get(pattern_name, "")
        params = pattern_params[pattern_name]
        
        restructured_pattern_params.append({
            'name': pattern_name,
            'description': pattern_description,
            'reason': reason,
            'params': params
        })
    
    return restructured_pattern_params, counts
