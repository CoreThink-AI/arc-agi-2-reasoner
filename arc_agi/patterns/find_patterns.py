import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from arc_agi.patterns.new_prompt import PROMPT
import os
import json
from typing import List, Dict, Optional
from pydantic import BaseModel,RootModel
import time
import random
from collections import Counter

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Semaphore to limit concurrent requests
CONCURRENT_REQUESTS = 10
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

class PatternDetectionResult(BaseModel):
    reason: str
    pattern_detected: bool
    pattern_name: str
    pattern_description: str
    params: Optional[Dict[str, List[str]]] = None

class PatternDetectionResponse(BaseModel):
    result: List[PatternDetectionResult]

def get_arr_viz(arr):
    viz = ""
    for row in arr:
        viz += " | ".join(str(i) for i in row) + "\n"
        
    return viz.strip()

async def get_completion_with_retry(prompt: str, max_retries: int = 3):
    """Get completion with retry logic and rate limiting using OpenAI"""
    async with semaphore:  # Limit concurrent requests
        for attempt in range(max_retries):
            try:
                # Add small delay to avoid hitting rate limits
                await asyncio.sleep(random.uniform(0.2, 0.8))
                
                response = await client.beta.chat.completions.parse(
                    model="o4-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format=PatternDetectionResponse,
                )
                print(len(prompt))
                result = response.choices[0].message.parsed
                return result
                # Only return results where pattern was detected
                #if result.pattern_detected:
                #    return result
                #else:
                #    return None  # Pattern not detected
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts: {e}")
                    return None  # Return None when all retries fail
                else:
                    print(f"Attempt {attempt + 1} failed: {e}, retrying...")
                    await asyncio.sleep(random.uniform(1.0, 2.0))  # Longer delay before retry

async def get_completion(prompt: str):
    return await get_completion_with_retry(prompt)

async def summarize_reasons(reasons_list: List[str]) -> str:
    """Summarize multiple reasons using GPT-4.1"""
    if not reasons_list:
        return ""
    
    if len(reasons_list) == 1:
        return reasons_list[0]
    
    combined_reasons = "\n\n".join([f"Reason {i+1}: {reason}" for i, reason in enumerate(reasons_list)])
    
    prompt = f"""You are given multiple explanations for why a specific pattern was detected in a transformation. Please provide a concise, unified summary that captures the key insights from all explanations.

Multiple Explanations:
{combined_reasons}

Please provide a clear, concise summary that combines the key points from all explanations above:"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4.1",  # Using gpt-4o-mini as it's more available than gpt-4.1
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error summarizing reasons: {e}")
        return combined_reasons  # Fallback to concatenated reasons

async def unit_patterns(input_grid, output_grid, before_list: List, after_list: List):
    with open("arc_agi/patterns/unit_patterns.json", "r") as f:
        pattern_data = json.load(f)
        prompts = []
        input_grid_viz = get_arr_viz(input_grid)
        output_grid_viz = get_arr_viz(output_grid)
        
        #for pattern in pattern_data:
        #pattern_specs = json.dumps(pattern_data[pattern])
        before_list_json = json.dumps(before_list)
        after_list_json = json.dumps(after_list)
        prompts.append(PROMPT.format(
            input_grid_viz, 
            output_grid_viz, 
            before_list_json, 
            after_list_json, 
            pattern_data
        ))
        prompts = prompts*10
        print(f"Processing {len(prompts)} patterns with {CONCURRENT_REQUESTS} concurrent requests...")
        tasks = [get_completion(p) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        counts = Counter()
        all_detected_patterns = []
        pattern_params = {}  # Will store unique params for each pattern name
        pattern_reasons = {}  # Will store reasons for each pattern name
        pattern_descriptions = {}  # Will store descriptions for each pattern name
        
        for r in results:
            if isinstance(r, Exception):
                print(f"Request failed: {r}")
                continue
                
            if r is not None and hasattr(r, 'result'):
                # r.result is a list of PatternDetectionResult objects
                for pattern_result in r.result:
                    if pattern_result.pattern_detected:  # Only count detected patterns
                        pattern_name = pattern_result.pattern_name
                        counts[pattern_name] += 1
                        all_detected_patterns.append(pattern_result.model_dump())
                        
                        # Collect reasons for summarization
                        if pattern_name not in pattern_reasons:
                            pattern_reasons[pattern_name] = []
                        pattern_reasons[pattern_name].append(pattern_result.reason)
                        
                        # Store description (should be same for all instances of this pattern)
                        if pattern_name not in pattern_descriptions:
                            pattern_descriptions[pattern_name] = pattern_result.pattern_description
                        
                        # Collect unique params for this pattern
                        if pattern_name not in pattern_params:
                            pattern_params[pattern_name] = {}
                        
                        if pattern_result.params:
                            for param_key, param_values in pattern_result.params.items():
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
        
        return restructured_pattern_params,counts
