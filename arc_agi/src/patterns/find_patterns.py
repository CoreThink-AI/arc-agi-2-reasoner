import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from arc_agi.src.patterns.pattern_detection_prompt import PROMPT
from arc_agi.src.objects.llm_inference import call_llm
import os
import json
from typing import List
import time
import random
load_dotenv()
client = AsyncOpenAI()

# Semaphore to limit concurrent requests
CONCURRENT_REQUESTS = 3
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

def get_arr_viz(arr):
    viz = ""
    for row in arr:
        viz += " | ".join(str(i) for i in row) + "\n"
        
    return viz.strip()

async def get_completion_openai(prompt):
    resp = await client.chat.completions.create(
        model="o4-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

def get_completion_sync(prompt: str):
    """Synchronous LLM call using the existing llm_inference module"""
    return call_llm(
        provider="anthropic",
        prompt=prompt,
        model="claude-3-opus-20240229",
        temperature=0.0,
        max_tokens=1024
    )

async def get_completion_with_retry(prompt: str, max_retries: int = 3):
    """Get completion with retry logic and rate limiting using asyncio.to_thread"""
    async with semaphore:  # Limit concurrent requests
        for attempt in range(max_retries):
            try:
                # Add small delay to avoid hitting rate limits
                await asyncio.sleep(random.uniform(0.5, 1.5))
                
                # Use asyncio.to_thread to run the sync function in thread pool
                result = await asyncio.to_thread(get_completion_sync, prompt)
                return result
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts: {e}")
                    raise e
                else:
                    wait_time = (2 ** attempt) + random.uniform(1, 3)
                    print(f"Attempt {attempt + 1} failed, retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)

async def get_completion(prompt: str):
    return await get_completion_with_retry(prompt)

async def unit_patterns(input_grid,output_grid,before_list:List, after_list:List):
    with open("arc_agi/patterns/unit_patterns.json", "r") as f:
        pattern_data = json.load(f)
        prompts = []
        input_grid = get_arr_viz(input_grid)
        output_grid = get_arr_viz(output_grid)
        for pattern in pattern_data:
            pattern_specs = json.dumps(pattern_data[pattern])
            before_list_json = json.dumps(before_list)
            after_list_json = json.dumps(after_list)
            prompts.append(PROMPT.format(input_grid,output_grid,before_list_json,after_list_json,pattern,pattern_specs))
        
        print(f"Processing {len(prompts)} patterns with {CONCURRENT_REQUESTS} concurrent requests...")
        tasks = [get_completion(p) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Pattern {i} failed: {result}")
                final_results.append(f"Error: {result}")
            else:
                final_results.append(result)
        
        return final_results

#if __name__ == "__main__":
#    results = asyncio.run(unit_patterns())   
#    print(results)
