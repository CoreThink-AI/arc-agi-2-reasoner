import os
import asyncio
import random
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from arc_agi.src.patterns.detailed_hint_prompt import HINT_SUMMARY_PROMPT
from .visualization_utils import array_to_base64_image

load_dotenv()

# Initialize Grok client
grok_client = OpenAI(
    api_key="",
    base_url="https://api.x.ai/v1",
    timeout=7200
)

# =====================================
# Unified LLM call function (Grok only)
# =====================================

def call_llm(provider: str, prompt: str, model: str = None, temperature: float = 0.0, max_tokens: int = 4096) -> str:
    """
    Unified function but now Grok-only.
    Ignores 'provider' and always calls Grok.
    """
    sys_prompt = "You are an expert at solving grid-based reasoning problems. Use markdown output. Enclose code or grids in ```."

    try:
        response = grok_client.chat.completions.create(
            model="grok-4",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {e}"

# =====================================
# Replace get_anthropic_response
# =====================================

def get_anthropic_response(prompt):
    """
    Replaced with Grok.
    """
    return call_llm("grok", prompt)

# =====================================
# Replace get_anthropic_response_stream
# =====================================

def get_anthropic_response_stream(prompt):
    """
    Stream Grok response.
    """
    final_response = ""
    try:
        stream = grok_client.chat.completions.create(
            model="grok-4",
            messages=[
                {"role": "system", "content": "You are an expert reasoner."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                final_response += chunk.choices[0].delta.content
        return final_response
    except Exception as e:
        return f"Error: {e}"

# =====================================
# Replace get_cerebras_response
# =====================================

def get_cerebras_response(prompt: str) -> str:
    """
    Now uses Grok instead of Cerebras.
    """
    return call_llm("grok", prompt)

# =====================================
# Pattern Detection with retry (Grok)
# =====================================

async def get_completion_with_retry(img1, img2, semaphore, PatternDetectionResponse, prompt: str, max_retries: int = 3):
    """
    Pattern detection using Grok (multimodal call).
    """
    async with semaphore:
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(random.uniform(0.2, 0.8))

                response = grok_client.chat.completions.create(
                    model="grok-4",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2}"}}
                        ]
                    }],
                    response_format=PatternDetectionResponse
                )

                return response.choices[0].message.parsed

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts: {e}")
                    return None
                else:
                    print(f"Attempt {attempt + 1} failed: {e}, retrying...")
                    await asyncio.sleep(random.uniform(1.0, 2.0))

async def get_completion(grid1, grid2, semaphore, PatternDetectionResponse, prompt: str):
    img1 = array_to_base64_image(grid1)
    img2 = array_to_base64_image(grid2)
    return await get_completion_with_retry(img1, img2, semaphore, PatternDetectionResponse, prompt)

# =====================================
# Summarization (Grok)
# =====================================

async def summarize_reasons(reasons_list: List[str]) -> str:
    if not reasons_list:
        return ""
    if len(reasons_list) == 1:
        return reasons_list[0]

    combined_reasons = "\n\n".join([f"Reason {i+1}: {reason}" for i, reason in enumerate(reasons_list)])

    prompt = f"""
You are given multiple explanations for why a pattern was detected.
Provide a detailed, unified summary that captures the key insights from all explanations.

Explanations:
{combined_reasons}

Provide a step-by-step explanation of how the input would transform into the output.
Use markdown.
"""

    return call_llm("grok", prompt)

async def summarize_hints(hint_list: List[str]) -> str:
    if not hint_list:
        return ""
    if len(hint_list) == 1:
        return hint_list[0]

    combined_hints = "\n\n".join([f"Hint {i+1}: {hint}" for i, hint in enumerate(hint_list)])

    prompt = HINT_SUMMARY_PROMPT.format(combined_hints)

    return call_llm("grok", prompt)

# =====================================
# Grid Parsing
# =====================================

class GridModel(BaseModel):
    grid: List[List[int]]

def parse_grid(response_text: str):
    prompt = f"""
You are given a description that includes a 2D grid.
Extract the 2D integer grid and return it as a Python list of lists.
Only output the grid enclosed in triple backticks.

Input:
{response_text}
"""
    raw_output = call_llm("grok", prompt)

    try:
        grid_text = raw_output.split("```")[1]
        grid = eval(grid_text)
        return grid
    except Exception as e:
        print(f"Failed to parse grid: {e}")
        return None
