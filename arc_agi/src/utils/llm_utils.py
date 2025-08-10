import os
import asyncio
import random
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from arc_agi.src.patterns.detailed_hint_prompt import HINT_SUMMARY_PROMPT
from .visualization_utils import array_to_base64_image
import openai
from openai import OpenAI, AsyncOpenAI
import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
)

load_dotenv()
deployment = "o4-mini"
# Initialize Grok client
grok_client = OpenAI(
    api_key=os.environ.get("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
    timeout=7200,
)

# =====================================
# Unified LLM call function (Grok only)
# =====================================

# Configure Async OpenAI client with explicit timeout and retries
DEFAULT_OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "7200"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))

_httpx_async_client = httpx.AsyncClient(timeout=DEFAULT_OPENAI_TIMEOUT_SECONDS)
openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=DEFAULT_OPENAI_TIMEOUT_SECONDS,
    max_retries=OPENAI_MAX_RETRIES,
    http_client=_httpx_async_client,
)

def call_llm(provider: str, prompt: str, model: str = None, temperature: float = 0.0, max_tokens: int = 4096) -> str:
    """
    Unified function but now Grok-only.
    Ignores 'provider' and always calls Grok.
    """
    sys_prompt = "You are an expert at solving grid-based reasoning problems. Use markdown output. Enclose code or grids in ```."

    provider = provider.lower()
    if provider == "openai":
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            #temperature=temperature,
            #max_tokens=max_tokens
        )
        return response.choices[0].message.content

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

def get_grok_response_stream(prompt):
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
# Pattern Detection with retry (OpenAI async)
# =====================================

async def get_completion_with_retry(
    img1,
    img2,
    semaphore,
    PatternDetectionResponse,
    prompt: str,
    max_retries: int = None,
):
    """Get completion with retry logic, timeouts, and rate limiting using OpenAI."""
    retry_limit = max_retries or OPENAI_MAX_RETRIES
    async with semaphore:  # Limit concurrent requests
        for attempt in range(retry_limit):
            try:
                # Small jitter to avoid herd behavior
                await asyncio.sleep(random.uniform(0.2, 0.8))

                response = await openai_client.beta.chat.completions.parse(
                    model=deployment,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{img1}"},
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{img2}"},
                                },
                            ],
                        }
                    ],
                    response_format=PatternDetectionResponse,
                    timeout=DEFAULT_OPENAI_TIMEOUT_SECONDS,
                )
                result = response.choices[0].message.parsed
                return result

            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                is_last = attempt == retry_limit - 1
                if is_last:
                    print(f"[OpenAI timeout/connection/rate-limit] Giving up after {retry_limit} attempts: {e}")
                    return None
                backoff_seconds = min(2 ** attempt + random.uniform(0, 0.5), 8.0)
                print(f"[OpenAI retryable error] attempt {attempt + 1}/{retry_limit} failed: {e} — backing off {backoff_seconds:.2f}s")
                await asyncio.sleep(backoff_seconds)
            except Exception as e:
                is_last = attempt == retry_limit - 1
                if is_last:
                    print(f"[OpenAI fatal error] Failed after {retry_limit} attempts: {e}")
                    return None
                backoff_seconds = min(1.5 ** (attempt + 1) + random.uniform(0, 0.25), 6.0)
                print(f"[OpenAI error] attempt {attempt + 1}/{retry_limit} failed: {e} — retrying in {backoff_seconds:.2f}s")
                await asyncio.sleep(backoff_seconds)

async def get_completion(grid1, grid2, semaphore, PatternDetectionResponse, prompt: str):
    img1 = array_to_base64_image(grid1)
    img2 = array_to_base64_image(grid2)
    return await get_completion_with_retry(img1, img2, semaphore, PatternDetectionResponse, prompt)

async def summarize_reasons(reasons_list: List[str]) -> str:
    """Summarize multiple reasons using GPT-4.1"""
    if not reasons_list:
        return ""
    
    if len(reasons_list) == 1:
        return reasons_list[0]
    
    combined_reasons = "\n\n".join([f"Reason {i+1}: {reason}" for i, reason in enumerate(reasons_list)])
    
    prompt = f"""You are given multiple explanations for why a specific pattern was detected in a transformation. Please provide a detailed, unified summary that captures the key insights from all explanations.

    Multiple Explanations:
    {combined_reasons}

    Please provide a step by step account of how the input image would go through the transformation to reach the output image:
    Give output in markdown syntax."""

    try:
        response = await openai_client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
            timeout=DEFAULT_OPENAI_TIMEOUT_SECONDS,
        )
        return response.choices[0].message.content.strip()
    except (APITimeoutError, APIConnectionError, RateLimitError) as e:
        print(f"OpenAI summarize_reasons timeout/connection/rate-limit: {e}")
        return combined_reasons
    except Exception as e:
        print(f"Error summarizing reasons: {e}")
        return combined_reasons 

async def summarize_hints(hint_list: List[str]) -> str:
    """Summarize multiple reasons using GPT-4.1"""
    if not hint_list:
        return ""
    
    if len(hint_list) == 1:
        return hint_list[0]
    
    combined_hints = "\n\n".join([f"Reason {i+1}: {reason}" for i, reason in enumerate(hint_list)])
    
    prompt = HINT_SUMMARY_PROMPT.format(combined_hints)

    try:
        response = await openai_client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
            timeout=DEFAULT_OPENAI_TIMEOUT_SECONDS,
        )
        return response.choices[0].message.content.strip()
    except (APITimeoutError, APIConnectionError, RateLimitError) as e:
        print(f"OpenAI summarize_hints timeout/connection/rate-limit: {e}")
        return hint_list[0] if hint_list else ""
    except Exception as e:
        print(f"Error summarizing reasons: {e}")
        return hint_list[0] if hint_list else ""

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