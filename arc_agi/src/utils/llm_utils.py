import os
import asyncio
import random
import json
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
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
DEFAULT_OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "72000"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))

_httpx_async_client = httpx.AsyncClient(timeout=DEFAULT_OPENAI_TIMEOUT_SECONDS)
openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=DEFAULT_OPENAI_TIMEOUT_SECONDS,
    max_retries=OPENAI_MAX_RETRIES,
    http_client=_httpx_async_client,
)

# Async Grok client (x.ai) mirroring the OpenAI async setup
_httpx_async_client_xai = httpx.AsyncClient(timeout=DEFAULT_OPENAI_TIMEOUT_SECONDS)
grok_async_client = AsyncOpenAI(
    api_key=os.environ.get("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
    timeout=DEFAULT_OPENAI_TIMEOUT_SECONDS,
    max_retries=OPENAI_MAX_RETRIES,
    http_client=_httpx_async_client_xai,
)

"""Get completion with retry logic, timeouts, and rate limiting using OpenAI."""
grok_api_key_2 = os.environ.get("XAI_API_KEY_FLOW_2")
if not grok_api_key_2:
    raise ValueError("Missing XAI API key. Please set XAI_API_KEY_FLOW_2 in the environment.")

grok_async_client_2 = AsyncOpenAI(
        api_key=grok_api_key_2,
        base_url="https://api.x.ai/v1",
        timeout=DEFAULT_OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
        http_client=_httpx_async_client_xai,
    )

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("Missing GROQ API key. Please set GROQ_API_KEY in the environment.")
groq_async_client = AsyncOpenAI(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=DEFAULT_OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
        http_client=_httpx_async_client,
    )

async def call_llm(provider: str, prompt: str, model: str = None, temperature: float = 0.0, max_tokens: int = 40096) -> str:
    """
    Unified function but now Grok-only.
    Ignores 'provider' and always calls Grok.
    """
    sys_prompt = "You are an expert at solving grid-based reasoning problems. Use markdown output. Enclose code or grids in ```."

    provider = "groq"
    try:
        if provider == "openai":
            response = await openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                #temperature=temperature,
                #max_tokens=max_tokens
            )
            return response.choices[0].message.content
        elif provider == "groq":
            response = await groq_async_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        else:
            return await aget_grok_response_stream(prompt)

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


async def aget_grok_response_stream(
        prompt: str,
        system_prompt: str = "",
        XAI_API_KEY: str = ""
) -> str:
    """
    Async streaming Grok (x.ai) response.
    Falls back to a normal non-streaming response if streaming fails.

    Args:
        prompt (str): The main user prompt.
        system_prompt (str): Optional system-level instructions.
        XAI_API_KEY (str): Optional API key (defaults to env var `XAI_API_KEY`).

    Returns:
        str: The full response text, or an error message on failure.
    """

    # Resolve API key
    api_key = XAI_API_KEY or os.environ.get("XAI_API_KEY")
    if not api_key:
        raise ValueError("Missing XAI API key. Please provide it as an argument or set it in the environment.")

    # Create async client
    grok_async_client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
        timeout=DEFAULT_OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
        http_client=_httpx_async_client_xai,
    )

    final_response = ""

    try:
        # Streaming request
        stream = await grok_async_client.chat.completions.create(
            model="grok-4",
            messages=[
                {"role": "system", "content": f"You are an expert reasoner. {system_prompt}"},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )

        async for chunk in stream:
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                final_response += content

        return final_response.strip()

    except Exception as e:
        # Fallback to non-streaming request
        try:
            response = await grok_async_client.chat.completions.create(
                model="grok-4",
                messages=[
                    {"role": "system", "content": f"You are an expert reasoner. {system_prompt}"},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()

        except Exception as e2:
            return f"Error: {e2}"

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
                # backoff_seconds = min(1.5 ** (attempt + 1) + random.uniform(0, 0.25), 6.0)
                backoff_seconds = 0.1
                print(f"[OpenAI error] attempt {attempt + 1}/{retry_limit} failed: {e} — retrying in {backoff_seconds:.2f}s")
                await asyncio.sleep(backoff_seconds)

async def get_completion_with_retry_grok(
    img1,
    img2,
    semaphore,
    PatternDetectionResponse,
    prompt: str,
    max_retries: int = None,
):

    retry_limit = max_retries or OPENAI_MAX_RETRIES
    async with semaphore:  # Limit concurrent requests
        for attempt in range(retry_limit):
            try:
                # Small jitter to avoid herd behavior
                await asyncio.sleep(random.uniform(0.05, 0.1))

                response = await grok_async_client_2.beta.chat.completions.parse(
                    model='grok-4',
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt}
                            ],
                        }
                    ],
                    response_format=PatternDetectionResponse,
                    timeout=DEFAULT_OPENAI_TIMEOUT_SECONDS,
                )
                result = response.choices[0].message.parsed
                print("***********************************")
                print(result)
                print("***********************************")
                return result

            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                is_last = attempt == retry_limit - 1
                if is_last:
                    print(f"[Grok timeout/connection/rate-limit] Giving up after {retry_limit} attempts: {e}")
                    return None
                backoff_seconds = min(2 ** attempt + random.uniform(0, 0.5), 8.0)
                print(f"[Grok retryable error] attempt {attempt + 1}/{retry_limit} failed: {e} — backing off {backoff_seconds:.2f}s")
                await asyncio.sleep(backoff_seconds)
            except Exception as e:
                is_last = attempt == retry_limit - 1
                if is_last:
                    print(f"[Grok fatal error] Failed after {retry_limit} attempts: {e}")
                    return None
                # backoff_seconds = min(1.5 ** (attempt + 1) + random.uniform(0, 0.25), 6.0)
                backoff_seconds = 0.1
                print(f"[Grok error] attempt {attempt + 1}/{retry_limit} failed: {e} — retrying in {backoff_seconds:.2f}s")
                await asyncio.sleep(backoff_seconds)

def flatten_lists_in_params(params):
    if params is None:
        return params
    new_params = {}
    for k, v in params.items():
        # If it's a list of lists, flatten it
        if isinstance(v, list) and any(isinstance(i, list) for i in v):
            # Flatten one level if all elements are lists
            flat = []
            for i in v:
                if isinstance(i, list):
                    flat.extend(i)
                else:
                    flat.append(i)
            new_params[k] = flat
        else:
            new_params[k] = v
    return new_params

def fix_result_dict(data):
    for result in data.get("result", []):
        if "params" in result:
            result["params"] = flatten_lists_in_params(result["params"])
    return data

async def get_completion_with_retry_groq(
    img1,
    img2,
    semaphore,
    PatternDetectionResponse,
    prompt: str,
    max_retries: int = None,
):

    retry_limit = max_retries or OPENAI_MAX_RETRIES
    async with semaphore:  # Limit concurrent requests
        for attempt in range(retry_limit):
            try:
                # Small jitter to avoid herd behavior
                await asyncio.sleep(random.uniform(0.05, 0.1))
                response = await groq_async_client.beta.chat.completions.parse(
                    model='openai/gpt-oss-120b',
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                """
                                You are a pattern detection engine.
                                Respond ONLY with the output JSON, enclosed in a markdown code block (triple backticks) and formatted as specified below.
                
                                The JSON must follow this exact schema:
                                {
                                  "result": [
                                    {
                                      "reason": "Short, precise explanation of why this pattern is or isn’t detected",
                                      "pattern_detected": true, // or false
                                      "pattern_name": "Exact pattern name",
                                      "pattern_description": "Detailed description of the detected pattern and its nature",
                                      "params": {
                                        "param_key1": ["string_value1", "string_value2"],
                                        "param_key2": ["string_value3"]
                                      }
                                    },
                                    {
                                      "reason": "Explanation for another pattern result",
                                      "pattern_detected": false,
                                      "pattern_name": "Another pattern name",
                                      "pattern_description": "Description here",
                                      "params": null
                                    }
                                  ]
                                }
                
                                Rules:
                                - Always include the root key \"result\". Its value is an array containing one or more pattern detection result objects.
                                - For each object:
                                    - \"reason\": Provide a concise explanation.
                                    - \"pattern_detected\": Boolean value only (true or false).
                                    - \"pattern_name\": Use a specific, descriptive name.
                                    - \"pattern_description\": Give a detailed, informative description of the pattern or absence thereof.
                                    - \"params\": If relevant, give a dictionary mapping string keys to lists of strings; otherwise, use null.
                                - NEVER add any text, annotation, or explanation outside the markdown code block.
                                - Only output raw JSON, and ONLY within the markdown code block.
                                - Ensure output is strict, valid JSON and matches the given schema.
                
                                Good Example:
                                {
                                  \"result\": [
                                    {
                                      \"reason\": \"Date in DD/MM/YYYY format found in the input.\",
                                      \"pattern_detected\": true,
                                      \"pattern_name\": \"Date Pattern\",
                                      \"pattern_description\": \"Looks for sequences like '17/08/2025' or '03-12-2024'.\",
                                      \"params\": {
                                        \"matches\": [\"17/08/2025\", \"03-12-2024\"]
                                      }
                                    },
                                    {
                                      \"reason\": \"No email addresses matched the regex.\",
                                      \"pattern_detected\": false,
                                      \"pattern_name\": \"Email Pattern\",
                                      \"pattern_description\": \"Detects standard email addresses in provided input.\",
                                      \"params\": null
                                    }
                                  ]
                                }
                
                                Good Example:
                                {
                                  \"result\": [
                                    {
                                      \"reason\": \"Faces detected in both images.\",
                                      \"pattern_detected\": true,
                                      \"pattern_name\": \"Face Detection\",
                                      \"pattern_description\": \"Identifies human faces present in photographs using computer vision.\",
                                      \"params\": {
                                        \"coordinates_img1\": [\"(245,120)\", \"(320,98)\"],
                                        \"coordinates_img2\": [\"(130,97)\"]
                                      }
                                    }
                                  ]
                                }
                
                                Bad Examples:
                                - Outputting anything before or after the code block.
                                - Using string instead of boolean for \"pattern_detected\".
                                - Omitting the \"result\" root key.
                                - Including extra keys, comments, or text.
                
                                Do not add explanations, comments, or markdown outside the JSON code block.
                                """
                            ),
                        },
                        {"role": "user", "content": prompt}
                    ],
                    # response_format=PatternDetectionResponse,
                    timeout=DEFAULT_OPENAI_TIMEOUT_SECONDS,
                    reasoning_effort="high"
                )
                model_output = response.choices[0].message.content
                model_output_ = model_output.split("```")
                raw_json = model_output_[1][5:].strip()
                try:
                    parsed = json.loads(raw_json)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to decode JSON: {e}\nRaw: {raw_json}")

                # Always return dict with 'result' key
                if isinstance(parsed, list):
                    result = {"result": parsed}
                elif isinstance(parsed, dict) and "result" in parsed:
                    result = parsed
                else:
                    raise ValueError("JSON must be a dict with 'result' or a list.")

                result = fix_result_dict(result)

                obj = PatternDetectionResponse.parse_obj(result)
                print("***********************************")
                print(obj)
                print("***********************************")

                return obj

            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                is_last = attempt == retry_limit - 1
                if is_last:
                    print(f"[Groq timeout/connection/rate-limit] Giving up after {retry_limit} attempts: {e}")
                    return None
                backoff_seconds = min(2 ** attempt + random.uniform(0, 0.5), 8.0)
                print(f"[Groq retryable error] attempt {attempt + 1}/{retry_limit} failed: {e} — backing off {backoff_seconds:.2f}s")
                await asyncio.sleep(backoff_seconds)
            except Exception as e:
                is_last = attempt == retry_limit - 1
                if is_last:
                    print(f"[Groq fatal error] Failed after {retry_limit} attempts: {e}")
                    return None
                # backoff_seconds = min(1.5 ** (attempt + 1) + random.uniform(0, 0.25), 6.0)
                backoff_seconds = 0.1
                print(f"[Groq error] attempt {attempt + 1}/{retry_limit} failed: {e} — retrying in {backoff_seconds:.2f}s")
                await asyncio.sleep(backoff_seconds)

async def get_completion(grid1, grid2, semaphore, PatternDetectionResponse, prompt: str):
    img1 = array_to_base64_image(grid1)
    img2 = array_to_base64_image(grid2)
    return await get_completion_with_retry(img1, img2, semaphore, PatternDetectionResponse, prompt)

async def get_completion_grok(grid1, grid2, semaphore, PatternDetectionResponse, prompt: str):
    img1 = array_to_base64_image(grid1)
    img2 = array_to_base64_image(grid2)
    return await get_completion_with_retry_grok(img1, img2, semaphore, PatternDetectionResponse, prompt)

async def get_completion_groq(grid1, grid2, semaphore, PatternDetectionResponse, prompt: str):
    img1 = array_to_base64_image(grid1)
    img2 = array_to_base64_image(grid2)
    return await get_completion_with_retry_groq(img1, img2, semaphore, PatternDetectionResponse, prompt)

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
        response = await groq_async_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=20000,
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
        response = await groq_async_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=20000,
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