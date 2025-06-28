import os
import openai
from openai import OpenAI
from openai import AsyncAzureOpenAI, AzureOpenAI
from pydantic import BaseModel
from typing import List
import anthropic
import asyncio
import random
import requests
from openai import AsyncOpenAI
from typing import List
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras
from arc_agi.src.patterns.detailed_hint_prompt import HINT_SUMMARY_PROMPT
load_dotenv()

from .visualization_utils import array_to_base64_image

anthropic_client = anthropic.Anthropic()
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
cerebras_client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME", "o4-mini")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

# Initialize Azure OpenAI client with key-based authentication
#openai_client = AsyncAzureOpenAI(
#    azure_endpoint=endpoint,
#    api_key=subscription_key,
#    api_version="2025-03-01-preview",
#)
#client = AzureOpenAI(
#    azure_endpoint=endpoint,
#    api_key=subscription_key,
#    api_version="2025-03-01-preview",
#)

def get_anthropic_response(prompt):
    response = anthropic_client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=32000,
        thinking={
            "type": "enabled",
            "budget_tokens": 16000
        },
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    # The response will contain summarized thinking blocks and text blocks
    for block in response.content:
        if block.type == "thinking":
            print(f"\nThinking summary: {block.thinking}")
        elif block.type == "text":
            print(f"\nResponse: {block.text}")


def get_anthropic_response_stream(prompt):
    response_text = ""
    with anthropic_client.messages.stream(
        model="claude-opus-4-20250514",
        max_tokens=32000,
        thinking={
            "type": "enabled",
            "budget_tokens": 16000
        },
        messages=[{
            "role": "user",
            "content": prompt
        }]
    ) as stream:
        for text in stream.text_stream:
            # print(text, end="", flush=True)
            response_text += text
    
    return response_text

def get_cerebras_response(prompt: str) -> str:
    response = cerebras_client.chat.completions.create(
        model="qwen-3-32b",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def call_llm(provider: str, prompt: str, model: str = None, temperature: float = 0.0, max_tokens: int = 4096) -> str:
    """
    Generic function to call different LLM providers.

    Args:
        provider (str): One of 'openai', 'anthropic', or 'together'
        prompt (str): The prompt to send to the LLM.
        model (str): Model name (required for OpenAI and Together).
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum tokens to generate.

    Returns:
        str: The generated response from the LLM.
    """
    provider = provider.lower()
    client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-03-01-preview",
    )
    if provider == "openai":
        #client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        #if not model:
        model = deployment
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            #temperature=temperature,
            #max_tokens=max_tokens
        )
        return response.choices[0].message.content

    elif provider == "anthropic":
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        if not model:
            model = "claude-3-opus-20240229"
        response = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.content[0].text

    elif provider == "together":
        if not model:
            model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        headers = {
            "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = requests.post("https://api.together.xyz/v1/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["text"]

    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
async def get_completion_with_retry(img1,img2,semaphore, PatternDetectionResponse, prompt: str, max_retries: int = 3):
    """Get completion with retry logic and rate limiting using OpenAI"""
    async with semaphore:  # Limit concurrent requests
        for attempt in range(max_retries):
            try:
                # Add small delay to avoid hitting rate limits
                await asyncio.sleep(random.uniform(0.2, 0.8))
                
                response = await openai_client.beta.chat.completions.parse(
                    model=deployment,
                    messages=[{
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
                    }
                ],
            }],
                    response_format=PatternDetectionResponse,
                )
                print(len(prompt))
                result = response.choices[0].message.parsed
                return result
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts: {e}")
                    return None  # Return None when all retries fail
                else:
                    print(f"Attempt {attempt + 1} failed: {e}, retrying...")
                    await asyncio.sleep(random.uniform(1.0, 2.0))  # Longer delay before retry

async def get_completion(grid1,grid2,semaphore,PatternDetectionResponse,prompt: str):
    img1 = array_to_base64_image(grid1)
    img2 = array_to_base64_image(grid2)
    return await get_completion_with_retry(img1,img2,semaphore,PatternDetectionResponse,prompt)

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

    Please provide a step by step account of how the input image would go through the transformation to reach the output image:"""

    try:
        response = await openai_client.chat.completions.create(
            model=deployment,  
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=200
        )
        return response.choices[0].message.content.strip()
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
            max_completion_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error summarizing reasons: {e}")
        return combined_hints[0] 

class GridModel(BaseModel):
    grid: List[List[int]]

def parse_grid(response):
    client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-03-01-preview",
)
    response = client.responses.parse(
        model=deployment,
        input=[
            {"role": "system", "content": "Extract the 2D grid from the description."},
            {
                "role": "user",
                "content": f"{response}",
            },
        ],
        text_format=GridModel,
    )

    gm: GridModel = response.output_parsed
    return [gm.grid]



