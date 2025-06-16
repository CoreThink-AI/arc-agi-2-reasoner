import os
import openai
import anthropic
import requests

anthropic_client = anthropic.Anthropic()


def get_anthropic_response(prompt):
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
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
    with anthropic_client.messages.stream(
        model="claude-opus-4-20250514",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
        },
        messages=[{
            "role": "user",
            "content": prompt
        }]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)


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

    if provider == "openai":
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not model:
            model = "gpt-4.1-2025-04-14"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
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


