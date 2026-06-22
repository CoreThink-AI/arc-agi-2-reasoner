"""
KiVA LLM Client — supports Google Gemini (native) and OpenRouter.

Backend selection (automatic):
  - GEMINI_API_KEY set + model slug has no "/" → Google Generative AI SDK
  - Otherwise → OpenRouter (OPENROUTER_API_KEY required)

Completely isolated from arc_agi/src/utils/llm_utils.py so that importing
this module does not require any ARC-specific env vars (XAI_API_KEY, etc.).

Usage:
    from kiva.llm_client import call_vision, call_text

    response = await call_vision(
        system="You are a visual reasoning expert.",
        user_text="What transformation is shown?",
        image_paths=["path/to/input.png", "path/to/output.png"],
    )
"""

from __future__ import annotations

import asyncio
import base64
import os
import random
from pathlib import Path
from typing import List, Optional

import httpx
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError

# ── Model defaults ─────────────────────────────────────────────────────────────

# Set KIVA_VISION_MODEL / KIVA_TEXT_MODEL in .env to override.
# Gemini key present  → default to "gemini-2.0-flash"  (native SDK)
# OpenRouter only     → default to "openai/gpt-4o-mini" (OpenRouter)
def _default_model() -> str:
    if os.getenv("GEMINI_API_KEY"):
        return "gemini-2.0-flash"
    if os.getenv("XAI_API_KEY"):
        return "grok-2-vision-1212"
    return "openai/gpt-4o-mini"

DEFAULT_VISION_MODEL = os.getenv("KIVA_VISION_MODEL") or _default_model()
DEFAULT_TEXT_MODEL   = os.getenv("KIVA_TEXT_MODEL")   or _default_model()

TIMEOUT_SECONDS = float(os.getenv("KIVA_LLM_TIMEOUT", "120"))
MAX_RETRIES     = int(os.getenv("KIVA_LLM_MAX_RETRIES", "3"))

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
XAI_BASE_URL        = "https://api.x.ai/v1"

# Global token cap — set via set_max_tokens_cap() at startup (e.g. from eval.py --max-tokens).
# When set, every LLM call is capped to this value regardless of per-call max_tokens.
_MAX_TOKENS_CAP: Optional[int] = None


def set_max_tokens_cap(n: int) -> None:
    global _MAX_TOKENS_CAP
    _MAX_TOKENS_CAP = n


# Global token usage counters — accumulated across all calls in a run.
_tokens_prompt: int = 0
_tokens_completion: int = 0
_tokens_total: int = 0


def get_token_stats() -> dict:
    return {
        "prompt_tokens":     _tokens_prompt,
        "completion_tokens": _tokens_completion,
        "total_tokens":      _tokens_total,
    }


def reset_token_stats() -> None:
    global _tokens_prompt, _tokens_completion, _tokens_total
    _tokens_prompt = _tokens_completion = _tokens_total = 0


def _record_usage(prompt: int, completion: int) -> None:
    global _tokens_prompt, _tokens_completion, _tokens_total
    _tokens_prompt     += prompt
    _tokens_completion += completion
    _tokens_total      += prompt + completion


def _use_gemini(model: str) -> bool:
    return bool(os.getenv("GEMINI_API_KEY")) and "/" not in model and model.startswith("gemini")


def _use_xai(model: str) -> bool:
    return bool(os.getenv("XAI_API_KEY")) and model.startswith("grok")


# ── OpenRouter client (lazy singleton) ────────────────────────────────────────

_openrouter_client: Optional[AsyncOpenAI] = None
_xai_client: Optional[AsyncOpenAI] = None


def _get_openrouter_client() -> AsyncOpenAI:
    global _openrouter_client
    if _openrouter_client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set.\n"
                "Add it to your .env file:  OPENROUTER_API_KEY=sk-or-..."
            )
        _openrouter_client = AsyncOpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            timeout=TIMEOUT_SECONDS,
            max_retries=0,
            http_client=httpx.AsyncClient(timeout=TIMEOUT_SECONDS),
        )
    return _openrouter_client


def _get_xai_client() -> AsyncOpenAI:
    global _xai_client
    if _xai_client is None:
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise EnvironmentError("XAI_API_KEY is not set.")
        _xai_client = AsyncOpenAI(
            api_key=api_key,
            base_url=XAI_BASE_URL,
            timeout=TIMEOUT_SECONDS,
            max_retries=0,
            http_client=httpx.AsyncClient(timeout=TIMEOUT_SECONDS),
        )
    return _xai_client


# ── Image encoding ─────────────────────────────────────────────────────────────

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_content_block(image_path: str) -> dict:
    """OpenAI-format image content block (used for OpenRouter calls)."""
    b64 = encode_image_to_base64(image_path)
    suffix = Path(image_path).suffix.lower().lstrip(".")
    mime = "image/png" if suffix == "png" else f"image/{suffix}"
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }


def _gemini_inline_image(image_path: str) -> dict:
    """Gemini inline_data image part built from a local file."""
    b64 = encode_image_to_base64(image_path)
    suffix = Path(image_path).suffix.lower().lstrip(".")
    mime = "image/png" if suffix == "png" else f"image/{suffix}"
    return {"inline_data": {"mime_type": mime, "data": b64}}


# ── Gemini backend ─────────────────────────────────────────────────────────────

async def _call_gemini_with_retry(
    messages: List[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    try:
        from google import genai as google_genai
        from google.genai import types as genai_types
    except ImportError:
        return (
            "Error: google-genai package not installed. "
            "Run: pip install google-genai"
        )

    client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Extract system instruction and build contents list
    system_instruction: Optional[str] = None
    contents = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_instruction = content if isinstance(content, str) else str(content)
            continue

        parts = []
        if isinstance(content, str):
            parts.append(genai_types.Part.from_text(text=content))
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    parts.append(genai_types.Part.from_text(text=block["text"]))
                elif block.get("type") == "image_url":
                    url = block["image_url"]["url"]  # data:<mime>;base64,<data>
                    header, b64 = url.split(",", 1)
                    mime = header.split(":")[1].split(";")[0]
                    parts.append(genai_types.Part.from_bytes(
                        data=base64.b64decode(b64), mime_type=mime
                    ))

        gemini_role = "user" if role == "user" else "model"
        contents.append(genai_types.Content(role=gemini_role, parts=parts))

    config = genai_types.GenerateContentConfig(
        system_instruction=system_instruction,
        max_output_tokens=max_tokens,
        temperature=temperature,
        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        safety_settings=[
            genai_types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            genai_types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            genai_types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            genai_types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        ],
    )

    last_error: Exception = RuntimeError("No attempts made")
    for attempt in range(MAX_RETRIES):
        try:
            await asyncio.sleep(random.uniform(0.1, 0.3) * (2 ** attempt))
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            # Extract text — skip internal "thought" parts (Gemini 2.5 thinking mode)
            if response.candidates:
                candidate = response.candidates[0]
                # Record token usage from Gemini metadata
                meta = getattr(response, "usage_metadata", None)
                if meta:
                    _record_usage(
                        getattr(meta, "prompt_token_count", 0) or 0,
                        getattr(meta, "candidates_token_count", 0) or 0,
                    )
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        # thought=True parts are internal reasoning, skip them
                        if getattr(part, "thought", False):
                            continue
                        if part.text:
                            return part.text
                finish = getattr(candidate, "finish_reason", "unknown")
                return f"Error: Response blocked (finish_reason={finish})"
            return "Error: No candidates returned"
        except Exception as e:
            last_error = e
            if attempt == MAX_RETRIES - 1:
                print(f"[KiVA LLM] Gemini fatal error after {MAX_RETRIES} attempts: {e}")
            else:
                print(f"[KiVA LLM] Gemini attempt {attempt + 1} failed: {e}, retrying...")

    return f"Error: {last_error}"


# ── OpenRouter backend ─────────────────────────────────────────────────────────

async def _call_openrouter_with_retry(
    messages: List[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    client: Optional[AsyncOpenAI] = None,
) -> str:
    client = client or _get_openrouter_client()
    last_error: Exception = RuntimeError("No attempts made")

    for attempt in range(MAX_RETRIES):
        try:
            await asyncio.sleep(random.uniform(0.1, 0.3) * (2 ** attempt))
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if response.usage:
                _record_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
            return response.choices[0].message.content or ""

        except (APITimeoutError, APIConnectionError, RateLimitError) as e:
            last_error = e
            if attempt == MAX_RETRIES - 1:
                print(f"[KiVA LLM] Giving up after {MAX_RETRIES} attempts: {e}")
            else:
                print(f"[KiVA LLM] Attempt {attempt + 1} failed ({type(e).__name__}), retrying...")

        except Exception as e:
            last_error = e
            if attempt == MAX_RETRIES - 1:
                print(f"[KiVA LLM] Fatal error after {MAX_RETRIES} attempts: {e}")
            else:
                print(f"[KiVA LLM] Attempt {attempt + 1} error: {e}, retrying...")

    return f"Error: {last_error}"


# ── Router ─────────────────────────────────────────────────────────────────────

async def _call_with_retry(
    messages: List[dict],
    model: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    if _MAX_TOKENS_CAP is not None:
        max_tokens = min(max_tokens, _MAX_TOKENS_CAP)
    if _use_gemini(model):
        return await _call_gemini_with_retry(messages, model, max_tokens, temperature)
    if _use_xai(model):
        return await _call_openrouter_with_retry(messages, model, max_tokens, temperature,
                                                  client=_get_xai_client())
    return await _call_openrouter_with_retry(messages, model, max_tokens, temperature)


# ── Public API ─────────────────────────────────────────────────────────────────

async def call_vision(
    user_text: str,
    image_paths: List[str],
    system: str = "You are an expert visual reasoning assistant.",
    model: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Call a vision model with text + images."""
    content: List[dict] = [{"type": "text", "text": user_text}]
    for path in image_paths:
        content.append(_image_content_block(path))

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
    return await _call_with_retry(messages, model or DEFAULT_VISION_MODEL, max_tokens, temperature)


async def call_text(
    user_text: str,
    system: str = "You are an expert reasoning assistant.",
    model: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Call a text-only model (for summarisation, aggregation, etc.)."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]
    return await _call_with_retry(messages, model or DEFAULT_TEXT_MODEL, max_tokens, temperature)
