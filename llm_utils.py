"""
llm_utils.py — Centralized LLM invocation with rate-limit handling and token accounting.

ALL ChatGroq calls in backend.py MUST route through invoke_llm() so that:
1. Groq 429 / RateLimitError is handled consistently with our own retry policy
   (not silently retried by the Groq SDK or LangChain internals).
2. Token usage metadata is extracted and returned where available.
3. Rate-limit behavior is transparent: short-wait (<=15s) → one retry;
   long-wait → immediate degradation without holding the user open.

ChatGroq instances MUST be created with max_retries=0 (enforced in make_groq_llm
in mcp_client.py) so our policy owns all retry decisions.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from agent_config import (
    PROVIDER_TIMEOUT_SECONDS,
    RATE_LIMIT_SHORT_WAIT_MAX_SECONDS,
    RATE_LIMIT_DEFAULT_RETRY_WAIT_SECONDS,
)


# ---------------------------------------------------------------------------
# Rate-limit detection helpers
# ---------------------------------------------------------------------------

def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if exc is a Groq 429 / RateLimitError."""
    exc_type = type(exc).__name__.lower()
    exc_msg = str(exc).lower()
    return (
        "ratelimit" in exc_type
        or "rate_limit" in exc_type
        or "rate limit" in exc_msg
        or "429" in exc_msg
        or getattr(exc, "status_code", None) == 429
    )


def _get_retry_after(exc: Exception) -> int | None:
    """
    Extract Retry-After seconds from a rate-limit exception, if available.
    Checks HTTP response headers first, then falls back to the error message string.
    Returns None when the value is absent or cannot be parsed.
    """
    # From HTTP response headers (Groq SDK APIStatusError.response)
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {}) or {}
        ra = headers.get("retry-after") or headers.get("Retry-After")
        if ra:
            try:
                return int(ra)
            except (ValueError, TypeError):
                pass

    # From error message body
    exc_msg = str(exc)
    m = re.search(r"(?:retry.?after|retry_after)[:\s=]+(\d+)", exc_msg, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except (ValueError, TypeError):
            pass

    return None


# ---------------------------------------------------------------------------
# Token usage extraction
# ---------------------------------------------------------------------------

def _extract_token_usage(response: Any) -> dict[str, int]:
    """
    Extract token usage counts from any LLM response type.

    Handles:
    - AIMessage (standard ChatGroq response)
    - include_raw=True dict: {"raw": AIMessage, "parsed": PydanticModel, ...}
    - Pydantic models from structured output without include_raw (no metadata)

    Returns an empty dict when metadata is unavailable (never raises).
    """
    # include_raw=True returns {"raw": AIMessage, "parsed": ..., "parsing_error": ...}
    if isinstance(response, dict) and "raw" in response:
        response = response.get("raw")

    if hasattr(response, "response_metadata"):
        usage = (response.response_metadata or {}).get("token_usage", {}) or {}
        return {k: int(v) for k, v in usage.items() if isinstance(v, (int, float))}

    # Pydantic model from structured output: no token metadata available here.
    # LangSmith trace (when LANGSMITH_TRACING=true) captures per-call tokens.
    return {}


def merge_token_usage(base: dict[str, int], delta: dict[str, int]) -> dict[str, int]:
    """Accumulate token usage deltas into a running total dict."""
    result = dict(base)
    for key, value in delta.items():
        result[key] = result.get(key, 0) + value
    return result


# ---------------------------------------------------------------------------
# Centralized LLM invocation
# ---------------------------------------------------------------------------

async def invoke_llm(
    runnable: Any,
    messages: Any,
    *,
    task_name: str = "llm",
    timeout_s: float | None = None,
) -> tuple[Any, dict[str, int]]:
    """
    Invoke a LangChain runnable (ChatGroq or structured output chain) with:
    - asyncio timeout
    - centralized 429 / RateLimitError handling:
        * Short Retry-After (<= RATE_LIMIT_SHORT_WAIT_MAX_SECONDS): one bounded retry
        * Long Retry-After (> threshold): degrade immediately, no retry, no sleep
    - Token usage extraction from response metadata

    Returns:
        (response, token_usage_delta)

        response: raw runnable output (AIMessage for generation calls; Pydantic
                  model for structured output calls)
        token_usage_delta: dict[str, int] with prompt_tokens / completion_tokens /
                           total_tokens — empty when metadata is unavailable (e.g.,
                           structured output without include_raw=True)
    """
    if timeout_s is None:
        timeout_s = PROVIDER_TIMEOUT_SECONDS.get("llm", 45.0)

    max_attempts = 2  # 1 initial + 1 retry (for short-wait only)

    for attempt in range(1, max_attempts + 1):
        try:
            async with asyncio.timeout(timeout_s):
                response = await runnable.ainvoke(messages)
            token_usage = _extract_token_usage(response)
            return response, token_usage

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            if not _is_rate_limit_error(exc):
                # Non-rate-limit error: propagate immediately to calling agent's handler
                raise

            retry_after = _get_retry_after(exc)

            if retry_after is not None and retry_after > RATE_LIMIT_SHORT_WAIT_MAX_SECONDS:
                # Long-wait quota exhaustion: degrade immediately without sleeping
                print(
                    f"[invoke_llm:{task_name}] Long-wait rate limit "
                    f"(Retry-After={retry_after}s > {RATE_LIMIT_SHORT_WAIT_MAX_SECONDS}s). "
                    "Degrading immediately.",
                    flush=True,
                )
                raise

            if attempt < max_attempts:
                wait = float(retry_after) if retry_after else RATE_LIMIT_DEFAULT_RETRY_WAIT_SECONDS
                print(
                    f"[invoke_llm:{task_name}] Short rate limit "
                    f"(Retry-After={retry_after}s), retry in {wait:.1f}s "
                    f"({attempt}/{max_attempts - 1})...",
                    flush=True,
                )
                await asyncio.sleep(wait)
            else:
                # All retries exhausted
                print(
                    f"[invoke_llm:{task_name}] Rate limit retries exhausted. Degrading.",
                    flush=True,
                )
                raise
