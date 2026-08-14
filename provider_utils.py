"""
provider_utils.py  TF1 async provider call wrapper with timeout, retry,
and failure classification.

Centralises the try/except/timeout/retry pattern so specialist nodes
do not duplicate boilerplate.  Returns a CapabilityResult[T] rather than
raising exceptions directly, keeping specialist logic clean.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

from agent_config import (
    PROVIDER_TIMEOUT_SECONDS,
    RETRY_MAX_ATTEMPTS,
    RETRY_BASE_DELAY_SECONDS,
)
from schemas import CapabilityResult, ErrorCode, RETRYABLE_ERROR_CODES


# ---------------------------------------------------------------------------
# Exception ? ErrorCode mapping
# ---------------------------------------------------------------------------

def _classify_exception(exc: BaseException) -> ErrorCode:
    """Map a raw exception to the closest semantic ErrorCode."""
    exc_type = type(exc).__name__
    exc_msg = str(exc).lower()

    if isinstance(exc, asyncio.TimeoutError):
        return ErrorCode.TIMEOUT

    if "rate" in exc_msg and "limit" in exc_msg:
        return ErrorCode.RATE_LIMITED

    if any(k in exc_msg for k in ("uvx was not found", "api key", "unauthorized", "forbidden", "missing")):
        return ErrorCode.AUTH_CONFIGURATION

    if any(k in exc_msg for k in ("connection", "network", "unavailable", "timeout", "refused")):
        return ErrorCode.UNAVAILABLE

    if "json" in exc_msg or "parse" in exc_msg or "schema" in exc_msg:
        return ErrorCode.INVALID_RESPONSE

    return ErrorCode.INTERNAL


def _is_retryable(code: ErrorCode) -> bool:
    return code in RETRYABLE_ERROR_CODES


# ---------------------------------------------------------------------------
# Main utility
# ---------------------------------------------------------------------------

async def async_call_provider(
    coro_factory: Callable[[], Awaitable[Any]],
    provider_name: str,
    safe_failure_message: str,
    *,
    timeout_override: float | None = None,
    max_attempts: int = RETRY_MAX_ATTEMPTS,
) -> CapabilityResult:
    """Call an async provider coroutine with bounded timeout and retry.

    Parameters
    ----------
    coro_factory:
        Zero-argument callable that returns a fresh coroutine each call.
        A new coroutine is needed per retry attempt.
    provider_name:
        Key into ``PROVIDER_TIMEOUT_SECONDS`` (e.g. ``"aviation"``).
    safe_failure_message:
        User-facing message returned on failure; MUST NOT contain internal
        details, credentials, or file paths.
    timeout_override:
        If provided, overrides the centralized timeout for this call.
    max_attempts:
        Total number of attempts.  Auth/config errors always stop immediately.
    """
    timeout_s = timeout_override if timeout_override is not None else PROVIDER_TIMEOUT_SECONDS.get(provider_name, 15.0)

    last_code = ErrorCode.INTERNAL
    last_exc: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        t_start = time.monotonic()
        try:
            async with asyncio.timeout(timeout_s):
                data = await coro_factory()
            latency_ms = int((time.monotonic() - t_start) * 1000)
            return CapabilityResult.ok(data=data, latency_ms=latency_ms)

        except Exception as exc:
            latency_ms = int((time.monotonic() - t_start) * 1000)
            last_exc = exc
            last_code = _classify_exception(exc)

            print(
                f"[provider_utils] {provider_name} attempt {attempt}/{max_attempts} "
                f"FAILED ({last_code}): {type(exc).__name__}: {exc}",
                flush=True,
            )

            # Never retry non-transient errors
            if not _is_retryable(last_code):
                break

            if attempt < max_attempts:
                await asyncio.sleep(RETRY_BASE_DELAY_SECONDS)

    return CapabilityResult.fail(
        error_code=last_code,
        safe_message=safe_failure_message,
    )
