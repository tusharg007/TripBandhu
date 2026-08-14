"""
provider_utils.py — TF2 async provider call wrapper with timeout, retry,
failure classification, cancellation safety, and trace capture.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from agent_config import (
    PROVIDER_TIMEOUT_SECONDS,
    RETRY_MAX_ATTEMPTS,
    RETRY_BASE_DELAY_SECONDS,
)
from schemas import (
    CapabilityResult,
    CapabilityTraceEntry,
    CapabilityHealth,
    ErrorCode,
    RETRYABLE_ERROR_CODES,
)


def _classify_exception(exc: BaseException) -> ErrorCode:
    """Map an exception to the closest semantic ErrorCode."""
    exc_type = type(exc).__name__
    exc_msg = str(exc).lower()

    if isinstance(exc, asyncio.TimeoutError):
        return ErrorCode.TIMEOUT

    if "rate" in exc_msg and "limit" in exc_msg:
        return ErrorCode.RATE_LIMITED

    if any(k in exc_msg for k in ("uvx was not found", "api key", "unauthorized", "forbidden", "missing")):
        return ErrorCode.AUTH_CONFIGURATION

    if "contract" in exc_msg or "mismatch" in exc_msg or "schema" in exc_msg:
        return ErrorCode.CONTRACT_MISMATCH

    if any(k in exc_msg for k in ("connection", "network", "unavailable", "timeout", "refused", "503")):
        return ErrorCode.UNAVAILABLE

    if "json" in exc_msg or "parse" in exc_msg:
        return ErrorCode.INVALID_RESPONSE

    return ErrorCode.INTERNAL


def _is_retryable(code: ErrorCode) -> bool:
    return code in RETRYABLE_ERROR_CODES


async def async_call_provider(
    coro_factory: Callable[[], Awaitable[Any]],
    provider_name: str,
    safe_failure_message: str,
    *,
    specialist: str = "specialist",
    capability: str = "generic_capability",
    server: str = "mcp_server",
    tool_name: str = "generic_tool",
    timeout_override: float | None = None,
    max_attempts: int = RETRY_MAX_ATTEMPTS,
) -> CapabilityResult:
    """Call an async provider coroutine with bounded timeout, bounded retry, and trace capture.

    Returns a CapabilityResult which also supports tuple unpacking: `result, trace = await ...`
    """
    timeout_s = timeout_override if timeout_override is not None else PROVIDER_TIMEOUT_SECONDS.get(provider_name, 15.0)

    started_at = datetime.now(timezone.utc).isoformat()
    last_code = ErrorCode.INTERNAL
    total_retries = 0
    t_start = time.monotonic()

    for attempt in range(1, max_attempts + 1):
        try:
            async with asyncio.timeout(timeout_s):
                data = await coro_factory()

            latency_ms = int((time.monotonic() - t_start) * 1000)
            completed_at = datetime.now(timezone.utc).isoformat()

            trace_entry = CapabilityTraceEntry(
                sequence=1,
                specialist=specialist,
                capability=capability,
                server=server,
                tool_name=tool_name,
                status=CapabilityHealth.AVAILABLE.value,
                latency_ms=latency_ms,
                retry_count=attempt - 1,
                started_at=started_at,
                completed_at=completed_at,
                error_code=None,
                source_count=1 if data else 0,
            )
            return CapabilityResult.ok(data=data, latency_ms=latency_ms, trace_entry=trace_entry)

        except asyncio.CancelledError:
            print(f"[provider_utils] Operation cancelled for {provider_name}/{capability}", flush=True)
            raise

        except Exception as exc:
            latency_ms = int((time.monotonic() - t_start) * 1000)
            last_code = _classify_exception(exc)
            total_retries = attempt - 1

            print(
                f"[provider_utils] {provider_name} attempt {attempt}/{max_attempts} "
                f"FAILED ({last_code}): {type(exc).__name__}: {exc}",
                flush=True,
            )

            if not _is_retryable(last_code):
                break

            if attempt < max_attempts:
                total_retries += 1
                await asyncio.sleep(RETRY_BASE_DELAY_SECONDS)

    completed_at = datetime.now(timezone.utc).isoformat()
    trace_entry = CapabilityTraceEntry(
        sequence=1,
        specialist=specialist,
        capability=capability,
        server=server,
        tool_name=tool_name,
        status=CapabilityHealth.UNAVAILABLE.value if last_code != ErrorCode.CONTRACT_MISMATCH else CapabilityHealth.CONTRACT_MISMATCH.value,
        latency_ms=int((time.monotonic() - t_start) * 1000),
        retry_count=total_retries,
        started_at=started_at,
        completed_at=completed_at,
        error_code=last_code.value,
        source_count=0,
    )

    return CapabilityResult.fail(
        error_code=last_code,
        safe_message=safe_failure_message,
        latency_ms=trace_entry.latency_ms,
        trace_entry=trace_entry,
    )
