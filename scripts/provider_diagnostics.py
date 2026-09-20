"""Run bounded, sanitized provider checks from the current TripBandhu runtime.

This command prints only health, timing, source counts, transport/cache metadata,
and typed error codes. It never prints credentials, request URLs, or raw payloads.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capability_registry import normalize_tavily_results
from mcp_client import (
    aviation_mcp_call,
    close_provider_clients,
    forecast_mcp_search,
    start_provider_clients,
    tavily_mcp_search,
    weather_mcp_search,
)
from provider_utils import async_call_provider
from tools.flight_tool import DEFAULT_ORIGIN_IATA, resolve_location_to_iata


def _count_records(value: Any, field: str = "data") -> int:
    if isinstance(value, dict) and isinstance(value.get(field), list):
        return len(value[field])
    if isinstance(value, list):
        return len(value)
    return 1 if value else 0


def _result_summary(name: str, result: Any) -> dict[str, Any]:
    trace = result.trace_entry
    return {
        "provider": name,
        "success": result.success,
        "status": trace.status if trace else "UNKNOWN",
        "error_code": result.error_code.value if result.error_code else None,
        "latency_ms": result.latency_ms,
        "source_count": trace.source_count if trace else 0,
        "transport": trace.transport if trace else None,
        "cache_status": trace.cache_status if trace else None,
        "provider_status_code": trace.provider_status_code if trace else None,
        "provider_request_id": trace.provider_request_id if trace else None,
        "stage_timings_ms": trace.stage_timings_ms if trace else {},
    }


async def _run(args: argparse.Namespace) -> list[dict[str, Any]]:
    await start_provider_clients()
    checks: list[tuple[str, Any]] = []
    try:
        if args.provider in {"all", "aviation"}:
            departure = resolve_location_to_iata(args.origin) or DEFAULT_ORIGIN_IATA
            arrival = resolve_location_to_iata(args.destination)
            aviation = await async_call_provider(
                lambda: aviation_mcp_call(
                    "list_routes",
                    {"dep_iata": departure, "arr_iata": arrival} if arrival else {"dep_iata": departure},
                ),
                provider_name="aviation",
                safe_failure_message="Aviation provider unavailable.",
                specialist="diagnostic",
                capability="FLIGHT_ROUTE_SEARCH",
                server="aviationstack",
                tool_name="list_routes",
                source_count_fn=lambda value: _count_records(value, "data"),
            )
            checks.append(("aviation", aviation))

        if args.provider in {"all", "tavily"}:
            query = f"Hotels accommodation and neighborhoods in {args.destination}"
            tavily = await async_call_provider(
                lambda: tavily_mcp_search(query),
                provider_name="tavily",
                safe_failure_message="Tavily provider unavailable.",
                specialist="diagnostic",
                capability="HOTEL_WEB_RESEARCH",
                server="tavily",
                tool_name="tavily_search",
            )
            if tavily.success:
                evidence = normalize_tavily_results(
                    tavily.data,
                    query,
                    latency_ms=tavily.latency_ms,
                    destination=args.destination,
                )
                tavily.trace_entry.source_count = len(evidence.sources)
                if not evidence.data:
                    tavily.success = False
                    tavily.trace_entry.status = "DEGRADED"
            checks.append(("tavily", tavily))

        if args.provider in {"all", "weather"}:
            current, forecast = await asyncio.gather(
                async_call_provider(
                    lambda: weather_mcp_search(args.destination),
                    provider_name="weather",
                    safe_failure_message="Current weather unavailable.",
                    specialist="diagnostic",
                    capability="WEATHER_CURRENT",
                    server="weather",
                    tool_name="get_current_weather",
                ),
                async_call_provider(
                    lambda: forecast_mcp_search(args.destination),
                    provider_name="weather",
                    safe_failure_message="Forecast unavailable.",
                    specialist="diagnostic",
                    capability="WEATHER_FORECAST",
                    server="weather",
                    tool_name="get_forecast",
                    source_count_fn=lambda value: _count_records(value, "forecast"),
                ),
            )
            checks.extend((("weather_current", current), ("weather_forecast", forecast)))
    finally:
        await close_provider_clients()

    return [_result_summary(name, result) for name, result in checks]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("all", "aviation", "tavily", "weather"), default="all")
    parser.add_argument("--origin", default="Delhi")
    parser.add_argument("--destination", default="Jaipur")
    args = parser.parse_args()
    results = asyncio.run(_run(args))
    print(json.dumps({"checks": results}, indent=2, sort_keys=True))
    return 0 if results and all(item["success"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
