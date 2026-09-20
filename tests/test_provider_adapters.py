"""Phase 1 direct-provider runtime and validation tests."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import provider_adapters
from capability_registry import normalize_tavily_results
from provider_adapters import (
    ProviderRequestError,
    ProviderRuntime,
    _validate_aviation,
    _validate_tavily,
    get_weather_forecast,
)
from provider_utils import async_call_provider
from schemas import ErrorCode


def test_provider_error_never_contains_request_url_or_key():
    error = ProviderRequestError(
        "aviation",
        ErrorCode.ACCESS_DENIED,
        status_code=403,
        provider_code="function_access_restricted",
    )
    rendered = str(error)
    assert "access_key" not in rendered
    assert "https://" not in rendered.casefold()
    assert "ACCESS_DENIED" in rendered


def test_tavily_validator_rejects_placeholder_and_keeps_real_sources():
    with pytest.raises(ProviderRequestError):
        _validate_tavily({"results": [{"title": "Web Search Result", "url": "", "content": ""}]})

    result = _validate_tavily(
        {
            "query": "hotels in Jaipur",
            "results": [
                {
                    "title": "Hotel Pearl Palace Jaipur",
                    "url": "https://example.test/jaipur-hotel",
                    "content": "A centrally located Jaipur hotel with practical neighborhood information.",
                }
            ],
        }
    )
    assert len(result["results"]) == 1


def test_hotel_normalizer_requires_destination_match_for_structured_api_results():
    evidence = normalize_tavily_results(
        {
            "results": [
                {
                    "title": "London hotel guide",
                    "url": "https://example.test/london-hotels",
                    "content": "A detailed guide to hotels and neighborhoods across central London.",
                }
            ]
        },
        "Hotels in Jaipur",
        destination="Jaipur",
    )
    assert evidence.data == []
    assert evidence.status.value == "DEGRADED"


def test_aviation_validator_rejects_wrong_route():
    payload = {
        "data": [
            {
                "departure": {"iata": "BOM"},
                "arrival": {"iata": "LHR"},
                "airline": {"name": "Example Air"},
            }
        ]
    }
    with pytest.raises(ProviderRequestError):
        _validate_aviation(payload, expected_departure="DEL", expected_arrival="LHR")


@pytest.mark.asyncio
async def test_runtime_coalesces_and_caches_validated_requests():
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)
        return httpx.Response(200, json={"data": [{"id": 1}]}, headers={"x-request-id": "req-1"})

    runtime = ProviderRuntime()
    await runtime.start()
    assert runtime._client is not None
    await runtime._client.aclose()
    runtime._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def request():
        return await runtime.request_json(
            provider="tavily",
            method="GET",
            url="https://provider.test/search",
            cache_key="same-request",
            ttl_seconds=60,
            validator=lambda payload: payload,
        )

    first, second = await asyncio.gather(request(), request())
    third = await request()
    try:
        assert calls == 1
        assert {first["_meta"]["cache_status"], second["_meta"]["cache_status"]} == {"miss", "coalesced"}
        assert third["_meta"]["cache_status"] == "hit"
        assert first["_meta"]["provider_request_id"] == "req-1"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_cancelled_creator_does_not_cancel_or_duplicate_shared_request():
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return httpx.Response(200, json={"data": [{"id": 1}]})

    runtime = ProviderRuntime()
    await runtime.start()
    assert runtime._client is not None
    await runtime._client.aclose()
    runtime._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def request():
        return await runtime.request_json(
            provider="tavily",
            method="GET",
            url="https://provider.test/search",
            cache_key="cancelled-owner",
            ttl_seconds=60,
            validator=lambda payload: payload,
        )

    creator = asyncio.create_task(request())
    await started.wait()
    creator.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creator

    waiter = asyncio.create_task(request())
    await asyncio.sleep(0)
    release.set()
    try:
        result = await waiter
        cached = await request()
        assert calls == 1
        assert result["_meta"]["cache_status"] == "coalesced"
        assert cached["_meta"]["cache_status"] == "hit"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_runtime_opens_circuit_after_repeated_provider_failures():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"error": "unavailable"})

    runtime = ProviderRuntime()
    await runtime.start()
    assert runtime._client is not None
    await runtime._client.aclose()
    runtime._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        for index in range(3):
            with pytest.raises(ProviderRequestError) as captured:
                await runtime.request_json(
                    provider="aviation",
                    method="GET",
                    url="https://provider.test/flights",
                    cache_key=f"failure-{index}",
                    ttl_seconds=60,
                )
            assert captured.value.error_code == ErrorCode.UNAVAILABLE

        with pytest.raises(ProviderRequestError) as circuit:
            await runtime.request_json(
                provider="aviation",
                method="GET",
                url="https://provider.test/flights",
                cache_key="circuit-open",
                ttl_seconds=60,
            )
        assert circuit.value.error_code == ErrorCode.CIRCUIT_OPEN
        assert calls == 3
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_provider_wrapper_rejects_empty_success_payload():
    async def empty_response():
        return {"data": []}

    result = await async_call_provider(
        empty_response,
        provider_name="aviation",
        safe_failure_message="Flight provider unavailable.",
        max_attempts=1,
    )
    assert result.success is False
    assert result.error_code == ErrorCode.INVALID_RESPONSE
    assert result.trace_entry.source_count == 0


@pytest.mark.asyncio
async def test_provider_wrapper_preserves_sanitized_direct_failure_metadata():
    async def denied_response():
        raise ProviderRequestError(
            "tavily",
            ErrorCode.ACCESS_DENIED,
            status_code=403,
            provider_request_id="request-123",
        )

    result = await async_call_provider(
        denied_response,
        provider_name="tavily",
        safe_failure_message="Hotel search unavailable.",
        max_attempts=2,
    )

    assert result.success is False
    assert result.error_code == ErrorCode.ACCESS_DENIED
    assert result.trace_entry.transport == "direct_http"
    assert result.trace_entry.provider_status_code == 403
    assert result.trace_entry.provider_request_id == "request-123"
    assert result.trace_entry.retry_count == 0


@pytest.mark.asyncio
async def test_rate_limit_without_retry_after_is_not_retried():
    attempts = 0

    async def limited_response():
        nonlocal attempts
        attempts += 1
        raise ProviderRequestError("tavily", ErrorCode.RATE_LIMITED, status_code=429)

    result = await async_call_provider(
        limited_response,
        provider_name="tavily",
        safe_failure_message="Hotel search rate limited.",
        max_attempts=2,
    )

    assert result.success is False
    assert result.error_code == ErrorCode.RATE_LIMITED
    assert attempts == 1
    assert result.trace_entry.retry_count == 0


@pytest.mark.asyncio
async def test_forecast_is_grouped_by_destination_local_day():
    location = {"name": "Jaipur", "country": "IN", "lat": 26.9, "lon": 75.8}
    raw_forecast = {
        "city": {"timezone": 19800},
        "list": [
            {
                "dt": 1789423200,
                "main": {"temp": 24.0},
                "weather": [{"description": "clear sky"}],
                "pop": 0.1,
            },
            {
                "dt": 1789434000,
                "main": {"temp": 29.0},
                "weather": [{"description": "clear sky"}],
                "pop": 0.3,
            },
        ],
        "_meta": {"stage_timings_ms": {"http": 20}},
    }
    request_json = AsyncMock(return_value=raw_forecast)

    with patch.object(provider_adapters, "geocode_weather_location", AsyncMock(return_value=location)), \
         patch.object(provider_adapters.runtime, "request_json", request_json), \
         patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-key"}):
        forecast = await get_weather_forecast("Jaipur")

    assert len(forecast["forecast"]) == 1
    assert forecast["forecast"][0]["temperature_min_c"] == 24.0
    assert forecast["forecast"][0]["temperature_max_c"] == 29.0
    assert forecast["forecast"][0]["precipitation_probability_max"] == 30
    assert forecast["timezone_offset_seconds"] == 19800
