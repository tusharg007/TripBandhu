"""Direct, typed async adapters for TripBandhu's fixed external providers.

The adapters deliberately keep credentials out of exception text and logs.  They
also provide application-scoped connection pooling, bounded concurrency,
single-flight request coalescing, short-lived validated caches, and a small
circuit breaker so one unhealthy provider cannot monopolize a user request.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

from agent_config import (
    PROVIDER_CACHE_TTL_SECONDS,
    PROVIDER_CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    PROVIDER_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    PROVIDER_CONCURRENCY_LIMITS,
    PROVIDER_TIMEOUT_SECONDS,
)
from project_config import load_project_env
from schemas import ErrorCode


load_project_env()


class ProviderRequestError(RuntimeError):
    """Sanitized provider failure safe to classify and log."""

    def __init__(
        self,
        provider: str,
        error_code: ErrorCode,
        *,
        status_code: int | None = None,
        provider_code: str | None = None,
        retry_after: float | None = None,
        provider_request_id: str | None = None,
    ) -> None:
        self.provider = provider
        self.error_code = error_code
        self.status_code = status_code
        self.provider_code = provider_code
        self.retry_after = retry_after
        self.provider_request_id = provider_request_id
        self.transport = "direct_http"
        parts = [provider, error_code.value]
        if status_code is not None:
            parts.append(f"HTTP {status_code}")
        if provider_code:
            parts.append(f"code {provider_code[:48]}")
        super().__init__("Provider request failed (" + ", ".join(parts) + ")")


@dataclass
class _CacheEntry:
    expires_at: float
    value: Any


@dataclass
class _CircuitState:
    failures: int = 0
    open_until: float = 0.0


def _safe_retry_after(value: str | None) -> float | None:
    try:
        seconds = float(value or "")
    except (TypeError, ValueError):
        return None
    return max(0.0, min(seconds, 3600.0))


def _status_error(status_code: int) -> ErrorCode:
    if status_code == 401:
        return ErrorCode.AUTH_CONFIGURATION
    if status_code == 403:
        return ErrorCode.ACCESS_DENIED
    if status_code == 429:
        return ErrorCode.RATE_LIMITED
    if status_code in {408, 504}:
        return ErrorCode.TIMEOUT
    if status_code >= 500:
        return ErrorCode.UNAVAILABLE
    return ErrorCode.INVALID_RESPONSE


def _request_id(headers: httpx.Headers) -> str | None:
    for name in ("x-request-id", "request-id", "x-correlation-id", "cf-ray"):
        value = str(headers.get(name) or "").strip()
        if value:
            return value[:96]
    return None


def _attach_meta(payload: Any, meta: dict[str, Any]) -> Any:
    if isinstance(payload, dict):
        result = copy.deepcopy(payload)
        existing = result.get("_meta") if isinstance(result.get("_meta"), dict) else {}
        result["_meta"] = {**existing, **meta}
        return result
    return {"data": copy.deepcopy(payload), "_meta": meta}


class ProviderRuntime:
    """Own shared HTTP resources for one application event loop."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock: asyncio.Lock | None = None
        self._cache: dict[str, _CacheEntry] = {}
        self._inflight: dict[str, asyncio.Task[Any]] = {}
        self._circuits: dict[str, _CircuitState] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        if self._client is not None and not self._client.is_closed and self._loop is loop:
            return
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._loop = loop
        self._lock = asyncio.Lock()
        self._cache.clear()
        self._inflight.clear()
        self._circuits.clear()
        self._semaphores = {
            provider: asyncio.Semaphore(max(1, int(limit)))
            for provider, limit in PROVIDER_CONCURRENCY_LIMITS.items()
        }
        self._client = httpx.AsyncClient(
            follow_redirects=True,
            limits=httpx.Limits(max_connections=12, max_keepalive_connections=6),
            headers={"User-Agent": "TripBandhu/1.1 provider-adapter"},
        )

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        self._loop = None
        self._lock = None
        self._inflight.clear()

    async def request_json(
        self,
        *,
        provider: str,
        method: str,
        url: str,
        cache_key: str,
        ttl_seconds: float,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
        validator: Callable[[Any], Any] | None = None,
    ) -> Any:
        await self.start()
        assert self._lock is not None
        now = time.monotonic()

        async with self._lock:
            cached = self._cache.get(cache_key)
            if cached and cached.expires_at > now:
                return _attach_meta(cached.value, {"cache_status": "hit"})
            if cached:
                self._cache.pop(cache_key, None)

            circuit = self._circuits.setdefault(provider, _CircuitState())
            if circuit.open_until > now:
                raise ProviderRequestError(provider, ErrorCode.CIRCUIT_OPEN)

            existing = self._inflight.get(cache_key)
            if existing is None:
                existing = asyncio.create_task(
                    self._perform_and_cache(
                        cache_key=cache_key,
                        ttl_seconds=ttl_seconds,
                        provider=provider,
                        method=method,
                        url=url,
                        params=params,
                        headers=headers,
                        json_body=json_body,
                        timeout_seconds=timeout_seconds,
                        validator=validator,
                    )
                )
                self._inflight[cache_key] = existing
                owner = True
            else:
                owner = False

        # A caller cancellation must not cancel the one provider request shared by
        # other waiters. The task itself publishes the cache and removes its
        # in-flight entry, so an abandoned creator cannot cause duplicate calls.
        payload = await asyncio.shield(existing)
        return _attach_meta(payload, {"cache_status": "miss" if owner else "coalesced"})

    async def _perform_and_cache(
        self,
        *,
        cache_key: str,
        ttl_seconds: float,
        **request_kwargs: Any,
    ) -> Any:
        try:
            payload = await self._perform_request(**request_kwargs)
            assert self._lock is not None
            async with self._lock:
                self._cache[cache_key] = _CacheEntry(
                    expires_at=time.monotonic() + max(0.0, ttl_seconds),
                    value=copy.deepcopy(payload),
                )
            return payload
        finally:
            assert self._lock is not None
            current = asyncio.current_task()
            async with self._lock:
                if self._inflight.get(cache_key) is current:
                    self._inflight.pop(cache_key, None)

    async def _perform_request(
        self,
        *,
        provider: str,
        method: str,
        url: str,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
        json_body: dict[str, Any] | None,
        timeout_seconds: float | None,
        validator: Callable[[Any], Any] | None,
    ) -> Any:
        assert self._client is not None
        semaphore = self._semaphores.setdefault(provider, asyncio.Semaphore(2))
        started = time.monotonic()
        try:
            async with semaphore:
                response = await self._client.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                    json=json_body,
                    timeout=timeout_seconds or PROVIDER_TIMEOUT_SECONDS.get(provider, 15.0),
                )
            network_ms = int((time.monotonic() - started) * 1000)
        except httpx.TimeoutException as exc:
            await self._record_failure(provider)
            raise ProviderRequestError(provider, ErrorCode.TIMEOUT) from None
        except httpx.HTTPError as exc:
            await self._record_failure(provider)
            raise ProviderRequestError(provider, ErrorCode.UNAVAILABLE) from None

        if response.status_code >= 400:
            await self._record_failure(provider)
            raise ProviderRequestError(
                provider,
                _status_error(response.status_code),
                status_code=response.status_code,
                retry_after=_safe_retry_after(response.headers.get("retry-after")),
                provider_request_id=_request_id(response.headers),
            )

        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError):
            await self._record_failure(provider)
            raise ProviderRequestError(
                provider,
                ErrorCode.INVALID_RESPONSE,
                status_code=response.status_code,
            ) from None

        embedded_error = payload.get("error") if isinstance(payload, dict) else None
        if embedded_error:
            provider_code = ""
            if isinstance(embedded_error, dict):
                provider_code = str(embedded_error.get("code") or embedded_error.get("type") or "")
            code_text = provider_code.casefold()
            error_code = (
                ErrorCode.ACCESS_DENIED
                if any(term in code_text for term in ("access", "plan", "function_access"))
                else ErrorCode.AUTH_CONFIGURATION
                if any(term in code_text for term in ("key", "auth", "invalid_access"))
                else ErrorCode.INVALID_RESPONSE
            )
            await self._record_failure(provider)
            raise ProviderRequestError(
                provider,
                error_code,
                status_code=response.status_code,
                provider_code=provider_code,
                provider_request_id=_request_id(response.headers),
            )

        if validator is not None:
            try:
                payload = validator(payload)
            except ProviderRequestError:
                await self._record_failure(provider)
                raise
            except Exception:
                await self._record_failure(provider)
                raise ProviderRequestError(provider, ErrorCode.INVALID_RESPONSE) from None

        await self._record_success(provider)
        return _attach_meta(
            payload,
            {
                "transport": "direct_http",
                "provider_status_code": response.status_code,
                "provider_request_id": _request_id(response.headers),
                "stage_timings_ms": {"http": network_ms},
            },
        )

    async def _record_success(self, provider: str) -> None:
        assert self._lock is not None
        async with self._lock:
            self._circuits[provider] = _CircuitState()

    async def _record_failure(self, provider: str) -> None:
        assert self._lock is not None
        async with self._lock:
            state = self._circuits.setdefault(provider, _CircuitState())
            state.failures += 1
            if state.failures >= PROVIDER_CIRCUIT_BREAKER_FAILURE_THRESHOLD:
                state.open_until = time.monotonic() + PROVIDER_CIRCUIT_BREAKER_COOLDOWN_SECONDS


runtime = ProviderRuntime()


def _env_key(*names: str) -> str:
    import os

    for name in names:
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
    raise ProviderRequestError(names[0], ErrorCode.AUTH_CONFIGURATION)


def _cache_key(provider: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"{provider}:{hashlib.sha256(encoded).hexdigest()}"


def _validate_tavily(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ProviderRequestError("tavily", ErrorCode.INVALID_RESPONSE)
    results: list[dict[str, Any]] = []
    for item in payload["results"]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        parsed = urlparse(url)
        title = str(item.get("title") or "").strip()
        content = str(item.get("content") or item.get("snippet") or "").strip()
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or not title or len(content) < 30:
            continue
        results.append(
            {
                "title": title[:300],
                "url": url,
                "content": content[:2000],
                "score": item.get("score"),
            }
        )
    if not results:
        raise ProviderRequestError("tavily", ErrorCode.INVALID_RESPONSE)
    return {"query": str(payload.get("query") or ""), "results": results}


async def search_tavily(query: str) -> dict[str, Any]:
    api_key = _env_key("TAVILY_API_KEY")
    body = {
        "query": query,
        "topic": "general",
        "search_depth": "basic",
        "max_results": 8,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }
    return await runtime.request_json(
        provider="tavily",
        method="POST",
        url="https://api.tavily.com/search",
        cache_key=_cache_key("tavily", body),
        ttl_seconds=PROVIDER_CACHE_TTL_SECONDS["tavily"],
        headers={"Authorization": f"Bearer {api_key}"},
        json_body=body,
        validator=_validate_tavily,
    )


def _validate_aviation(
    payload: Any,
    *,
    expected_departure: str = "",
    expected_arrival: str = "",
) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ProviderRequestError("aviation", ErrorCode.INVALID_RESPONSE)
    records: list[dict[str, Any]] = []
    for item in payload["data"]:
        if not isinstance(item, dict):
            continue
        departure = item.get("departure") if isinstance(item.get("departure"), dict) else {}
        arrival = item.get("arrival") if isinstance(item.get("arrival"), dict) else {}
        if not departure.get("iata") or not arrival.get("iata"):
            continue
        if expected_departure and str(departure.get("iata")).upper() != expected_departure:
            continue
        if expected_arrival and str(arrival.get("iata")).upper() != expected_arrival:
            continue
        records.append(item)
    if not records:
        raise ProviderRequestError("aviation", ErrorCode.INVALID_RESPONSE)
    return {"data": records[:10], "pagination": payload.get("pagination") or {}}


async def search_aviation_flights(tool_args: dict[str, Any] | None = None) -> dict[str, Any]:
    api_key = _env_key("AVIATION_STACK_API_KEY", "AVIATIONSTACK_API_KEY", "AVIATIONSTACK_KEY")
    requested = dict(tool_args or {})
    params: dict[str, Any] = {"access_key": api_key, "limit": 10}
    for name in ("dep_iata", "arr_iata", "flight_date"):
        value = str(requested.get(name) or "").strip()
        if value:
            params[name] = value.upper() if name.endswith("iata") else value
    safe_key_params = {key: value for key, value in params.items() if key != "access_key"}
    expected_departure = str(safe_key_params.get("dep_iata") or "")
    expected_arrival = str(safe_key_params.get("arr_iata") or "")
    return await runtime.request_json(
        provider="aviation",
        method="GET",
        url="https://api.aviationstack.com/v1/flights",
        cache_key=_cache_key("aviation", safe_key_params),
        ttl_seconds=PROVIDER_CACHE_TTL_SECONDS["aviation"],
        params=params,
        validator=lambda payload: _validate_aviation(
            payload,
            expected_departure=expected_departure,
            expected_arrival=expected_arrival,
        ),
    )


def _validate_geocode(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise ProviderRequestError("weather", ErrorCode.INVALID_RESPONSE)
    first = payload[0]
    try:
        latitude = float(first["lat"])
        longitude = float(first["lon"])
    except (KeyError, TypeError, ValueError):
        raise ProviderRequestError("weather", ErrorCode.INVALID_RESPONSE) from None
    return {
        "name": str(first.get("name") or "").strip(),
        "state": str(first.get("state") or "").strip(),
        "country": str(first.get("country") or "").strip(),
        "lat": latitude,
        "lon": longitude,
    }


async def geocode_weather_location(city: str) -> dict[str, Any]:
    api_key = _env_key("OPENWEATHER_API_KEY", "OPENWEATHER_KEY")
    query = " ".join(str(city or "").split()).strip()
    if not query:
        raise ProviderRequestError("weather", ErrorCode.INVALID_RESPONSE)
    safe_params = {"q": query, "limit": 1}
    return await runtime.request_json(
        provider="weather",
        method="GET",
        url="https://api.openweathermap.org/geo/1.0/direct",
        cache_key=_cache_key("weather-geocode", safe_params),
        ttl_seconds=PROVIDER_CACHE_TTL_SECONDS["geocode"],
        params={**safe_params, "appid": api_key},
        validator=_validate_geocode,
    )


def _location_label(location: dict[str, Any]) -> str:
    return ", ".join(
        value for value in (location.get("name"), location.get("state"), location.get("country")) if value
    )


def _validate_current(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ProviderRequestError("weather", ErrorCode.INVALID_RESPONSE)
    weather = payload.get("weather")
    main = payload.get("main")
    if not isinstance(weather, list) or not weather or not isinstance(main, dict):
        raise ProviderRequestError("weather", ErrorCode.INVALID_RESPONSE)
    return payload


def _validate_forecast(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("list"), list) or not payload["list"]:
        raise ProviderRequestError("weather", ErrorCode.INVALID_RESPONSE)
    return payload


async def get_current_weather(city: str) -> dict[str, Any]:
    api_key = _env_key("OPENWEATHER_API_KEY", "OPENWEATHER_KEY")
    geocode_started = time.monotonic()
    location = await geocode_weather_location(city)
    geocode_ms = int((time.monotonic() - geocode_started) * 1000)
    params = {"lat": location["lat"], "lon": location["lon"], "units": "metric"}
    payload = await runtime.request_json(
        provider="weather",
        method="GET",
        url="https://api.openweathermap.org/data/2.5/weather",
        cache_key=_cache_key("weather-current", params),
        ttl_seconds=PROVIDER_CACHE_TTL_SECONDS["weather_current"],
        params={**params, "appid": api_key},
        validator=_validate_current,
    )
    weather = payload["weather"][0]
    meta = dict(payload.get("_meta") or {})
    timings = dict(meta.get("stage_timings_ms") or {})
    timings["geocode"] = geocode_ms
    meta["stage_timings_ms"] = timings
    return {
        "city": _location_label(location) or city,
        "coordinates": {"lat": location["lat"], "lon": location["lon"]},
        "country_code": location.get("country"),
        "observed_at": datetime.fromtimestamp(int(payload.get("dt") or 0), tz=timezone.utc).isoformat()
        if payload.get("dt") else None,
        "timezone_offset_seconds": payload.get("timezone"),
        "temperature_c": payload["main"].get("temp"),
        "feels_like_c": payload["main"].get("feels_like"),
        "humidity": payload["main"].get("humidity"),
        "condition": weather.get("description"),
        "wind_speed": (payload.get("wind") or {}).get("speed"),
        "_meta": meta,
    }


async def get_weather_forecast(city: str) -> dict[str, Any]:
    api_key = _env_key("OPENWEATHER_API_KEY", "OPENWEATHER_KEY")
    geocode_started = time.monotonic()
    location = await geocode_weather_location(city)
    geocode_ms = int((time.monotonic() - geocode_started) * 1000)
    params = {"lat": location["lat"], "lon": location["lon"], "units": "metric"}
    payload = await runtime.request_json(
        provider="weather",
        method="GET",
        url="https://api.openweathermap.org/data/2.5/forecast",
        cache_key=_cache_key("weather-forecast", params),
        ttl_seconds=PROVIDER_CACHE_TTL_SECONDS["weather_forecast"],
        params={**params, "appid": api_key},
        validator=_validate_forecast,
    )
    timezone_offset = int((payload.get("city") or {}).get("timezone") or 0)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in payload["list"]:
        if not isinstance(entry, dict) or not entry.get("dt"):
            continue
        local_dt = datetime.fromtimestamp(
            int(entry["dt"]), tz=timezone.utc
        ) + timedelta(seconds=timezone_offset)
        main = entry.get("main") if isinstance(entry.get("main"), dict) else {}
        weather = entry.get("weather") if isinstance(entry.get("weather"), list) else []
        grouped.setdefault(local_dt.date().isoformat(), []).append(
            {
                "temperature_c": main.get("temp"),
                "condition": (weather[0] if weather else {}).get("description"),
                "precipitation_probability": entry.get("pop"),
            }
        )

    daily: list[dict[str, Any]] = []
    for date_key, entries in sorted(grouped.items()):
        temperatures = [float(item["temperature_c"]) for item in entries if item["temperature_c"] is not None]
        conditions = [str(item["condition"]) for item in entries if item.get("condition")]
        precipitation = [float(item["precipitation_probability"]) for item in entries if item.get("precipitation_probability") is not None]
        if not temperatures:
            continue
        daily.append(
            {
                "date": date_key,
                "temperature_min_c": round(min(temperatures), 1),
                "temperature_max_c": round(max(temperatures), 1),
                "condition": Counter(conditions).most_common(1)[0][0] if conditions else "Not reported",
                "precipitation_probability_max": round(max(precipitation) * 100) if precipitation else None,
                "forecast_slots": len(entries),
            }
        )
    if not daily:
        raise ProviderRequestError("weather", ErrorCode.INVALID_RESPONSE)
    meta = dict(payload.get("_meta") or {})
    timings = dict(meta.get("stage_timings_ms") or {})
    timings["geocode"] = geocode_ms
    meta["stage_timings_ms"] = timings
    return {
        "city": _location_label(location) or city,
        "coordinates": {"lat": location["lat"], "lon": location["lon"]},
        "country_code": location.get("country"),
        "timezone_offset_seconds": timezone_offset,
        "forecast": daily,
        "_meta": meta,
    }


async def start_provider_runtime() -> None:
    await runtime.start()


async def close_provider_runtime() -> None:
    await runtime.close()
