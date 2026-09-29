import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

import backend
from provider_adapters import ProviderRequestError, _validate_geocode
from schemas import CapabilityEvidence, EvidenceKind


def test_budget_prompt_uses_compact_price_relevant_evidence_only():
    routes = [
        {
            "flight_date": "2026-09-30",
            "airline": {"name": "Japan Airlines"},
            "departure": {"iata": "DEL", "raw": "do-not-copy-raw-flight-fields" * 100},
            "arrival": {"iata": "NRT"},
        }
        for _ in range(10)
    ]
    hotel_items = [
        {"title": f"Tokyo hotel source {index}", "snippet": "Mid-range room reference. " * 100}
        for index in range(6)
    ]
    evidence = {
        "FLIGHT_ROUTE_SEARCH": CapabilityEvidence(
            capability="FLIGHT_ROUTE_SEARCH", provider="aviationstack", tool_name="flights",
            data={"origin": "Delhi", "destination": "Japan", "routes": {"data": routes}},
            summary="raw-flight-summary " * 2000,
        ).model_dump(),
        "HOTEL_WEB_RESEARCH": CapabilityEvidence(
            capability="HOTEL_WEB_RESEARCH", provider="tavily", tool_name="search",
            evidence_kind=EvidenceKind.WEB_SOURCE, data=hotel_items,
            summary="raw-hotel-summary " * 2000,
        ).model_dump(),
        "WEATHER_FORECAST": CapabilityEvidence(
            capability="WEATHER_FORECAST", provider="weather", tool_name="forecast",
            summary="Weather does not belong in a budget prompt."
        ).model_dump(),
    }
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="Complete compact budget"))
    state = {
        "user_query": "Plan a 7-day mid-range Japan trip from Delhi",
        "trip_constraints": {"origin": "Delhi", "destination": "Japan", "duration": "7 days"},
        "evidence_store": evidence, "specialist_statuses": {}, "llm_calls": 0, "llm_token_usage": {},
    }

    with patch.object(backend, "_llm_budget", llm):
        result = asyncio.run(backend.budget_agent(state))

    prompt = llm.ainvoke.await_args.args[0][1].content
    assert len(prompt) < 6500
    assert "10 route-matched status records" in prompt
    assert "do-not-copy-raw-flight-fields" not in prompt
    assert "raw-flight-summary" not in prompt
    assert "Weather does not belong" not in prompt
    assert result["specialist_statuses"]["budget_agent"] == "COMPLETED"


def test_budget_rate_limit_returns_complete_deterministic_estimate():
    state = {
        "user_query": "Plan a 7-day mid-range trip from Delhi to Japan under ₹2.5 lakh",
        "trip_constraints": {
            "origin": "Delhi", "destination": "Japan", "duration": "7 days",
            "budget": "₹2.5 lakh", "travel_style": "mid-range",
        },
        "evidence_store": {}, "specialist_statuses": {}, "llm_calls": 4, "llm_token_usage": {},
    }
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=RuntimeError("rate limited"))

    with patch.object(backend, "_llm_budget", llm):
        result = asyncio.run(backend.budget_agent(state))

    output = result["budget_results"]
    assert result["specialist_statuses"]["budget_agent"] == "COMPLETED"
    assert "Planning estimate — not a live quotation" in output
    assert "₹250,000" in output
    assert "## 5. Budget assumptions and verification checklist" in output
    assert "salary" not in output.casefold()
    assert not output.rstrip().endswith("for")


def test_country_destination_is_qualified_for_weather_provider():
    seen = []

    async def current(city):
        seen.append(city)
        return {"city": "Tokyo, JP", "temperature_c": 20, "condition": "clear", "humidity": 50}

    async def forecast(city):
        seen.append(city)
        return {"city": "Tokyo, JP", "forecast": [{"date": "2026-09-30", "temperature_min_c": 17, "temperature_max_c": 23, "condition": "clear"}]}

    weather_llm = MagicMock()
    weather_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Tokyo is clear. Pack light layers."))
    state = {
        "user_query": "Weather for a Japan trip", "trip_constraints": {"destination": "Japan"},
        "specialist_statuses": {}, "evidence_store": {}, "capability_trace": [],
        "llm_calls": 0, "llm_token_usage": {},
    }
    with patch.object(backend, "weather_mcp_search", side_effect=current), \
         patch.object(backend, "forecast_mcp_search", side_effect=forecast), \
         patch.object(backend, "_llm_weather", weather_llm):
        result = asyncio.run(backend.weather_agent(state))

    assert seen == ["Tokyo,JP", "Tokyo,JP"]
    assert "Pennsylvania" not in result["weather_results"]
    assert result["specialist_statuses"]["weather_agent"] == "COMPLETED"


def test_geocoder_selects_requested_country_instead_of_first_same_name():
    payload = [
        {"name": "Japan", "state": "Pennsylvania", "country": "US", "lat": 40, "lon": -79},
        {"name": "Tokyo", "country": "JP", "lat": 35.6, "lon": 139.7},
    ]
    selected = _validate_geocode(payload, expected_country="JP")
    assert selected["name"] == "Tokyo"
    assert selected["country"] == "JP"


def test_geocoder_rejects_country_mismatch():
    try:
        _validate_geocode(
            [{"name": "Japan", "state": "Pennsylvania", "country": "US", "lat": 40, "lon": -79}],
            expected_country="JP",
        )
    except ProviderRequestError:
        return
    raise AssertionError("A same-named location in the wrong country must not be accepted")
