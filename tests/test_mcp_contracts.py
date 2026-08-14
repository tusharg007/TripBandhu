"""
test_mcp_contracts.py — TF2 Deterministic MCP capability contract, evidence,
provenance, and grounding tests.
"""

import asyncio
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from schemas import (
    CapabilityDefinition,
    CapabilityEvidence,
    CapabilityHealth,
    CapabilityTraceEntry,
    ErrorCode,
    EvidenceKind,
    SourceReference,
)
from capability_registry import (
    CAPABILITY_REGISTRY,
    SPECIALIST_ALLOWED_CAPABILITIES,
    validate_specialist_capability,
    normalize_tavily_results,
    normalize_weather_results,
    normalize_flight_results,
    format_grounded_evidence_context,
)
from provider_utils import async_call_provider


class TestCapabilityRegistry(unittest.TestCase):
    """Verify tool discovery, permission maps, and unknown tool rejection."""

    def test_allowed_capabilities_registered(self):
        """All core capabilities must be present in the registry."""
        required_caps = {
            "FLIGHT_ROUTE_SEARCH",
            "AIRPORT_LOOKUP",
            "AIRLINE_LOOKUP",
            "HOTEL_WEB_RESEARCH",
            "WEATHER_CURRENT",
            "WEATHER_FORECAST",
        }
        for cap in required_caps:
            self.assertIn(cap, CAPABILITY_REGISTRY)
            self.assertIsInstance(CAPABILITY_REGISTRY[cap], CapabilityDefinition)

    def test_specialist_permission_validation(self):
        """Specialists are permitted only their designated capabilities."""
        # flight_agent
        self.assertEqual(
            validate_specialist_capability("flight_agent", "FLIGHT_ROUTE_SEARCH").server,
            "aviationstack",
        )
        self.assertEqual(
            validate_specialist_capability("flight_agent", "AIRPORT_LOOKUP").server,
            "aviationstack",
        )

        # hotel_agent
        self.assertEqual(
            validate_specialist_capability("hotel_agent", "HOTEL_WEB_RESEARCH").server,
            "tavily",
        )

        # weather_agent
        self.assertEqual(
            validate_specialist_capability("weather_agent", "WEATHER_FORECAST").server,
            "weather",
        )

    def test_unknown_or_unauthorized_tool_rejected(self):
        """Arbitrary or cross-specialist tool execution is strictly rejected."""
        # Unknown capability name
        with self.assertRaises(ValueError) as ctx:
            validate_specialist_capability("flight_agent", "ARBITRARY_MODEL_TOOL")
        self.assertIn("not in the TripBandhu Capability Registry", str(ctx.exception))

        # Unauthorized specialist mapping (hotel_agent cannot execute aviationstack)
        with self.assertRaises(ValueError) as ctx:
            validate_specialist_capability("hotel_agent", "FLIGHT_ROUTE_SEARCH")
        self.assertIn("not permitted to execute capability", str(ctx.exception))


class TestTavilyNormalization(unittest.TestCase):
    """Verify Tavily web search evidence normalization and source preservation."""

    def test_tavily_sources_preserved_and_bounded(self):
        """Tavily search results extract sources and are bounded to top results."""
        raw_mock = {
            "results": [
                {"title": f"Hotel {i}", "url": f"https://example.com/hotel{i}", "content": f"Description {i}"}
                for i in range(15)  # 15 raw results
            ]
        }

        evidence = normalize_tavily_results(raw_mock, query="Tokyo luxury hotels", latency_ms=120)

        self.assertEqual(evidence.capability, "HOTEL_WEB_RESEARCH")
        self.assertEqual(evidence.evidence_kind, EvidenceKind.WEB_SOURCE)
        self.assertEqual(evidence.status, CapabilityHealth.AVAILABLE)
        self.assertEqual(len(evidence.data), 6, "Must be bounded to top 6 results")
        self.assertEqual(len(evidence.sources), 6)
        self.assertEqual(evidence.sources[0].url, "https://example.com/hotel0")
        self.assertIn("https://example.com/hotel0", evidence.summary)


class TestWeatherNormalization(unittest.TestCase):
    """Verify Weather MCP structured normalization and degradation."""

    def test_weather_evidence_normalized(self):
        """Current observations and forecast are cleanly structured into typed evidence."""
        curr_raw = {"city": "Tokyo", "temperature_c": 18.5, "condition": "clear sky"}
        fore_raw = {"city": "Tokyo", "forecast": [{"datetime": "2026-04-01", "temp": 19}]}

        evidence = normalize_weather_results(curr_raw, fore_raw, city="Tokyo", latency_ms=85)

        self.assertEqual(evidence.capability, "WEATHER_FORECAST")
        self.assertEqual(evidence.evidence_kind, EvidenceKind.LIVE_PROVIDER)
        self.assertEqual(evidence.status, CapabilityHealth.AVAILABLE)
        self.assertEqual(len(evidence.sources), 1)
        self.assertEqual(evidence.sources[0].provider, "OpenWeather API")
        self.assertIn("Tokyo", evidence.summary)


class TestFlightNormalizationAndPricingTruth(unittest.TestCase):
    """Verify AviationStack normalization and pricing truth."""

    def test_flight_evidence_normalized(self):
        """Route, airport, and airline data are normalized without claiming live fares."""
        routes_raw = [{"airline": "ANA", "flight_number": "NH828", "departure": "DEL", "arrival": "HND"}]
        airports_raw = [{"iata": "HND", "name": "Haneda Airport"}]
        airlines_raw = [{"iata": "NH", "name": "All Nippon Airways"}]

        evidence = normalize_flight_results(
            routes_data=routes_raw,
            airports_data=airports_raw,
            airlines_data=airlines_raw,
            destination="Tokyo",
            origin="Delhi",
            latency_ms=150,
        )

        self.assertEqual(evidence.capability, "FLIGHT_ROUTE_SEARCH")
        self.assertEqual(evidence.evidence_kind, EvidenceKind.LIVE_PROVIDER)
        self.assertEqual(evidence.status, CapabilityHealth.AVAILABLE)
        self.assertIn("Delhi -> Tokyo", evidence.summary)
        self.assertIn("NH828", evidence.summary)


class TestGroundingLabelsAndContextPacking(unittest.TestCase):
    """Verify that evidence context formatting tags provenance kinds distinctly."""

    def test_grounded_context_packager(self):
        """format_grounded_evidence_context explicitly labels LIVE_PROVIDER vs WEB_SOURCE vs UNAVAILABLE."""
        evidence_store = {
            "WEATHER_FORECAST": CapabilityEvidence(
                capability="WEATHER_FORECAST",
                provider="weather",
                tool_name="get_current_weather",
                evidence_kind=EvidenceKind.LIVE_PROVIDER,
                status=CapabilityHealth.AVAILABLE,
                summary="Temperature: 20C, Sunny",
                sources=[SourceReference(provider="OpenWeather", title="Live Tokyo Weather")],
            ).model_dump(),
            "HOTEL_WEB_RESEARCH": CapabilityEvidence(
                capability="HOTEL_WEB_RESEARCH",
                provider="tavily",
                tool_name="tavily_search",
                evidence_kind=EvidenceKind.WEB_SOURCE,
                status=CapabilityHealth.AVAILABLE,
                summary="Hotel Gracery Shinjuku near JR Station",
                sources=[SourceReference(provider="Tavily", title="Tokyo Hotels", url="https://example.com")],
            ).model_dump(),
            "FLIGHT_ROUTE_SEARCH": CapabilityEvidence(
                capability="FLIGHT_ROUTE_SEARCH",
                provider="aviationstack",
                tool_name="list_routes",
                evidence_kind=EvidenceKind.UNAVAILABLE,
                status=CapabilityHealth.UNAVAILABLE,
                summary="Live flight data temporarily unavailable.",
            ).model_dump(),
        }

        context = format_grounded_evidence_context(evidence_store)

        self.assertIn("[LIVE_PROVIDER] WEATHER_FORECAST", context)
        self.assertIn("[WEB_SOURCE] HOTEL_WEB_RESEARCH", context)
        self.assertIn("[UNAVAILABLE] FLIGHT_ROUTE_SEARCH", context)
        self.assertIn("https://example.com", context)


class TestProviderTraceAndCancellation(unittest.TestCase):
    """Verify capability trace capture, secret suppression, and cancellation propagation."""

    def test_trace_entry_captured_without_secrets(self):
        """async_call_provider produces clean trace entry without credentials."""
        async def dummy_coro():
            return {"status": "ok"}

        result, trace = asyncio.run(
            async_call_provider(
                dummy_coro,
                provider_name="tavily",
                safe_failure_message="Failed",
                specialist="hotel_agent",
                capability="HOTEL_WEB_RESEARCH",
                server="tavily",
                tool_name="tavily_search",
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(trace.specialist, "hotel_agent")
        self.assertEqual(trace.capability, "HOTEL_WEB_RESEARCH")
        self.assertEqual(trace.status, "AVAILABLE")
        self.assertEqual(trace.source_count, 1)

        # Ensure public representation has no secrets
        pub = trace.to_public_dict()
        self.assertNotIn("api_key", str(pub).lower())
        self.assertNotIn("token", str(pub).lower())
        self.assertNotIn("password", str(pub).lower())

    def test_cancellation_propagates_cleanly(self):
        """async_call_provider must cleanly raise asyncio.CancelledError when cancelled."""
        async def slow_coro():
            await asyncio.sleep(10)
            return "done"

        async def run_with_cancel():
            task = asyncio.create_task(
                async_call_provider(
                    slow_coro,
                    provider_name="tavily",
                    safe_failure_message="Failed",
                    specialist="test",
                    capability="test_cap",
                )
            )
            await asyncio.sleep(0.05)
            task.cancel()
            await task

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(run_with_cancel())


if __name__ == "__main__":
    unittest.main()
