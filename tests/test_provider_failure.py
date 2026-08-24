
"""TF1: Provider failure, timeout, and grounding contract tests."""
import asyncio
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.messages import AIMessage

from schemas import ErrorCode
from provider_utils import async_call_provider, _classify_exception


class ProviderFailureClassificationTest(unittest.TestCase):

    def test_timeout_classified(self):
        code = _classify_exception(asyncio.TimeoutError())
        self.assertEqual(code, ErrorCode.TIMEOUT)

    def test_uvx_classified_as_auth(self):
        code = _classify_exception(RuntimeError("uvx was not found. Install uv"))
        self.assertEqual(code, ErrorCode.AUTH_CONFIGURATION)

    def test_missing_api_key_classified_as_auth(self):
        code = _classify_exception(RuntimeError("API key is missing"))
        self.assertEqual(code, ErrorCode.AUTH_CONFIGURATION)

    def test_connection_error_classified_as_unavailable(self):
        code = _classify_exception(ConnectionError("Connection refused"))
        self.assertEqual(code, ErrorCode.UNAVAILABLE)

    def test_generic_error_classified_as_internal(self):
        code = _classify_exception(ValueError("Some internal error"))
        self.assertEqual(code, ErrorCode.INTERNAL)


class AsyncCallProviderTest(unittest.IsolatedAsyncioTestCase):

    async def test_success_returns_ok(self):
        result = await async_call_provider(
            lambda: asyncio.coroutine(lambda: "data")(),
            provider_name="tavily",
            safe_failure_message="Unavailable",
        )
        # Coroutine might work differently; use a real coroutine
        async def _good():
            return "data"

        result = await async_call_provider(
            _good,
            provider_name="tavily",
            safe_failure_message="Unavailable",
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data, "data")

    async def test_auth_error_not_retried(self):
        call_count = 0

        async def _fail():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("uvx was not found. Install uv")

        result = await async_call_provider(
            _fail,
            provider_name="aviation",
            safe_failure_message="Flight unavailable",
            max_attempts=3,
        )
        self.assertFalse(result.success)
        self.assertEqual(call_count, 1, "AUTH_CONFIGURATION errors must not be retried")
        self.assertEqual(result.error_code, ErrorCode.AUTH_CONFIGURATION)

    async def test_safe_message_returned_on_failure(self):
        async def _fail():
            raise ConnectionError("Connection refused")

        result = await async_call_provider(
            _fail,
            provider_name="tavily",
            safe_failure_message="Service temporarily unavailable",
        )
        self.assertFalse(result.success)
        self.assertEqual(result.safe_message, "Service temporarily unavailable")

    async def test_embedded_mcp_429_is_classified_as_rate_limit(self):
        """A text MCP response containing HTTP 429 is a failure, not successful data."""
        call_count = 0

        async def _embedded_429():
            nonlocal call_count
            call_count += 1
            return [
                {
                    "type": "text",
                    "text": (
                        '{"error":"Search failed","status":429,'
                        '"detail":{"error":"Your request has been blocked due to excessive requests."}}'
                    ),
                }
            ]

        result = await async_call_provider(
            _embedded_429,
            provider_name="tavily",
            safe_failure_message="Hotel search temporarily unavailable",
            specialist="hotel_agent",
            capability="HOTEL_WEB_RESEARCH",
            max_attempts=1,
        )

        self.assertFalse(result.success)
        self.assertEqual(call_count, 1)
        self.assertEqual(result.error_code, ErrorCode.RATE_LIMITED)
        self.assertEqual(result.trace_entry.error_code, ErrorCode.RATE_LIMITED.value)
        self.assertEqual(result.trace_entry.status, "UNAVAILABLE")


class FlightAgentDegradationTest(unittest.TestCase):

    def test_flight_agent_failure_sets_degraded(self):
        """Flight agent must degrade gracefully and not leak runtime errors."""
        import backend
        state = {
            "user_query": "Plan a 7-day trip to Tokyo",
            "selected_agents": ["flight_agent", "itinerary_agent"],
            "specialist_statuses": {"flight_agent": "SELECTED"},
            "llm_calls": 0,
        }

        async def _fail_aviation(tool_name):
            raise RuntimeError("uvx was not found. Install uv")

        with patch.object(backend, 'aviation_mcp_call', side_effect=_fail_aviation):
            result = asyncio.run(backend.flight_agent(state))

        self.assertEqual(result["specialist_statuses"]["flight_agent"], "DEGRADED")
        self.assertFalse(result["flight_data_available"])
        self.assertNotIn("uvx was not found", result["flight_results"])
        self.assertNotIn("RuntimeError", result["flight_results"])
        self.assertIn("Live flight data is temporarily unavailable", result["flight_results"])

    def test_hotel_agent_counts_presentation_llm(self):
        """hotel_agent counts the user-facing evidence presentation pass."""
        import backend
        state = {
            "user_query": "Plan a trip to Dubai",
            "specialist_statuses": {},
            "llm_calls": 3,
        }

        async def _mock_tavily(query):
            return "Hotel results from Tavily"

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="## Dubai hotels\nHotel results from Tavily"))

        with patch.object(backend, 'tavily_mcp_search', side_effect=_mock_tavily), \
             patch.object(backend, '_llm_hotel', mock_llm):
            result = asyncio.run(backend.hotel_agent(state))

        self.assertEqual(result["llm_calls"], 4)
        self.assertEqual(result["specialist_statuses"]["hotel_agent"], "COMPLETED")

    def test_hotel_agent_reports_tavily_rate_limit_without_calling_llm(self):
        import backend

        state = {
            "user_query": "Hotels in London",
            "trip_constraints": {"destination": "London", "travel_style": "mid-range"},
            "specialist_statuses": {},
            "llm_calls": 2,
            "llm_token_usage": {},
        }

        async def _mock_tavily(_query):
            return [
                {
                    "type": "text",
                    "text": (
                        '{"error":"Search failed","status":429,'
                        '"detail":{"error":"Your request has been blocked due to excessive requests. '
                        'Please reduce the rate of requests."}}'
                    ),
                }
            ]

        hotel_llm = MagicMock()
        hotel_llm.ainvoke = AsyncMock(side_effect=AssertionError("Hotel LLM must not run without evidence"))

        with patch.object(backend, "tavily_mcp_search", side_effect=_mock_tavily), \
             patch.object(backend, "_llm_hotel", hotel_llm):
            result = asyncio.run(backend.hotel_agent(state))

        self.assertEqual(result["specialist_statuses"]["hotel_agent"], "DEGRADED")
        self.assertFalse(result["hotel_data_available"])
        self.assertEqual(result["llm_calls"], 2)
        self.assertIn("temporarily rate-limited by Tavily", result["hotel_results"])
        self.assertIn("not a hotel-agent reasoning failure", result["hotel_results"])
        self.assertNotIn("No reliable hotel-specific sources", result["hotel_results"])
        hotel_llm.ainvoke.assert_not_awaited()

    def test_grounding_contract_maintained_on_degradation(self):
        """When flight data is unavailable, final_agent must not fabricate flight info."""
        import backend
        state = {
            "user_query": "Plan a 5-day trip to Japan",
            "trip_constraints": {"destination": "Japan"},
            "flight_results": backend.FLIGHT_UNAVAILABLE_MESSAGE,
            "hotel_results": "Good hotels in Tokyo",
            "weather_results": "Clear skies",
            "budget_results": "INR 1.5 lakh estimated",
            "itinerary": "Day 1: Arrive in Tokyo",
            "approved": True,
            "human_feedback": "",
            "review_iteration": 0,
            "llm_calls": 4,
            "llm_token_usage": {},
        }
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Live flight information was unavailable for this run. Verify schedules and fares with an airline or booking provider before booking. Hotels: APA Hotel...")
        )

        with patch.object(backend, '_llm_final', mock_llm):
            result = asyncio.run(backend.final_agent(state))

        # Must not contain fabricated prices
        response = result["final_response"]
        self.assertNotIn("INR 45,000", response)
        self.assertNotIn("\u20b945,000", response)


if __name__ == "__main__":
    unittest.main()
