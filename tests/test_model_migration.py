"""
test_model_migration.py — Phase 1+3+7+8 regression tests for the LLM migration patch.

Covers:
- Deprecated model ID absence from runtime module strings
- Centralized factory configuration (max_retries=0)
- Weather agent destination priority (no LLM when supervisor provided destination)
- Structured output schema compatibility
- Degraded itinerary routing (Phase 6)
- LLM rate-limit classification (Phase 7)
- Token usage accumulation (Phase 8)
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage


class DeprecatedModelAbsenceTest(unittest.TestCase):
    """Phase 1: Deprecated model ID must not appear in any runtime module."""

    def test_deprecated_model_id_absent_from_backend(self):
        """llama-3.3-70b-versatile must not appear in runtime backend module strings."""
        import backend
        import inspect
        source = inspect.getsource(backend)
        self.assertNotIn(
            "llama-3.3-70b-versatile",
            source,
            "Deprecated model 'llama-3.3-70b-versatile' must not appear in backend.py",
        )

    def test_deprecated_model_id_absent_from_mcp_client(self):
        """llama-3.3-70b-versatile must not appear in mcp_client module strings."""
        import mcp_client
        import inspect
        source = inspect.getsource(mcp_client)
        self.assertNotIn(
            "llama-3.3-70b-versatile",
            source,
            "Deprecated model 'llama-3.3-70b-versatile' must not appear in mcp_client.py",
        )

    def test_llm_control_uses_env_model_id(self):
        """_llm_control must use the configured GROQ_CONTROL_MODEL, not the deprecated model."""
        import backend
        from agent_config import GROQ_CONTROL_MODEL
        # _llm_control is the base control instance
        model_name = getattr(backend._llm_control, "model_name", None)
        self.assertEqual(
            model_name,
            GROQ_CONTROL_MODEL,
            f"_llm_control model_name must be {GROQ_CONTROL_MODEL!r}, got {model_name!r}",
        )

    def test_llm_generation_uses_env_model_id(self):
        """_llm_generation must use the configured GROQ_GENERATION_MODEL."""
        import backend
        from agent_config import GROQ_GENERATION_MODEL
        model_name = getattr(backend._llm_generation, "model_name", None)
        self.assertEqual(
            model_name,
            GROQ_GENERATION_MODEL,
            f"_llm_generation model_name must be {GROQ_GENERATION_MODEL!r}, got {model_name!r}",
        )


class LLMFactoryConfigTest(unittest.TestCase):
    """Phase 1: make_groq_llm factory must configure max_retries=0."""

    def test_make_groq_llm_sets_max_retries_zero(self):
        """All ChatGroq instances must have max_retries=0 to prevent hidden SDK retries."""
        from mcp_client import make_groq_llm
        from agent_config import GROQ_CONTROL_MODEL
        llm = make_groq_llm(GROQ_CONTROL_MODEL)
        # max_retries is stored as groq_client attribute or directly
        mr = getattr(llm, "max_retries", None)
        self.assertEqual(mr, 0, f"make_groq_llm must set max_retries=0, got {mr}")

    def test_backend_control_instance_max_retries_zero(self):
        """_llm_control base instance must have max_retries=0."""
        import backend
        mr = getattr(backend._llm_control, "max_retries", None)
        self.assertEqual(mr, 0, "_llm_control must have max_retries=0")

    def test_backend_generation_instance_max_retries_zero(self):
        """_llm_generation base instance must have max_retries=0."""
        import backend
        mr = getattr(backend._llm_generation, "max_retries", None)
        self.assertEqual(mr, 0, "_llm_generation must have max_retries=0")


class WeatherDestinationPriorityTest(unittest.TestCase):
    """Phase 3: Weather agent must use supervisor destination without LLM call."""

    def test_no_llm_called_when_supervisor_destination_present(self):
        """weather_agent must skip extract_destination_async when trip_constraints.destination is set."""
        import backend
        state = {
            "user_query": "Weather in Kyoto for my trip",
            "specialist_statuses": {},
            "trip_constraints": {"destination": "Kyoto"},
            "llm_calls": 0,
            "llm_token_usage": {},
        }

        async def _weather_ok(city):
            return f"Sunny in {city}"

        async def _forecast_ok(city):
            return f"Clear for a week in {city}"

        with patch.object(backend, "extract_destination_async",
                          side_effect=AssertionError("LLM extraction called unexpectedly")), \
             patch.object(backend, "weather_mcp_search", side_effect=_weather_ok), \
             patch.object(backend, "forecast_mcp_search", side_effect=_forecast_ok):
            result = asyncio.run(backend.weather_agent(state))

        self.assertEqual(result["llm_calls"], 0, "No LLM call when destination is in constraints")
        self.assertTrue(result.get("weather_data_available", False))

    def test_llm_called_when_destination_absent(self):
        """weather_agent MUST call extract_destination_async when trip_constraints.destination is empty."""
        import backend
        state = {
            "user_query": "What is the weather like next week?",
            "specialist_statuses": {},
            "trip_constraints": {},
            "llm_calls": 0,
            "llm_token_usage": {},
        }
        called = []

        async def _extract(query):
            called.append(True)
            return "Kyoto"

        async def _weather_ok(city):
            return "Sunny"

        async def _forecast_ok(city):
            return "Clear"

        with patch.object(backend, "extract_destination_async", side_effect=_extract), \
             patch.object(backend, "weather_mcp_search", side_effect=_weather_ok), \
             patch.object(backend, "forecast_mcp_search", side_effect=_forecast_ok):
            result = asyncio.run(backend.weather_agent(state))

        self.assertTrue(called, "extract_destination_async must be called when destination is absent")
        self.assertEqual(result["llm_calls"], 1)


class RateLimitClassificationTest(unittest.TestCase):
    """Phase 7: llm_utils rate-limit classification tests."""

    def test_is_rate_limit_error_detects_429_message(self):
        from llm_utils import _is_rate_limit_error
        exc = RuntimeError("HTTP 429 Too Many Requests")
        self.assertTrue(_is_rate_limit_error(exc))

    def test_is_rate_limit_error_detects_rate_limit_message(self):
        from llm_utils import _is_rate_limit_error
        exc = Exception("rate limit exceeded, please wait")
        self.assertTrue(_is_rate_limit_error(exc))

    def test_is_rate_limit_error_rejects_non_429(self):
        from llm_utils import _is_rate_limit_error
        exc = ValueError("Invalid API key")
        self.assertFalse(_is_rate_limit_error(exc))

    def test_get_retry_after_from_message(self):
        from llm_utils import _get_retry_after
        exc = Exception("Rate limit exceeded. Retry-After: 5 seconds")
        ra = _get_retry_after(exc)
        self.assertEqual(ra, 5)

    def test_short_wait_rate_limit_triggers_retry(self):
        """Short Retry-After (<= 15s) should trigger one retry in invoke_llm."""
        from llm_utils import invoke_llm
        from agent_config import RATE_LIMIT_SHORT_WAIT_MAX_SECONDS

        call_count = 0

        class ShortRateLimit(Exception):
            """Simulated 429 with short Retry-After."""
            status_code = 429

        async def _failing_then_ok(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ShortRateLimit("rate limit exceeded, Retry-After: 1")
            return AIMessage(content="Success on retry")

        mock_llm = MagicMock()
        mock_llm.ainvoke = _failing_then_ok

        response, tokens = asyncio.run(invoke_llm(mock_llm, [], task_name="test", timeout_s=10.0))
        self.assertEqual(call_count, 2, "Short rate limit should trigger exactly one retry")
        self.assertEqual(response.content, "Success on retry")

    def test_long_wait_rate_limit_degrades_immediately(self):
        """Long Retry-After (> 15s) must raise immediately, no retry sleep."""
        from llm_utils import invoke_llm

        class LongRateLimit(Exception):
            status_code = 429

        async def _always_fail(messages):
            raise LongRateLimit("rate limit exceeded, Retry-After: 3600")

        mock_llm = MagicMock()
        mock_llm.ainvoke = _always_fail
        call_count_tracker = []

        async def run():
            try:
                await invoke_llm(mock_llm, [], task_name="test", timeout_s=10.0)
            except LongRateLimit:
                call_count_tracker.append(True)
                return
            raise AssertionError("Should have raised LongRateLimit")

        asyncio.run(run())
        self.assertEqual(len(call_count_tracker), 1, "Long-wait rate limit must raise on first attempt")


class TokenUsageAccumulationTest(unittest.TestCase):
    """Phase 8: Token usage must accumulate in state across LLM calls."""

    def test_merge_token_usage_accumulates(self):
        from llm_utils import merge_token_usage
        base = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        delta = {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20}
        result = merge_token_usage(base, delta)
        self.assertEqual(result["prompt_tokens"], 15)
        self.assertEqual(result["completion_tokens"], 35)
        self.assertEqual(result["total_tokens"], 50)

    def test_extract_token_usage_from_ai_message(self):
        from llm_utils import _extract_token_usage
        from langchain_core.messages import AIMessage
        msg = AIMessage(content="test")
        msg.response_metadata = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}}
        result = _extract_token_usage(msg)
        self.assertEqual(result["prompt_tokens"], 100)
        self.assertEqual(result["total_tokens"], 300)

    def test_extract_token_usage_returns_empty_for_pydantic_model(self):
        """Structured output Pydantic models have no response_metadata; must return {}."""
        from llm_utils import _extract_token_usage
        from schemas import GuardrailDecision, GuardrailCategory
        gd = GuardrailDecision(allowed=True, reason="test", category=GuardrailCategory.TRAVEL)
        result = _extract_token_usage(gd)
        self.assertEqual(result, {})

    def test_budget_agent_accumulates_token_usage(self):
        """budget_agent must merge token usage from LLM response into state."""
        import backend
        state = {
            "user_query": "Budget for Tokyo trip",
            "specialist_statuses": {},
            "trip_constraints": {},
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "llm_calls": 0,
            "llm_token_usage": {"prompt_tokens": 50, "total_tokens": 50},
        }

        mock_response = AIMessage(content="Budget: ~$2000")
        mock_response.response_metadata = {
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 150, "total_tokens": 250}
        }

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch.object(backend, "_llm_budget", mock_llm):
            result = asyncio.run(backend.budget_agent(state))

        token_usage = result.get("llm_token_usage", {})
        self.assertEqual(token_usage.get("prompt_tokens"), 150, "Should accumulate: 50 + 100 = 150")
        self.assertEqual(token_usage.get("total_tokens"), 300, "Should accumulate: 50 + 250 = 300")


class RunStatusTest(unittest.TestCase):
    """Phase 6+8: run_status enum in state and serialization."""

    def test_run_status_enum_values(self):
        from schemas import RunStatus
        self.assertEqual(RunStatus.COMPLETED.value, "COMPLETED")
        self.assertEqual(RunStatus.DEGRADED.value, "DEGRADED")
        self.assertEqual(RunStatus.BLOCKED.value, "BLOCKED")

    def test_guardrail_blocked_sets_blocked_status(self):
        """Guardrail-blocked response must have run_status=BLOCKED."""
        import backend
        state = {"user_query": "Explain quantum computing", "llm_calls": 0, "llm_token_usage": {}}
        from schemas import GuardrailDecision, GuardrailCategory
        blocked_decision = GuardrailDecision(
            allowed=False,
            reason="Not a travel request",
            category=GuardrailCategory.OUT_OF_SCOPE,
        )
        with patch.object(backend, "_llm_guardrail") as mg:
            mg.ainvoke = AsyncMock(return_value=blocked_decision)
            result = asyncio.run(backend.supervisor_agent(state))

        from schemas import RunStatus
        self.assertEqual(result["run_status"], RunStatus.BLOCKED.value)

    def test_itinerary_degraded_sets_degraded_status(self):
        """itinerary_degraded_agent must set run_status=DEGRADED."""
        import backend
        from schemas import RunStatus
        state = {"specialist_statuses": {"itinerary_agent": "DEGRADED"}}
        result = asyncio.run(backend.itinerary_degraded_agent(state))
        self.assertEqual(result["run_status"], RunStatus.DEGRADED.value)


if __name__ == "__main__":
    unittest.main()
