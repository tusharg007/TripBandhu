
"""TF1: Precise LLM call accounting tests."""
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage

from schemas import GuardrailDecision, GuardrailCategory, SupervisorDecision, AgentName, TravelConstraints


class LLMAccountingTest(unittest.TestCase):

    def test_hotel_agent_does_not_count_llm(self):
        """hotel_agent calls Tavily only — llm_calls stays unchanged."""
        import backend
        state = {
            "user_query": "Find hotels in Kyoto",
            "specialist_statuses": {},
            "llm_calls": 5,
        }

        async def _mock_tavily(query):
            return "Hotel: Kyoto Century Hotel"

        with patch.object(backend, 'tavily_mcp_search', side_effect=_mock_tavily):
            result = asyncio.run(backend.hotel_agent(state))

        self.assertEqual(result["llm_calls"], 5, "hotel_agent must not increment llm_calls")

    def test_flight_agent_counts_llm_on_success(self):
        """flight_agent increments llm_calls exactly once on LLM success."""
        import backend
        state = {
            "user_query": "Flights from Delhi to Tokyo",
            "specialist_statuses": {},
            "llm_calls": 2,
        }
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="ANA and JAL fly this route"))

        async def _mock_aviation(tool_name):
            return ["HND", "NRT"] if tool_name == "list_airports" else ["ANA", "JAL"]

        with patch.object(backend, 'aviation_mcp_call', side_effect=_mock_aviation), \
             patch.object(backend, 'llm', mock_llm):
            result = asyncio.run(backend.flight_agent(state))

        self.assertEqual(result["llm_calls"], 3)

    def test_flight_agent_does_not_count_llm_on_failure(self):
        """flight_agent must NOT increment llm_calls when provider fails before LLM is called."""
        import backend
        state = {
            "user_query": "Flights from Delhi to Tokyo",
            "specialist_statuses": {},
            "llm_calls": 2,
        }

        async def _fail_aviation(tool_name):
            raise RuntimeError("uvx not found")

        with patch.object(backend, 'aviation_mcp_call', side_effect=_fail_aviation):
            result = asyncio.run(backend.flight_agent(state))

        self.assertEqual(result["llm_calls"], 2)

    def test_budget_agent_counts_llm(self):
        """budget_agent calls LLM — must increment llm_calls by 1."""
        import backend
        state = {
            "user_query": "Budget for Tokyo trip",
            "specialist_statuses": {},
            "trip_constraints": {},
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "llm_calls": 3,
        }
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Budget feasible"))

        with patch.object(backend, 'llm', mock_llm):
            result = asyncio.run(backend.budget_agent(state))

        self.assertEqual(result["llm_calls"], 4)

    def test_itinerary_agent_counts_llm(self):
        """itinerary_agent calls LLM — must increment llm_calls by 1."""
        import backend
        state = {
            "user_query": "7-day Tokyo itinerary",
            "specialist_statuses": {},
            "trip_constraints": {},
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "llm_calls": 4,
        }
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Day 1: Arrive in Tokyo"))

        with patch.object(backend, 'llm', mock_llm):
            result = asyncio.run(backend.itinerary_agent(state))

        self.assertEqual(result["llm_calls"], 5)

    def test_itinerary_revision_agent_counts_llm(self):
        """itinerary_revision_agent calls LLM — must increment llm_calls by 1."""
        import backend
        state = {
            "user_query": "7-day Tokyo itinerary",
            "specialist_statuses": {},
            "trip_constraints": {},
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "Draft",
            "human_feedback": "Add more museums",
            "review_iteration": 0,
            "llm_calls": 5,
        }
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Revised Day 1: Museum"))

        with patch.object(backend, 'llm', mock_llm):
            result = asyncio.run(backend.itinerary_revision_agent(state))

        self.assertEqual(result["llm_calls"], 6)

    def test_final_agent_counts_llm(self):
        """final_agent calls LLM — must increment llm_calls by 1."""
        import backend
        state = {
            "user_query": "Tokyo trip",
            "specialist_statuses": {},
            "trip_constraints": {},
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "Draft",
            "approved": True,
            "human_feedback": "",
            "review_iteration": 0,
            "llm_calls": 6,
        }
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Polished final plan"))

        with patch.object(backend, 'llm', mock_llm):
            result = asyncio.run(backend.final_agent(state))

        self.assertEqual(result["llm_calls"], 7)

    def test_weather_agent_counts_destination_extraction(self):
        """weather_agent calls extract_destination_async (LLM) — must count it."""
        import backend
        state = {
            "user_query": "Weather in Kyoto next week",
            "specialist_statuses": {},
            "trip_constraints": {"destination": "Kyoto"},
            "llm_calls": 2,
        }

        async def _mock_extract(query):
            return "Kyoto"

        async def _mock_weather(city):
            return "Sunny 25C"

        async def _mock_forecast(city):
            return "Clear skies for a week"

        with patch.object(backend, 'extract_destination_async', side_effect=_mock_extract), \
             patch.object(backend, 'weather_mcp_search', side_effect=_mock_weather), \
             patch.object(backend, 'forecast_mcp_search', side_effect=_mock_forecast):
            result = asyncio.run(backend.weather_agent(state))

        self.assertEqual(result["llm_calls"], 3)

    def test_supervisor_guardrail_counts_both_llm_invocations(self):
        """supervisor_agent counts 1 for guardrail LLM and 1 for supervisor LLM (total +2)."""
        import backend
        state = {"user_query": "Plan a 7-day trip to Tokyo", "llm_calls": 0}
        allowed = GuardrailDecision(allowed=True, reason="travel", category=GuardrailCategory.TRAVEL)
        decision = SupervisorDecision(
            selected_agents=[AgentName.ITINERARY],
            constraints=TravelConstraints(destination="Tokyo"),
            reasoning="Plan trip",
        )
        with patch.object(backend, '_llm_guardrail') as mg, \
             patch.object(backend, '_llm_supervisor') as ms:
            mg.ainvoke = AsyncMock(return_value=allowed)
            ms.ainvoke = AsyncMock(return_value=decision)
            result = asyncio.run(backend.supervisor_agent(state))

        self.assertEqual(result["llm_calls"], 2)

    def test_supervisor_guardrail_fallback_does_not_count_second_guardrail_llm(self):
        """When guardrail LLM fails and uses keyword fallback, it does NOT count a second LLM call."""
        import backend
        state = {"user_query": "Plan a flight to Tokyo", "llm_calls": 0}
        decision = SupervisorDecision(
            selected_agents=[AgentName.FLIGHT],
            constraints=TravelConstraints(destination="Tokyo"),
            reasoning="Flight query",
        )
        with patch.object(backend, '_llm_guardrail') as mg, \
             patch.object(backend, '_llm_supervisor') as ms:
            mg.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))
            ms.ainvoke = AsyncMock(return_value=decision)
            result = asyncio.run(backend.supervisor_agent(state))

        # 0 for failed guardrail + 1 for supervisor = 1
        self.assertEqual(result["llm_calls"], 1)


if __name__ == "__main__":
    unittest.main()
