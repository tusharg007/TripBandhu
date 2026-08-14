
"""TF1: Structured supervisor output and dynamic specialist routing tests."""
import asyncio
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

from schemas import AgentName, SupervisorDecision, TravelConstraints, GuardrailDecision, GuardrailCategory


def GuardrailDecision_allowed():
    return GuardrailDecision(allowed=True, reason="Travel query", category=GuardrailCategory.TRAVEL)


class StructuredSupervisorTest(unittest.TestCase):

    def _make_state(self, query="Plan a 7-day trip to Tokyo from Delhi"):
        return {"user_query": query, "llm_calls": 0}

    def test_valid_full_trip_route(self):
        """Full-trip query should route to specialists and itinerary."""
        import backend
        state = self._make_state("Plan a 7-day Tokyo trip from Delhi with hotels and budget")
        mock_guardrail = GuardrailDecision_allowed()
        mock_decision = SupervisorDecision(
            selected_agents=[AgentName.FLIGHT, AgentName.HOTEL, AgentName.WEATHER, AgentName.BUDGET, AgentName.ITINERARY],
            constraints=TravelConstraints(destination="Tokyo", origin="Delhi", duration="7 days"),
            reasoning="Selected all specialists for a comprehensive Tokyo trip.",
        )
        with patch.object(backend, '_llm_guardrail') as mg, \
             patch.object(backend, '_llm_supervisor') as ms:
            mg.ainvoke = AsyncMock(return_value=mock_guardrail)
            ms.ainvoke = AsyncMock(return_value=mock_decision)
            result = asyncio.run(backend.supervisor_agent(state))

        self.assertTrue(result["guardrail_allowed"])
        self.assertIn("flight_agent", result["selected_agents"])
        self.assertIn("hotel_agent", result["selected_agents"])
        self.assertIn("itinerary_agent", result["selected_agents"])
        self.assertEqual(result["trip_constraints"]["destination"], "Tokyo")
        self.assertEqual(result["trip_constraints"]["origin"], "Delhi")

    def test_flight_only_route_does_not_force_itinerary(self):
        """Flight-only query should select ONLY flight_agent, NOT force itinerary_agent."""
        import backend
        state = self._make_state("What airlines fly from Delhi to Tokyo?")
        mock_guardrail = GuardrailDecision_allowed()
        mock_decision = SupervisorDecision(
            selected_agents=[AgentName.FLIGHT],
            constraints=TravelConstraints(destination="Tokyo", origin="Delhi"),
            reasoning="Selected flight specialist for route query.",
        )
        with patch.object(backend, '_llm_guardrail') as mg, \
             patch.object(backend, '_llm_supervisor') as ms:
            mg.ainvoke = AsyncMock(return_value=mock_guardrail)
            ms.ainvoke = AsyncMock(return_value=mock_decision)
            result = asyncio.run(backend.supervisor_agent(state))

        self.assertTrue(result["guardrail_allowed"])
        self.assertEqual(result["selected_agents"], ["flight_agent"],
                         "Flight-only query must not force itinerary_agent")
        self.assertEqual(result["specialist_statuses"]["flight_agent"], "SELECTED")
        self.assertEqual(result["specialist_statuses"]["itinerary_agent"], "NOT_SELECTED")
        self.assertEqual(result["specialist_statuses"]["hotel_agent"], "NOT_SELECTED")

    def test_hotel_only_route_does_not_force_itinerary(self):
        """Hotel-only query should select ONLY hotel_agent."""
        import backend
        state = self._make_state("Find luxury hotels in Tokyo Ginza")
        mock_guardrail = GuardrailDecision_allowed()
        mock_decision = SupervisorDecision(
            selected_agents=[AgentName.HOTEL],
            constraints=TravelConstraints(destination="Tokyo"),
            reasoning="Selected hotel specialist.",
        )
        with patch.object(backend, '_llm_guardrail') as mg, \
             patch.object(backend, '_llm_supervisor') as ms:
            mg.ainvoke = AsyncMock(return_value=mock_guardrail)
            ms.ainvoke = AsyncMock(return_value=mock_decision)
            result = asyncio.run(backend.supervisor_agent(state))

        self.assertEqual(result["selected_agents"], ["hotel_agent"],
                         "Hotel-only query must not force itinerary_agent")
        self.assertEqual(result["specialist_statuses"]["hotel_agent"], "SELECTED")
        self.assertEqual(result["specialist_statuses"]["itinerary_agent"], "NOT_SELECTED")

    def test_weather_only_route(self):
        """Weather-only query should select ONLY weather_agent."""
        import backend
        state = self._make_state("What is the weather in Paris next week?")
        mock_guardrail = GuardrailDecision_allowed()
        mock_decision = SupervisorDecision(
            selected_agents=[AgentName.WEATHER],
            constraints=TravelConstraints(destination="Paris"),
            reasoning="Selected weather specialist.",
        )
        with patch.object(backend, '_llm_guardrail') as mg, \
             patch.object(backend, '_llm_supervisor') as ms:
            mg.ainvoke = AsyncMock(return_value=mock_guardrail)
            ms.ainvoke = AsyncMock(return_value=mock_decision)
            result = asyncio.run(backend.supervisor_agent(state))

        self.assertEqual(result["selected_agents"], ["weather_agent"])
        self.assertEqual(result["specialist_statuses"]["weather_agent"], "SELECTED")
        self.assertEqual(result["specialist_statuses"]["itinerary_agent"], "NOT_SELECTED")

    def test_budget_only_route(self):
        """Budget-only query should select ONLY budget_agent."""
        import backend
        state = self._make_state("Is 1000 USD enough for 3 days in London?")
        mock_guardrail = GuardrailDecision_allowed()
        mock_decision = SupervisorDecision(
            selected_agents=[AgentName.BUDGET],
            constraints=TravelConstraints(destination="London", budget="1000 USD"),
            reasoning="Selected budget specialist.",
        )
        with patch.object(backend, '_llm_guardrail') as mg, \
             patch.object(backend, '_llm_supervisor') as ms:
            mg.ainvoke = AsyncMock(return_value=mock_guardrail)
            ms.ainvoke = AsyncMock(return_value=mock_decision)
            result = asyncio.run(backend.supervisor_agent(state))

        self.assertEqual(result["selected_agents"], ["budget_agent"])

    def test_invalid_agent_name_rejected(self):
        """AgentName validator should reject unknown agent names."""
        decision = SupervisorDecision(
            selected_agents=["flight_agent", "random_agent_xyz", "itinerary_agent"],
            constraints=TravelConstraints(),
            reasoning="Test",
        )
        agent_values = [a.value for a in decision.selected_agents]
        self.assertIn("flight_agent", agent_values)
        self.assertIn("itinerary_agent", agent_values)
        self.assertNotIn("random_agent_xyz", agent_values)

    def test_supervisor_structured_failure_uses_safe_defaults(self):
        """When structured output fails, supervisor must use safe defaults, not crash."""
        import backend
        state = self._make_state("Plan a trip to Paris")
        mock_guardrail = GuardrailDecision_allowed()
        with patch.object(backend, '_llm_guardrail') as mg, \
             patch.object(backend, '_llm_supervisor') as ms:
            mg.ainvoke = AsyncMock(return_value=mock_guardrail)
            ms.ainvoke = AsyncMock(side_effect=RuntimeError("LLM API error"))
            result = asyncio.run(backend.supervisor_agent(state))

        self.assertTrue(result["guardrail_allowed"])
        self.assertIn("itinerary_agent", result["selected_agents"])

    def test_specialist_only_routing_graph_execution(self):
        """When only flight_agent is selected, route_after_agent routes directly to final_agent without interrupt."""
        import backend
        state = {
            "selected_agents": ["flight_agent"],
            "specialist_statuses": {"flight_agent": "COMPLETED"},
        }
        dest = backend.route_after_agent("flight_agent")(state)
        self.assertEqual(dest, "final_agent", "Specialist-only route must route to final_agent")

    def test_duplicate_agent_selections_collapse(self):
        """Duplicate agent names in the LLM response should not cause duplicates."""
        decision = SupervisorDecision(
            selected_agents=["flight_agent", "flight_agent", "hotel_agent"],
            constraints=TravelConstraints(),
            reasoning="Test",
        )
        agent_values = [a.value for a in decision.selected_agents]
        self.assertEqual(agent_values.count("flight_agent"), 1)


if __name__ == "__main__":
    unittest.main()
