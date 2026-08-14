"""TF1: Specialist-only end-to-end execution tests."""
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

import backend
from schemas import (
    AgentName,
    SupervisorDecision,
    TravelConstraints,
    GuardrailDecision,
    GuardrailCategory,
)


class SpecialistOnlyE2ETest(unittest.TestCase):

    def _build_service(self):
        checkpointer = InMemorySaver()
        return backend.create_travel_service(checkpointer=checkpointer)

    def test_flight_only_request_does_not_invoke_itinerary_or_interrupt(self):
        """A flight-only request runs flight_agent -> final_agent, produces final answer, and requires NO approval."""
        service = self._build_service()

        allowed = GuardrailDecision(allowed=True, reason="travel", category=GuardrailCategory.TRAVEL)
        supervisor_decision = SupervisorDecision(
            selected_agents=[AgentName.FLIGHT],
            constraints=TravelConstraints(destination="Tokyo", origin="Delhi"),
            reasoning="Selected only flight specialist.",
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[
            AIMessage(content="Direct flights: ANA, JAL, Air India. Duration: 8 hours."), # flight_agent
            AIMessage(content="Flight Summary for Delhi to Tokyo:\n1. ANA...\n2. JAL..."),  # final_agent
        ])

        async def _mock_aviation(tool_name):
            return ["DEL", "HND"] if tool_name == "list_airports" else ["ANA", "JAL", "Air India"]

        with patch.object(backend, '_llm_guardrail') as mg, \
             patch.object(backend, '_llm_supervisor') as ms, \
             patch.object(backend, 'aviation_mcp_call', side_effect=_mock_aviation), \
             patch.object(backend, 'llm', mock_llm):

            mg.ainvoke = AsyncMock(return_value=allowed)
            ms.ainvoke = AsyncMock(return_value=supervisor_decision)

            result = asyncio.run(service.run("Find flights from Delhi to Tokyo"))

            self.assertTrue(result["guardrail_allowed"])
            self.assertEqual(result["selected_agents"], ["flight_agent"])
            self.assertFalse(result["requires_approval"], "Flight-only query must NOT require human approval")
            self.assertEqual(result["specialist_statuses"]["flight_agent"], "COMPLETED")
            self.assertEqual(result["specialist_statuses"]["itinerary_agent"], "NOT_SELECTED")
            self.assertIn("Flight Summary", result["answer"])

    def test_hotel_only_request_does_not_invoke_itinerary_or_interrupt(self):
        """A hotel-only request runs hotel_agent -> final_agent, requires NO approval."""
        service = self._build_service()

        allowed = GuardrailDecision(allowed=True, reason="travel", category=GuardrailCategory.TRAVEL)
        supervisor_decision = SupervisorDecision(
            selected_agents=[AgentName.HOTEL],
            constraints=TravelConstraints(destination="Paris"),
            reasoning="Selected only hotel specialist.",
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Hotel Summary for Paris: Le Bristol, Hotel Lutetia..."))

        async def _mock_tavily(query):
            return "Top luxury hotels in Paris: Le Bristol, Hotel Lutetia, Ritz Paris."

        with patch.object(backend, '_llm_guardrail') as mg, \
             patch.object(backend, '_llm_supervisor') as ms, \
             patch.object(backend, 'tavily_mcp_search', side_effect=_mock_tavily), \
             patch.object(backend, 'llm', mock_llm):

            mg.ainvoke = AsyncMock(return_value=allowed)
            ms.ainvoke = AsyncMock(return_value=supervisor_decision)

            result = asyncio.run(service.run("Find luxury hotels in Paris"))

            self.assertTrue(result["guardrail_allowed"])
            self.assertEqual(result["selected_agents"], ["hotel_agent"])
            self.assertFalse(result["requires_approval"], "Hotel-only query must NOT require approval")
            self.assertEqual(result["specialist_statuses"]["hotel_agent"], "COMPLETED")
            self.assertEqual(result["specialist_statuses"]["itinerary_agent"], "NOT_SELECTED")
            self.assertIn("Hotel Summary", result["answer"])


if __name__ == "__main__":
    unittest.main()
