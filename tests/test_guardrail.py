
"""TF1: Input guardrail tests including deterministic fallback."""
import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from schemas import GuardrailDecision, GuardrailCategory


class GuardrailTest(unittest.TestCase):

    def test_travel_input_accepted(self):
        """Valid travel query should pass guardrail."""
        import backend
        state = {"user_query": "Plan a 5-day trip to Japan", "llm_calls": 0}
        allowed_decision = GuardrailDecision(
            allowed=True,
            reason="Travel query",
            category=GuardrailCategory.TRAVEL,
        )
        blocked_supervisor = GuardrailDecision(
            allowed=True, reason="travel", category=GuardrailCategory.TRAVEL
        )
        with patch.object(backend, '_llm_guardrail') as mg, \
             patch.object(backend, '_llm_supervisor') as ms:
            mg.ainvoke = AsyncMock(return_value=allowed_decision)
            # Supervisor returns structured decision
            from schemas import SupervisorDecision, AgentName, TravelConstraints
            ms.ainvoke = AsyncMock(return_value=SupervisorDecision(
                selected_agents=[AgentName.ITINERARY],
                constraints=TravelConstraints(destination="Japan"),
                reasoning="Trip to Japan.",
            ))
            result = asyncio.run(backend.supervisor_agent(state))
        self.assertTrue(result["guardrail_allowed"])

    def test_off_topic_rejected(self):
        """Off-topic query should be rejected by guardrail."""
        import backend
        state = {"user_query": "Write me a Python sorting algorithm", "llm_calls": 0}
        blocked_decision = GuardrailDecision(
            allowed=False,
            reason="Not a travel request",
            category=GuardrailCategory.OUT_OF_SCOPE,
        )
        with patch.object(backend, '_llm_guardrail') as mg:
            mg.ainvoke = AsyncMock(return_value=blocked_decision)
            result = asyncio.run(backend.supervisor_agent(state))
        self.assertFalse(result["guardrail_allowed"])
        self.assertEqual(result.get("selected_agents", []), [],
                         "Blocked requests must have no selected agents")

    def test_guardrail_parser_failure_uses_keyword_fallback_for_travel(self):
        """When guardrail LLM fails, keyword fallback allows travel queries."""
        import backend
        state = {"user_query": "I want to book a flight to Paris", "llm_calls": 0}
        with patch.object(backend, '_llm_guardrail') as mg, \
             patch.object(backend, '_llm_supervisor') as ms:
            mg.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
            from schemas import SupervisorDecision, AgentName, TravelConstraints
            ms.ainvoke = AsyncMock(return_value=SupervisorDecision(
                selected_agents=[AgentName.FLIGHT, AgentName.ITINERARY],
                constraints=TravelConstraints(destination="Paris"),
                reasoning="Flight to Paris.",
            ))
            result = asyncio.run(backend.supervisor_agent(state))
        # Keyword "flight" matches travel, so allowed=True
        self.assertTrue(result["guardrail_allowed"])

    def test_guardrail_parser_failure_blocks_nontransavel_query(self):
        """When guardrail LLM fails, keyword fallback blocks non-travel queries."""
        import backend
        state = {"user_query": "Explain quantum computing to me", "llm_calls": 0}
        with patch.object(backend, '_llm_guardrail') as mg:
            mg.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
            result = asyncio.run(backend.supervisor_agent(state))
        # No travel keywords -> blocked by deterministic fallback
        self.assertFalse(result["guardrail_allowed"])

    def test_deterministic_travel_check_direct(self):
        """Direct test of the deterministic guardrail keyword check."""
        import backend
        travel_result = backend._deterministic_travel_check("Plan a hotel trip to Bali")
        self.assertTrue(travel_result.allowed)

        nontransavel_result = backend._deterministic_travel_check("What is the capital of France?")
        self.assertFalse(nontransavel_result.allowed)

    def test_guardrail_category_schema(self):
        """GuardrailDecision can hold all defined categories."""
        for cat in GuardrailCategory:
            gd = GuardrailDecision(allowed=False, reason="test", category=cat)
            self.assertEqual(gd.category, cat)


if __name__ == "__main__":
    unittest.main()
