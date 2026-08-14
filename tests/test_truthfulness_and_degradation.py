from pathlib import Path
import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import AIMessage

import backend

ROOT = Path(__file__).resolve().parents[1]


class TruthfulnessAndDegradationTest(unittest.TestCase):
    def test_flight_agent_failure_sets_degraded_and_safe_message(self):
        state = {
            "user_query": "Plan a 7-day trip to Tokyo",
            "selected_agents": ["flight_agent", "itinerary_agent"],
            "specialist_statuses": {"flight_agent": "SELECTED", "itinerary_agent": "SELECTED"},
            "llm_calls": 0,
        }

        # Mock aviation_mcp_call to simulate a tool/uvx failure
        async def _fail_aviation(tool_name):
            raise RuntimeError("uvx was not found. Install uv and run uvx --version.")

        with patch("backend.aviation_mcp_call", side_effect=_fail_aviation):
            result = asyncio.run(backend.flight_agent(state))

        # 1. Must set specialist status to DEGRADED
        self.assertEqual(result["specialist_statuses"]["flight_agent"], "DEGRADED")
        # 2. Must set flight_data_available to False
        self.assertFalse(result["flight_data_available"])
        # 3. Must NOT leak internal runtime error / uvx message
        self.assertNotIn("uvx was not found", result["flight_results"])
        self.assertNotIn("RuntimeError", result["flight_results"])
        # 4. Must return safe user-facing message
        self.assertIn("Live flight data is temporarily unavailable", result["flight_results"])

    def test_flight_agent_success_sets_completed_and_available(self):
        state = {
            "user_query": "Plan a 7-day trip to Tokyo",
            "selected_agents": ["flight_agent", "itinerary_agent"],
            "specialist_statuses": {"flight_agent": "SELECTED", "itinerary_agent": "SELECTED"},
            "llm_calls": 0,
        }

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Direct flights available on ANA and JAL")
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Direct flights available on ANA and JAL"))

        async def _mock_aviation(tool_name):
            return ["HND", "NRT"]

        with patch("backend.aviation_mcp_call", side_effect=_mock_aviation), \
             patch("backend.llm", mock_llm):
            result = asyncio.run(backend.flight_agent(state))

        self.assertEqual(result["specialist_statuses"]["flight_agent"], "COMPLETED")
        self.assertTrue(result["flight_data_available"])
        self.assertIn("ANA and JAL", result["flight_results"])

    def test_supervisor_marks_unselected_agents_as_not_selected(self):
        state = {"user_query": "What is the weather in Paris?", "llm_calls": 0}

        from schemas import GuardrailDecision, GuardrailCategory, SupervisorDecision, AgentName, TravelConstraints

        allowed = GuardrailDecision(allowed=True, reason="travel", category=GuardrailCategory.TRAVEL)
        decision = SupervisorDecision(
            selected_agents=[AgentName.WEATHER, AgentName.ITINERARY],
            constraints=TravelConstraints(destination="Paris"),
            reasoning="Selected weather and itinerary specialists.",
        )

        with patch("backend._llm_guardrail") as mg, patch("backend._llm_supervisor") as ms:
            mg.ainvoke = AsyncMock(return_value=allowed)
            ms.ainvoke = AsyncMock(return_value=decision)
            result = asyncio.run(backend.supervisor_agent(state))

        self.assertEqual(result["specialist_statuses"]["weather_agent"], "SELECTED")
        self.assertEqual(result["specialist_statuses"]["itinerary_agent"], "SELECTED")
        self.assertEqual(result["specialist_statuses"]["flight_agent"], "NOT_SELECTED")
        self.assertEqual(result["specialist_statuses"]["hotel_agent"], "NOT_SELECTED")
        self.assertEqual(result["specialist_statuses"]["budget_agent"], "NOT_SELECTED")

    def test_prompts_explicitly_forbid_fabricating_unavailable_data(self):
        backend_code = (ROOT / "backend.py").read_text(encoding="utf-8")

        # Budget prompt instructions
        self.assertIn("CRITICAL GROUNDEDNESS RULES:", backend_code)
        self.assertIn("DO NOT invent exact flight fares or prices", backend_code)

        # Itinerary prompt instructions
        self.assertIn("DO NOT invent or fabricate flight prices", backend_code)

        # Final agent prompt instructions
        self.assertIn("CRITICAL GROUNDEDNESS & TRUTHFULNESS RULES:", backend_code)
        self.assertIn("Live flight information was unavailable for this run. Verify schedules and fares with an airline or booking provider before booking.", backend_code)

    def test_frontend_script_and_style_have_degraded_and_sticky_review(self):
        script_code = (ROOT / "static" / "script.js").read_text(encoding="utf-8")
        style_code = (ROOT / "static" / "style.css").read_text(encoding="utf-8")

        # Script handles degraded status
        self.assertIn("degraded", script_code)
        self.assertIn("specialist_statuses", script_code)

        # Style has .agent-degraded and sticky review actions
        self.assertIn(".agent-degraded", style_code)
        self.assertIn("position: sticky", style_code)
        self.assertIn(".review-actions", style_code)


if __name__ == "__main__":
    unittest.main()
