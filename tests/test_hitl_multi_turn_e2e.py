"""TF1: End-to-end multi-turn HITL graph execution tests (in-memory checkpointer)."""
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
from agent_config import MAX_REVIEW_ITERATIONS


class HITLMultiTurnE2ETest(unittest.TestCase):

    def _build_mock_service(self):
        """Create a TravelAgentService with in-memory checkpointer."""
        checkpointer = InMemorySaver()
        return backend.create_travel_service(checkpointer=checkpointer)

    def test_multi_turn_hitl_revision_loop_to_approval(self):
        """E2E flow: Initial run -> Review (iteration 0) -> Request Revision 1 -> Review (iteration 1) -> Request Revision 2 -> Review (iteration 2) -> Approve -> Final Plan."""
        service = self._build_mock_service()

        allowed = GuardrailDecision(allowed=True, reason="travel", category=GuardrailCategory.TRAVEL)
        supervisor_decision = SupervisorDecision(
            selected_agents=[AgentName.FLIGHT, AgentName.ITINERARY],
            constraints=TravelConstraints(destination="Tokyo", duration="5 days"),
            reasoning="Selected flights and itinerary.",
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[
            AIMessage(content="Flight options on ANA and JAL"), # flight_agent
            AIMessage(content="Draft Itinerary v0"),            # itinerary_agent
            AIMessage(content="Revised Itinerary v1"),          # itinerary_revision 1
            AIMessage(content="Revised Itinerary v2"),          # itinerary_revision 2
            AIMessage(content="Final Approved Plan v2"),        # final_agent
        ])

        async def _mock_aviation(tool_name):
            return ["HND"]

        thread_id = "test_hitl_multi_turn_thread"

        with patch.object(backend, '_llm_guardrail') as mg, \
             patch.object(backend, '_llm_supervisor') as ms, \
             patch.object(backend, 'aviation_mcp_call', side_effect=_mock_aviation), \
             patch.object(backend, 'llm', mock_llm):

            mg.ainvoke = AsyncMock(return_value=allowed)
            ms.ainvoke = AsyncMock(return_value=supervisor_decision)

            # Turn 1: Initial planning run -> should pause at human_review
            r1 = asyncio.run(service.run("Plan a 5-day trip to Tokyo", thread_id=thread_id))
            self.assertTrue(r1["requires_approval"], "Turn 1 must pause for review")
            self.assertIn("Draft Itinerary v0", r1["itinerary"])
            self.assertEqual(r1["review_iteration"], 0)

            # Turn 2: Request Revision 1
            r2 = asyncio.run(service.resume(thread_id=thread_id, approved=False, feedback="Add Asakusa temple"))
            self.assertTrue(r2["requires_approval"], "Turn 2 must pause for second review")
            self.assertIn("Revised Itinerary v1", r2["itinerary"])
            self.assertEqual(r2["review_iteration"], 1)

            # Turn 3: Request Revision 2
            r3 = asyncio.run(service.resume(thread_id=thread_id, approved=False, feedback="Add Shibuya crossing"))
            self.assertTrue(r3["requires_approval"], "Turn 3 must pause for third review")
            self.assertIn("Revised Itinerary v2", r3["itinerary"])
            self.assertEqual(r3["review_iteration"], 2)

            # Turn 4: Approve
            r4 = asyncio.run(service.resume(thread_id=thread_id, approved=True, feedback=""))
            self.assertFalse(r4["requires_approval"], "Turn 4 should complete after approval")
            self.assertTrue(r4["approved"])
            self.assertIn("Final Approved Plan", r4["answer"])

    def test_review_cap_terminates_safely_without_approval(self):
        """MANDATORY TEST: 10 rejected reviews -> no final approved plan -> REVIEW_LIMIT_REACHED."""
        service = self._build_mock_service()

        allowed = GuardrailDecision(allowed=True, reason="travel", category=GuardrailCategory.TRAVEL)
        supervisor_decision = SupervisorDecision(
            selected_agents=[AgentName.ITINERARY],
            constraints=TravelConstraints(destination="Paris"),
            reasoning="Plan Paris trip.",
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Draft Itinerary"))

        thread_id = "test_review_cap_thread"

        with patch.object(backend, '_llm_guardrail') as mg, \
             patch.object(backend, '_llm_supervisor') as ms, \
             patch.object(backend, 'llm', mock_llm):

            mg.ainvoke = AsyncMock(return_value=allowed)
            ms.ainvoke = AsyncMock(return_value=supervisor_decision)

            # Turn 0: initial run
            r = asyncio.run(service.run("Plan a trip to Paris", thread_id=thread_id))
            self.assertTrue(r["requires_approval"])

            # 9 revisions (iterations 1..9): each should still pause for review
            for i in range(1, MAX_REVIEW_ITERATIONS):
                r = asyncio.run(service.resume(thread_id=thread_id, approved=False, feedback=f"Change {i}"))
                self.assertTrue(r["requires_approval"], f"Iteration {i} must pause for review")
                self.assertEqual(r["review_iteration"], i)

            # 10th revision request (iteration 10 reaches MAX_REVIEW_ITERATIONS):
            # Must route to review_limit_reached node, NOT final_agent, and NOT approve!
            r_cap = asyncio.run(service.resume(thread_id=thread_id, approved=False, feedback="Change 10"))

            self.assertFalse(r_cap["requires_approval"], "Must terminate, not pause again")
            self.assertFalse(r_cap["approved"], "Must NOT be marked approved")
            self.assertTrue(r_cap["review_limit_reached"], "review_limit_reached must be True")
            self.assertIn("Review limit reached", r_cap["answer"])
            self.assertIn("No final approved travel plan was produced", r_cap["answer"])


if __name__ == "__main__":
    unittest.main()
