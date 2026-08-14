
"""TF1: Iterative HITL loop and review cap tests."""
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage

from schemas import ReviewDecision


class HITLLoopTest(unittest.TestCase):

    def _base_state(self, iteration=0, approved=False, feedback=""):
        return {
            "user_query": "Plan a 5-day trip to Tokyo",
            "itinerary": "Day 1: Arrive in Tokyo...",
            "approval_request": "Review this draft.",
            "selected_agents": ["flight_agent", "itinerary_agent"],
            "supervisor_reasoning": "Trip planning.",
            "specialist_statuses": {"itinerary_agent": "COMPLETED"},
            "approved": approved,
            "human_feedback": feedback,
            "review_iteration": iteration,
            "review_history": [],
            "llm_calls": 3,
            "trip_constraints": {"destination": "Tokyo"},
            "flight_results": "",
        }

    def test_route_approve_goes_to_final(self):
        """Approved review routes to final_agent."""
        import backend
        state = self._base_state(approved=True)
        dest = backend.route_after_review(state)
        self.assertEqual(dest, "final_agent")

    def test_route_reject_goes_to_revision(self):
        """Rejected review with feedback routes to itinerary_revision."""
        import backend
        state = self._base_state(approved=False, feedback="Add more budget hotels.")
        dest = backend.route_after_review(state)
        self.assertEqual(dest, "itinerary_revision")

    def test_review_cap_must_never_auto_finalize(self):
        """MANDATORY TF1 REQUIREMENT: When MAX_REVIEW_ITERATIONS is reached without approval,
        it must route to review_limit_reached, NEVER to final_agent or auto-approve."""
        import backend
        from agent_config import MAX_REVIEW_ITERATIONS
        state = self._base_state(
            iteration=MAX_REVIEW_ITERATIONS,
            approved=False,
            feedback="Still want changes",
        )
        dest = backend.route_after_review(state)
        self.assertEqual(dest, "review_limit_reached",
                         "Cap reached without approval must route to review_limit_reached, NOT final_agent")

    def test_review_limit_reached_node_returns_stable_state(self):
        """review_limit_reached_agent returns approved=False, review_limit_reached=True, and safe message."""
        import backend
        from agent_config import MAX_REVIEW_ITERATIONS
        state = self._base_state(
            iteration=MAX_REVIEW_ITERATIONS,
            approved=False,
            feedback="Need changes",
        )
        result = asyncio.run(backend.review_limit_reached_agent(state))

        self.assertFalse(result["approved"], "Review limit state must NOT be approved")
        self.assertTrue(result["review_limit_reached"])
        self.assertIn("Review limit reached", result["final_response"])
        self.assertIn("No final approved travel plan was produced", result["final_response"])
        self.assertIn("Please start a new planning run", result["final_response"])

    def test_revision_increments_iteration(self):
        """itinerary_revision_agent increments review_iteration."""
        import backend
        state = self._base_state(iteration=0, feedback="Add a museum visit.")
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Revised: Day 1 museum..."))

        with patch.object(backend, 'llm', mock_llm):
            result = asyncio.run(backend.itinerary_revision_agent(state))

        self.assertEqual(result["review_iteration"], 1)

    def test_review_history_accumulates(self):
        """ReviewDecision entries accumulate in review_history."""
        from schemas import ReviewDecision
        entries = []
        for i in range(3):
            entry = ReviewDecision(
                iteration=i,
                approved=(i == 2),
                feedback=f"Feedback {i}" if i < 2 else "",
            )
            entries.append(entry.model_dump())
        self.assertEqual(len(entries), 3)
        self.assertFalse(entries[0]["approved"])
        self.assertFalse(entries[1]["approved"])
        self.assertTrue(entries[2]["approved"])


class HITLApiValidationTest(unittest.TestCase):

    def test_revision_without_feedback_rejected(self):
        """API endpoint must reject revision request with empty feedback (400)."""
        import asyncio, app, json
        from unittest.mock import MagicMock
        req = MagicMock()
        req.app.state.travel_service = None
        response = asyncio.run(
            app.resume_travel_planner(
                req,
                app.TravelResumeRequest(
                    thread_id="user_test_123",
                    approved=False,
                    feedback="",
                )
            )
        )
        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(payload["error_code"], "REVISION_REQUIRES_FEEDBACK")

    def test_approve_without_feedback_is_ok(self):
        """Approve without feedback succeeds."""
        import asyncio, app, json
        from unittest.mock import patch, AsyncMock, MagicMock
        req = MagicMock()
        req.app.state.travel_service = None
        with patch('app.resume_travel_agent', new=AsyncMock(return_value={
            "thread_id": "user_test_123",
            "answer": "Final plan",
            "requires_approval": False,
            "approval_request": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "selected_agents": [],
            "specialist_statuses": {},
            "trip_constraints": {},
            "supervisor_reasoning": "",
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "approved": True,
            "human_feedback": "",
            "review_iteration": 0,
            "review_limit_reached": False,
            "llm_calls": 5,
        })):
            response = asyncio.run(
                app.resume_travel_planner(
                    req,
                    app.TravelResumeRequest(
                        thread_id="user_test_123",
                        approved=True,
                        feedback="",
                    )
                )
            )
        payload = json.loads(response.body.decode("utf-8"))
        self.assertTrue(payload["success"])


if __name__ == "__main__":
    unittest.main()
