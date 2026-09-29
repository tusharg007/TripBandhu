"""Regression checks for a browser resuming stale or completed trip threads."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from backend import ReviewNotPendingError, TravelAgentService


class ResumeStateTests(unittest.TestCase):
    def test_unknown_thread_does_not_invoke_the_graph(self):
        graph = MagicMock()
        graph.aget_state = AsyncMock(return_value=SimpleNamespace(values={}, next=()))
        graph.ainvoke = AsyncMock()

        with self.assertRaises(ReviewNotPendingError):
            asyncio.run(TravelAgentService(graph).resume("missing-thread", approved=True))

        graph.ainvoke.assert_not_awaited()

    def test_completed_approval_returns_saved_result_without_rerunning(self):
        graph = MagicMock()
        graph.aget_state = AsyncMock(return_value=SimpleNamespace(
            values={"approved": True, "final_response": "Approved itinerary", "itinerary": "Day 1: Arrival"},
            next=(),
        ))
        graph.ainvoke = AsyncMock()

        result = asyncio.run(TravelAgentService(graph).resume("completed-thread", approved=True))

        self.assertTrue(result["approved"])
        self.assertEqual(result["answer"], "Approved itinerary")
        graph.ainvoke.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
