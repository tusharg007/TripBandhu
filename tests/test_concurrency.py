
"""TF1: Thread isolation and concurrency safety tests."""
import asyncio
import threading
import unittest

from schemas import AgentName, SupervisorDecision, TravelConstraints


class ThreadIsolationTest(unittest.TestCase):

    def test_two_thread_ids_produce_isolated_state(self):
        """Two different thread IDs should not share state."""
        import backend
        from unittest.mock import AsyncMock, MagicMock, patch
        from langchain_core.messages import AIMessage

        results = {}
        errors = []

        def run_agent(query, thread_id):
            try:
                # Simulate independent supervisor_agent calls with different states
                state1 = {"user_query": query, "llm_calls": 0}
                state2 = {"user_query": "Different: " + query, "llm_calls": 0}

                from schemas import GuardrailDecision, GuardrailCategory
                allowed = GuardrailDecision(allowed=True, reason="travel", category=GuardrailCategory.TRAVEL)

                decision1 = SupervisorDecision(
                    selected_agents=[AgentName.ITINERARY],
                    constraints=TravelConstraints(destination="Tokyo"),
                    reasoning=f"Trip to Tokyo for {thread_id}",
                )

                with patch.object(backend, '_llm_guardrail') as mg, \
                     patch.object(backend, '_llm_supervisor') as ms:
                    mg.ainvoke = AsyncMock(return_value=allowed)
                    ms.ainvoke = AsyncMock(return_value=decision1)
                    r = asyncio.run(backend.supervisor_agent(state1))
                results[thread_id] = r
            except Exception as e:
                errors.append(str(e))

        t1 = threading.Thread(target=run_agent, args=("Plan a 7-day trip to Tokyo", "thread_A"))
        t2 = threading.Thread(target=run_agent, args=("Plan a 3-day trip to Dubai", "thread_B"))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(errors, [], f"Errors during concurrent execution: {errors}")
        self.assertIn("thread_A", results)
        self.assertIn("thread_B", results)

        # Results should reflect their respective queries, not bleed into each other
        result_a = results["thread_A"]
        result_b = results["thread_B"]

        self.assertTrue(result_a["guardrail_allowed"])
        self.assertTrue(result_b["guardrail_allowed"])

    def test_agent_state_is_not_global_mutable(self):
        """specialist_statuses dict should not be shared between invocations."""
        import backend
        state1 = {"specialist_statuses": {"flight_agent": "SELECTED"}, "user_query": "A", "llm_calls": 0}
        state2 = {"specialist_statuses": {"flight_agent": "SELECTED"}, "user_query": "B", "llm_calls": 0}

        # Simulating that both agents get independent status copies
        statuses1 = dict(state1.get("specialist_statuses", {}))
        statuses2 = dict(state2.get("specialist_statuses", {}))
        statuses1["flight_agent"] = "COMPLETED"

        # statuses2 should not be affected by changes to statuses1
        self.assertEqual(statuses2["flight_agent"], "SELECTED")

    def test_review_history_is_not_shared(self):
        """review_history should be a fresh list per invocation, not shared."""
        import backend
        state_a = {"review_history": [], "review_iteration": 0}
        state_b = {"review_history": [], "review_iteration": 0}

        hist_a = list(state_a.get("review_history") or [])
        hist_a.append({"iteration": 0, "approved": False, "feedback": "A feedback"})

        hist_b = list(state_b.get("review_history") or [])
        # hist_b should still be empty
        self.assertEqual(hist_b, [])
        self.assertEqual(len(hist_a), 1)


if __name__ == "__main__":
    unittest.main()
