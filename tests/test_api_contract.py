import asyncio
import json
import unittest
from unittest.mock import patch

import app


def response_json(response):
    return json.loads(response.body.decode("utf-8"))


SAMPLE_RUN_RESPONSE = {
    "thread_id": "user_test",
    "answer": "Draft itinerary",
    "requires_approval": True,
    "approval_request": "Review this draft.",
    "flight_results": "Flights",
    "hotel_results": "Hotels",
    "weather_results": "Weather",
    "budget_results": "Budget",
    "itinerary": "Draft itinerary",
    "selected_agents": ["flight_agent", "weather_agent", "itinerary_agent"],
    "specialist_statuses": {
        "flight_agent": "COMPLETED",
        "weather_agent": "COMPLETED",
        "itinerary_agent": "COMPLETED",
        "hotel_agent": "NOT_SELECTED",
        "budget_agent": "NOT_SELECTED",
    },
    "trip_constraints": {"origin": "Delhi", "destination": "Tokyo"},
    "supervisor_reasoning": "Selected relevant specialists.",
    "guardrail_allowed": True,
    "guardrail_reason": "",
    "approved": None,
    "human_feedback": "",
    "llm_calls": 4,
}

SAMPLE_RESUME_RESPONSE = {
    "thread_id": "user_test",
    "answer": "Final plan",
    "requires_approval": False,
    "approval_request": "",
    "flight_results": "Flights",
    "hotel_results": "Hotels",
    "weather_results": "Weather",
    "budget_results": "Budget",
    "itinerary": "Draft itinerary",
    "selected_agents": ["flight_agent", "weather_agent", "itinerary_agent"],
    "specialist_statuses": {
        "flight_agent": "COMPLETED",
        "weather_agent": "COMPLETED",
        "itinerary_agent": "COMPLETED",
        "hotel_agent": "NOT_SELECTED",
        "budget_agent": "NOT_SELECTED",
    },
    "trip_constraints": {"origin": "Delhi", "destination": "Tokyo"},
    "supervisor_reasoning": "Selected relevant specialists.",
    "guardrail_allowed": True,
    "guardrail_reason": "",
    "approved": False,
    "human_feedback": "Add slower pacing.",
    "llm_calls": 5,
}


class ApiContractTest(unittest.TestCase):
    def test_initial_response_contains_agentic_contract(self):
        with patch("app.run_travel_agent", return_value=SAMPLE_RUN_RESPONSE):
            response = asyncio.run(
                app.travel_planner(
                    app.TravelRequest(message="Plan Tokyo from Delhi")
                )
            )
            payload = response_json(response)

            self.assertTrue(payload["success"])
            self.assertTrue(payload["requires_approval"])
            self.assertIn("weather_results", payload)
            self.assertIn("budget_results", payload)
            self.assertIn("selected_agents", payload)
            self.assertIn("specialist_statuses", payload)
            self.assertEqual(payload["trip_constraints"]["origin"], "Delhi")

    def test_resume_endpoint_validates_thread_id(self):
        response = asyncio.run(
            app.resume_travel_planner(
                app.TravelResumeRequest(thread_id=" ", approved=True)
            )
        )
        payload = response_json(response)

        self.assertEqual(response.status_code, 400)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "INVALID_REQUEST")

    def test_resume_endpoint_delegates_and_returns_contract(self):
        with patch("app.resume_travel_agent", return_value=SAMPLE_RESUME_RESPONSE):
            response = asyncio.run(
                app.resume_travel_planner(
                    app.TravelResumeRequest(
                        thread_id="user_test",
                        approved=False,
                        feedback="Add slower pacing.",
                    )
                )
            )
            payload = response_json(response)

            self.assertTrue(payload["success"])
            self.assertFalse(payload["requires_approval"])
            self.assertFalse(payload["approved"])
            self.assertEqual(payload["human_feedback"], "Add slower pacing.")
            self.assertEqual(payload["answer"], "Final plan")
            self.assertIn("specialist_statuses", payload)

    def test_unexpected_errors_return_safe_public_message(self):
        def failing_run(user_input, thread_id=None):
            raise RuntimeError("DATABASE_URL leaked from F:\\TripBandhu\\.env")

        with patch("app.run_travel_agent", side_effect=failing_run):
            response = asyncio.run(
                app.travel_planner(
                    app.TravelRequest(message="Plan Tokyo from Delhi")
                )
            )
            payload = response_json(response)

            self.assertEqual(response.status_code, 500)
            self.assertEqual(payload["error_code"], "TRAVEL_PLANNING_FAILED")
            self.assertNotIn("DATABASE_URL", payload["error"])
            self.assertNotIn("F:\\", payload["error"])


if __name__ == "__main__":
    unittest.main()
