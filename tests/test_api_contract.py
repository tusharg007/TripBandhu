import asyncio
import importlib
import json
import sys
import types
import unittest


def load_app_module():
    backend_stub = types.ModuleType("backend")

    backend_stub.run_travel_agent = lambda user_input, thread_id=None: {
        "thread_id": thread_id or "user_test",
        "answer": "Draft itinerary",
        "requires_approval": True,
        "approval_request": "Review this draft.",
        "flight_results": "Flights",
        "hotel_results": "Hotels",
        "weather_results": "Weather",
        "budget_results": "Budget",
        "itinerary": "Draft itinerary",
        "selected_agents": ["flight_agent", "weather_agent", "itinerary_agent"],
        "trip_constraints": {"origin": "Delhi", "destination": "Tokyo"},
        "supervisor_reasoning": "Selected relevant specialists.",
        "guardrail_allowed": True,
        "guardrail_reason": "",
        "approved": None,
        "human_feedback": "",
        "llm_calls": 4,
    }

    backend_stub.resume_travel_agent = lambda thread_id, approved, feedback="": {
        "thread_id": thread_id,
        "answer": "Final plan",
        "requires_approval": False,
        "approval_request": "",
        "flight_results": "Flights",
        "hotel_results": "Hotels",
        "weather_results": "Weather",
        "budget_results": "Budget",
        "itinerary": "Draft itinerary",
        "selected_agents": ["flight_agent", "weather_agent", "itinerary_agent"],
        "trip_constraints": {"origin": "Delhi", "destination": "Tokyo"},
        "supervisor_reasoning": "Selected relevant specialists.",
        "guardrail_allowed": True,
        "guardrail_reason": "",
        "approved": approved,
        "human_feedback": feedback,
        "llm_calls": 5,
    }

    sys.modules["backend"] = backend_stub
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def response_json(response):
    return json.loads(response.body.decode("utf-8"))


class ApiContractTest(unittest.TestCase):
    def test_initial_response_contains_agentic_contract(self):
        app_module = load_app_module()

        response = asyncio.run(
            app_module.travel_planner(
                app_module.TravelRequest(message="Plan Tokyo from Delhi")
            )
        )
        payload = response_json(response)

        self.assertTrue(payload["success"])
        self.assertTrue(payload["requires_approval"])
        self.assertIn("weather_results", payload)
        self.assertIn("budget_results", payload)
        self.assertIn("selected_agents", payload)
        self.assertEqual(payload["trip_constraints"]["origin"], "Delhi")

    def test_resume_endpoint_validates_thread_id(self):
        app_module = load_app_module()

        response = asyncio.run(
            app_module.resume_travel_planner(
                app_module.TravelResumeRequest(thread_id=" ", approved=True)
            )
        )
        payload = response_json(response)

        self.assertEqual(response.status_code, 400)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error_code"], "INVALID_REQUEST")

    def test_resume_endpoint_delegates_and_returns_contract(self):
        app_module = load_app_module()

        response = asyncio.run(
            app_module.resume_travel_planner(
                app_module.TravelResumeRequest(
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

    def test_unexpected_errors_return_safe_public_message(self):
        app_module = load_app_module()

        def failing_run(user_input, thread_id=None):
            raise RuntimeError("DATABASE_URL leaked from F:\\TripBandhu\\.env")

        app_module.run_travel_agent = failing_run

        response = asyncio.run(
            app_module.travel_planner(
                app_module.TravelRequest(message="Plan Tokyo from Delhi")
            )
        )
        payload = response_json(response)

        self.assertEqual(response.status_code, 500)
        self.assertEqual(payload["error_code"], "TRAVEL_PLANNING_FAILED")
        self.assertNotIn("DATABASE_URL", payload["error"])
        self.assertNotIn("F:\\", payload["error"])


if __name__ == "__main__":
    unittest.main()
