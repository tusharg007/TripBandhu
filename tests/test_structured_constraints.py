
"""TF1: Structured travel constraint extraction tests."""
import unittest
from schemas import TravelConstraints, SupervisorDecision, AgentName


class StructuredConstraintsTest(unittest.TestCase):

    def test_complete_constraint_extraction(self):
        """All fields present should round-trip cleanly via to_dict."""
        constraints = TravelConstraints(
            destination="Tokyo",
            origin="Delhi",
            duration="7 days",
            budget="INR 2 lakh",
            travel_style="budget",
            special_preferences=["sightseeing", "vegetarian food"],
        )
        d = constraints.to_dict()
        self.assertEqual(d["destination"], "Tokyo")
        self.assertEqual(d["origin"], "Delhi")
        self.assertEqual(d["duration"], "7 days")
        self.assertEqual(d["budget"], "INR 2 lakh")
        self.assertEqual(d["travel_style"], "budget")
        self.assertIn("sightseeing", d["special_preferences"])

    def test_partial_constraints(self):
        """Partial constraints should return empty string for missing string fields."""
        constraints = TravelConstraints(destination="Paris")
        d = constraints.to_dict()
        self.assertEqual(d["destination"], "Paris")
        self.assertEqual(d["origin"], "")
        self.assertEqual(d["duration"], "")
        self.assertEqual(d["budget"], "")
        self.assertEqual(d["special_preferences"], [])

    def test_missing_optional_fields(self):
        """All-None constraints should produce empty strings, not None values."""
        constraints = TravelConstraints()
        d = constraints.to_dict()
        for field in ["destination", "origin", "duration", "budget", "travel_style"]:
            self.assertEqual(d[field], "", f"Expected empty string for {field}")
        self.assertEqual(d["special_preferences"], [])

    def test_constraints_in_supervisor_decision(self):
        """SupervisorDecision can hold a TravelConstraints and serialize it."""
        decision = SupervisorDecision(
            selected_agents=[AgentName.ITINERARY],
            constraints=TravelConstraints(destination="Rome", origin="London"),
            reasoning="Rome trip planning.",
        )
        d = decision.constraints.to_dict()
        self.assertEqual(d["destination"], "Rome")
        self.assertEqual(d["origin"], "London")

    def test_none_fields_are_truly_optional(self):
        """Optional fields accept None without validation errors."""
        constraints = TravelConstraints(
            destination="Bali",
            origin=None,
            budget=None,
        )
        self.assertIsNone(constraints.origin)
        self.assertIsNone(constraints.budget)
        self.assertEqual(constraints.to_dict()["origin"], "")


if __name__ == "__main__":
    unittest.main()
