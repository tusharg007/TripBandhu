"""Regression tests for clean, complete end-user specialist output."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage
import requests

from capability_registry import (
    normalize_flight_results,
    normalize_tavily_results,
    normalize_weather_results,
)
from location_utils import normalize_weather_location
from schemas import CapabilityHealth


class HotelEvidenceSafetyTest(unittest.TestCase):
    def test_irrelevant_search_results_never_reach_hotel_evidence(self):
        noisy_results = {
            "results": [
                {
                    "title": "Weather in New York City",
                    "url": "https://weatherapi.com/",
                    "content": "{'location': {'name': 'New York'}, 'current': {'temp_c': 23.1}}",
                },
                {
                    "title": "BEST BUY - Updated August 2026",
                    "url": "https://www.yelp.com/biz/best-buy-oklahoma-city-8",
                    "content": "Electronics store opening hours.",
                },
                {
                    "title": "BEST Definition & Meaning",
                    "url": "https://www.merriam-webster.com/dictionary/best",
                    "content": "Definition, rhymes and word of the day.",
                },
                {
                    "title": "The Wallace Hotel New York",
                    "url": "https://example.com/wallace-hotel",
                    "content": "Upper West Side hotel near Central Park with guest rooms and subway access.",
                },
            ]
        }

        evidence = normalize_tavily_results(noisy_results, "hotels in New York City")

        self.assertEqual(evidence.status, CapabilityHealth.AVAILABLE)
        self.assertEqual(len(evidence.data), 1)
        self.assertIn("Wallace Hotel", evidence.summary)
        self.assertNotIn("Best Buy", evidence.summary)
        self.assertNotIn("Merriam-Webster", evidence.summary)
        self.assertNotIn("'location'", evidence.summary)

    def test_only_noise_returns_safe_degraded_message(self):
        evidence = normalize_tavily_results(
            {"results": [{"title": "BEST Definition", "content": "dictionary definition"}]},
            "best hotels",
        )
        self.assertEqual(evidence.status, CapabilityHealth.DEGRADED)
        self.assertEqual(evidence.data, [])
        self.assertIn("No reliable hotel-specific sources", evidence.summary)


class WeatherPresentationSafetyTest(unittest.TestCase):
    def test_weather_aliases_are_provider_friendly(self):
        self.assertEqual(normalize_weather_location("NYC"), "New York City")
        self.assertEqual(normalize_weather_location("DEL"), "Delhi")
        self.assertEqual(normalize_weather_location("Kyoto"), "Kyoto")

    def test_normalizer_formats_weather_without_raw_payloads_or_errors(self):
        evidence = normalize_weather_results(
            {
                "city": "New York",
                "temperature_c": 23.1,
                "feels_like_c": 24.0,
                "humidity": 51,
                "condition": "sunny",
                "wind_speed": 4.1,
            },
            {
                "city": "New York",
                "forecast": [
                    {"datetime": "2026-08-24 15:00", "temperature_c": 25, "condition": "clear sky"}
                ],
            },
            city="New York City",
        )

        self.assertIn("Current weather in New York", evidence.summary)
        self.assertIn("23.1°C", evidence.summary)
        self.assertIn("Near-term forecast", evidence.summary)
        self.assertNotIn("{'city'", evidence.summary)
        self.assertNotIn("api.openweathermap.org", evidence.summary)
        self.assertNotIn("Error executing tool", evidence.summary)

    def test_weather_agent_calls_provider_with_normalized_city(self):
        import backend

        seen_cities = []

        async def current(city):
            seen_cities.append(city)
            return {"city": city, "temperature_c": 20, "condition": "clear"}

        async def forecast(city):
            seen_cities.append(city)
            return {"city": city, "forecast": [{"datetime": "Tomorrow", "temperature_c": 22, "condition": "clear"}]}

        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="Clear weather. Pack a light layer."))
        state = {
            "user_query": "Weather in NYC",
            "trip_constraints": {"destination": "NYC"},
            "specialist_statuses": {},
            "llm_calls": 0,
            "llm_token_usage": {},
        }

        with patch.object(backend, "weather_mcp_search", side_effect=current), \
             patch.object(backend, "forecast_mcp_search", side_effect=forecast), \
             patch.object(backend, "_llm_weather", llm):
            result = asyncio.run(backend.weather_agent(state))

        self.assertEqual(seen_cities, ["New York City", "New York City"])
        self.assertEqual(result["weather_results"], "Clear weather. Pack a light layer.")

    def test_weather_agent_rejects_itinerary_and_budget_contamination(self):
        import backend

        async def current(_city):
            return {
                "city": "Berlin",
                "temperature_c": 22.5,
                "feels_like_c": 22.1,
                "humidity": 50,
                "condition": "scattered clouds",
                "wind_speed": 3.8,
            }

        async def forecast(_city):
            return {
                "city": "Berlin",
                "forecast": [
                    {"datetime": "2026-08-25 06:00", "temperature_c": 14.2, "condition": "clear sky"}
                ],
            }

        contaminated = (
            "# 7-Day Cultural Itinerary\n"
            "## Day 1 — Berlin\nSuggested Hotel: Example Hotel\n"
            "## Budget Overview\nEstimated cost: ₹2,00,000"
        )
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content=contaminated))
        state = {
            "user_query": "Plan a 7-day Germany trip with flights, hotels, weather and budget",
            "trip_constraints": {"destination": "Berlin"},
            "specialist_statuses": {},
            "llm_calls": 4,
            "llm_token_usage": {},
        }

        with patch.object(backend, "weather_mcp_search", side_effect=current), \
             patch.object(backend, "forecast_mcp_search", side_effect=forecast), \
             patch.object(backend, "_llm_weather", llm):
            result = asyncio.run(backend.weather_agent(state))

        weather_text = result["weather_results"]
        self.assertIn("Current weather in Berlin", weather_text)
        self.assertIn("Near-term forecast", weather_text)
        self.assertNotIn("Itinerary", weather_text)
        self.assertNotIn("Budget", weather_text)
        self.assertNotIn("Hotel", weather_text)
        self.assertEqual(result["llm_calls"], 5, "Rejected LLM output must still be accounted for")
        presentation_prompt = llm.ainvoke.await_args.args[0][-1].content
        self.assertNotIn(state["user_query"], presentation_prompt)

    def test_weather_provider_error_never_leaks_api_key_or_request_url(self):
        from custom_weather_mcp_server import _request_json

        response = MagicMock(status_code=404)
        response.json.return_value = {"cod": "404", "message": "city not found"}
        response.raise_for_status.side_effect = requests.HTTPError(
            "404 for https://api.openweathermap.org/data?appid=TOP_SECRET",
            response=response,
        )

        with patch("custom_weather_mcp_server.requests.get", return_value=response):
            with self.assertRaises(RuntimeError) as raised:
                _request_json("https://api.openweathermap.org/data", {"appid": "TOP_SECRET"})

        public_error = str(raised.exception)
        self.assertIn("could not find that location", public_error)
        self.assertNotIn("TOP_SECRET", public_error)
        self.assertNotIn("api.openweathermap.org", public_error)


class CompleteOutputTest(unittest.TestCase):
    def test_flight_normalizer_does_not_character_slice_provider_records(self):
        tail_marker = "TAIL_RECORD_MUST_SURVIVE"
        routes = [{"route": "DEL-JFK", "notes": "x" * 1200 + tail_marker}]
        evidence = normalize_flight_results(routes, [], [], destination="New York", origin="Delhi")
        self.assertIn(tail_marker, evidence.summary)

    def test_final_agent_passes_complete_specialist_and_itinerary_text(self):
        import backend

        markers = {
            "flight_results": "FLIGHT_" + "x" * 1200 + "_FLIGHT_END",
            "hotel_results": "HOTEL_" + "x" * 2200 + "_HOTEL_END",
            "weather_results": "WEATHER_" + "x" * 1200 + "_WEATHER_END",
            "budget_results": "BUDGET_" + "x" * 1400 + "_BUDGET_END",
            "itinerary": "ITINERARY_" + "x" * 3200 + "_DAY_FINAL_END",
        }
        captured = {}

        async def fake_invoke(_llm, messages, **_kwargs):
            captured["prompt"] = messages[-1].content
            return AIMessage(content="Complete final response"), {}

        state = {
            "user_query": "Plan a trip",
            "selected_agents": ["itinerary_agent"],
            "trip_constraints": {},
            "specialist_statuses": {},
            "approved": True,
            "llm_calls": 0,
            "llm_token_usage": {},
            **markers,
        }
        with patch.object(backend, "invoke_llm", side_effect=fake_invoke):
            asyncio.run(backend.final_agent(state))

        for end_marker in ("_FLIGHT_END", "_HOTEL_END", "_WEATHER_END", "_BUDGET_END", "_DAY_FINAL_END"):
            self.assertIn(end_marker, captured["prompt"])

    def test_length_limited_generation_is_continued_and_merged(self):
        from llm_utils import invoke_llm_complete_text

        first = AIMessage(
            content="Day 1 through Day 5",
            response_metadata={
                "finish_reason": "length",
                "token_usage": {"completion_tokens": 100, "total_tokens": 150},
            },
        )
        second = AIMessage(
            content="Day 6 and Day 7. Before you book: verify reservations.",
            response_metadata={
                "finish_reason": "stop",
                "token_usage": {"completion_tokens": 80, "total_tokens": 120},
            },
        )
        runnable = MagicMock()
        runnable.ainvoke = AsyncMock(side_effect=[first, second])

        text, usage, calls = asyncio.run(
            invoke_llm_complete_text(
                runnable,
                [{"role": "user", "content": "Create seven days"}],
                task_name="itinerary",
            )
        )

        self.assertEqual(calls, 2)
        self.assertIn("Day 1 through Day 5", text)
        self.assertIn("Day 6 and Day 7", text)
        self.assertEqual(usage["completion_tokens"], 180)
        self.assertEqual(usage["total_tokens"], 270)
        continuation_messages = runnable.ainvoke.await_args_list[1].args[0]
        self.assertIn("Continue exactly where you stopped", continuation_messages[-1]["content"])

    def test_budget_agent_continues_after_length_limit(self):
        import backend

        first = AIMessage(
            content="## Cost breakdown\nFlights and accommodation.",
            response_metadata={"finish_reason": "length"},
        )
        second = AIMessage(
            content="## Risk areas\nVerify rates.\n## Budget assumptions and verification checklist\nComplete.",
            response_metadata={"finish_reason": "stop"},
        )
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=[first, second])
        state = {
            "user_query": "Budget for a 7-day Germany trip from Delhi",
            "trip_constraints": {"origin": "Delhi", "destination": "Germany", "duration": "7 days"},
            "specialist_statuses": {},
            "evidence_store": {},
            "llm_calls": 3,
            "llm_token_usage": {},
        }

        with patch.object(backend, "_llm_budget", llm):
            result = asyncio.run(backend.budget_agent(state))

        self.assertEqual(result["llm_calls"], 5)
        self.assertIn("Cost breakdown", result["budget_results"])
        self.assertIn("Budget assumptions and verification checklist", result["budget_results"])
        self.assertTrue(result["budget_results"].rstrip().endswith("Complete."))


if __name__ == "__main__":
    unittest.main()
