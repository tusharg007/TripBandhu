"""
test_postgres_checkpoint_restart.py — Integration test proving AsyncPostgresSaver
preserves multi-turn state across service/graph/process recreations.
"""

import asyncio
import os
import sys
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import backend
from schemas import (
    AgentName,
    GuardrailCategory,
    GuardrailDecision,
    SupervisorDecision,
    TravelConstraints,
)


def _get_integration_db_url() -> str | None:
    """Return database URL for integration testing, or None if unavailable."""
    url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        return None
    if "sslmode=" not in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}sslmode=require"
    return url


async def execute_postgres_checkpoint_restart_flow(db_url: str) -> bool:
    """Core async verification flow testing checkpoint persistence across recreations."""
    unique_suffix = uuid.uuid4().hex[:8]
    thread_id = f"tf1-postgres-restart-{unique_suffix}"

    allowed = GuardrailDecision(allowed=True, reason="travel", category=GuardrailCategory.TRAVEL)
    supervisor_decision = SupervisorDecision(
        selected_agents=[
            AgentName.FLIGHT,
            AgentName.HOTEL,
            AgentName.WEATHER,
            AgentName.BUDGET,
            AgentName.ITINERARY,
        ],
        constraints=TravelConstraints(destination="Tokyo", origin="Delhi", duration="5 days"),
        reasoning="Full 5-day Tokyo trip planning.",
    )

    async def _mock_aviation(tool_name):
        return ["HND", "NRT"]

    async def _mock_tavily(query):
        return "Recommended Hotels: Hotel Gracery Shinjuku"

    async def _mock_weather(city):
        return "Tokyo: 19C, Mild spring weather"

    async def _mock_forecast(city):
        return "7-day forecast: Clear skies"

    async def _mock_extract(query):
        return "Tokyo"

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(content="Direct flights on ANA and JAL"),                        # flight_agent
        AIMessage(content="Budget feasibility: INR 1.8 lakh is realistic"),        # budget_agent
        AIMessage(content="### Draft Itinerary v0\nDay 1: Tokyo Arrival..."),       # itinerary_agent
        AIMessage(content="### Revised Itinerary v1\nDay 1: Relaxed pacing..."),    # itinerary_revision 1
        AIMessage(content="### Revised Itinerary v2\nDay 1: Local food market..."), # itinerary_revision 2
        AIMessage(content="### Final Approved Travel Plan for Tokyo\nApproved!"),  # final_agent
    ])

    with patch.object(backend, "_llm_guardrail") as mg, \
         patch.object(backend, "_llm_supervisor") as ms, \
         patch.object(backend, "aviation_mcp_call", side_effect=_mock_aviation), \
         patch.object(backend, "tavily_mcp_search", side_effect=_mock_tavily), \
         patch.object(backend, "weather_mcp_search", side_effect=_mock_weather), \
         patch.object(backend, "forecast_mcp_search", side_effect=_mock_forecast), \
         patch.object(backend, "extract_destination_async", side_effect=_mock_extract), \
         patch.object(backend, "llm", mock_llm):

        mg.ainvoke = AsyncMock(return_value=allowed)
        ms.ainvoke = AsyncMock(return_value=supervisor_decision)

        # Turn 1: Service Instance #1
        async with AsyncPostgresSaver.from_conn_string(db_url) as checkpointer1:
            await checkpointer1.setup()
            service1 = backend.create_travel_service(checkpointer1)
            r1 = await service1.run("Plan a 5-day trip to Tokyo from Delhi", thread_id=thread_id)

            assert r1["requires_approval"] is True
            assert r1["review_iteration"] == 0
            assert "Draft Itinerary v0" in r1["itinerary"]

        # Service #1 is now completely destroyed

        # Turn 2: Service Instance #2 -> Resume Revision 1
        async with AsyncPostgresSaver.from_conn_string(db_url) as checkpointer2:
            await checkpointer2.setup()
            service2 = backend.create_travel_service(checkpointer2)
            r2 = await service2.resume(
                thread_id=thread_id,
                approved=False,
                feedback="Make the itinerary less rushed.",
            )

            assert r2["requires_approval"] is True
            assert r2["review_iteration"] == 1
            assert "Revised Itinerary v1" in r2["itinerary"]

        # Service #2 is now completely destroyed

        # Turn 3: Service Instance #3 -> Resume Revision 2
        async with AsyncPostgresSaver.from_conn_string(db_url) as checkpointer3:
            await checkpointer3.setup()
            service3 = backend.create_travel_service(checkpointer3)
            r3 = await service3.resume(
                thread_id=thread_id,
                approved=False,
                feedback="Add more local food experiences.",
            )

            assert r3["requires_approval"] is True
            assert r3["review_iteration"] == 2
            assert "Revised Itinerary v2" in r3["itinerary"]

        # Service #3 is now completely destroyed

        # Turn 4: Service Instance #4 -> Final Approval
        async with AsyncPostgresSaver.from_conn_string(db_url) as checkpointer4:
            await checkpointer4.setup()
            service4 = backend.create_travel_service(checkpointer4)
            r4 = await service4.resume(
                thread_id=thread_id,
                approved=True,
                feedback="",
            )

            assert r4["requires_approval"] is False
            assert r4["approved"] is True
            assert "Final Approved Travel Plan" in r4["answer"]

            # Completed thread cannot resume with duplicate execution
            r5 = await service4.resume(
                thread_id=thread_id,
                approved=True,
                feedback="Try again",
            )
            assert r5["requires_approval"] is False
            assert r5["approved"] is True

            # Thread isolation: create thread B
            thread_id_b = f"tf1-postgres-restart-isolated-{unique_suffix}"
            supervisor_decision_b = SupervisorDecision(
                selected_agents=[AgentName.FLIGHT],
                constraints=TravelConstraints(destination="Paris", origin="London"),
                reasoning="Flight-only query to Paris.",
            )
            ms.ainvoke = AsyncMock(return_value=supervisor_decision_b)
            mock_llm.ainvoke = AsyncMock(side_effect=[
                AIMessage(content="Direct Flights to Paris"),
                AIMessage(content="Final Flight Summary for Paris"),
            ])

            rb = await service4.run("Find flights from London to Paris", thread_id=thread_id_b)
            assert rb["trip_constraints"]["destination"] == "Paris"
            assert rb["trip_constraints"]["origin"] == "London"
            assert rb["selected_agents"] == ["flight_agent"]
            assert rb["requires_approval"] is False

    return True


@pytest.mark.integration
class TestPostgresCheckpointRestartPersistence:

    @pytest.fixture(autouse=True)
    def check_db_availability(self):
        db_url = _get_integration_db_url()
        if not db_url:
            pytest.skip("No PostgreSQL database URL configured for integration tests.")

    def test_postgres_checkpoint_survives_service_recreation(self):
        """Prove that state/interrupts persist across full service & checkpointer disposals."""
        db_url = _get_integration_db_url()
        asyncio.run(execute_postgres_checkpoint_restart_flow(db_url))


if __name__ == "__main__":
    url = _get_integration_db_url()
    if url:
        asyncio.run(execute_postgres_checkpoint_restart_flow(url))
        print("Postgres checkpoint restart verification passed!")
    else:
        print("DATABASE_URL not set")
