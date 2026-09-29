"""Exercise connection loss on a real, explicitly selected test database.

TEST_DATABASE_URL must point at an isolated database. We terminate only the
backend PID borrowed by our own test pool, never arbitrary production sessions.
"""

import os
import uuid
from typing import TypedDict
from unittest.mock import AsyncMock, patch

import pytest
import app as application  # Apply the app's Windows loop policy before async tests.
from fastapi import FastAPI, Request
from psycopg import AsyncConnection, OperationalError
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

from artifact_store import ArtifactStore
from database import create_database_pool, database_pool_ready
from itinerary_document import build_itinerary_document


class ReviewState(TypedDict):
    draft: str
    approved: bool


def review(state):
    approved = interrupt("Approve this itinerary?")
    return {"approved": approved}


def build_graph(saver):
    graph = StateGraph(ReviewState)
    graph.add_node("review", review)
    graph.add_edge(START, "review")
    graph.add_edge("review", END)
    return graph.compile(checkpointer=saver)


@pytest.fixture
def isolated_database_url():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to an isolated PostgreSQL test database.")
    return url


async def kill_idle_connection(pool, url):
    async with pool.connection() as connection:
        pid = connection.info.backend_pid
    async with await AsyncConnection.connect(url, autocommit=True) as admin:
        cursor = await admin.execute("SELECT pg_terminate_backend(%s)", (pid,))
        assert (await cursor.fetchone())[0] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_original_single_connection_reproduces_reported_failure(isolated_database_url):
    async with AsyncPostgresSaver.from_conn_string(isolated_database_url) as saver:
        await saver.setup()
        await saver.conn.close()
        with pytest.raises(OperationalError, match="connection is closed"):
            await saver.aget_tuple({"configurable": {"thread_id": "closed-connection-repro"}})


@pytest.mark.integration
@pytest.mark.asyncio
async def test_checkpoint_read_review_and_restart_survive_idle_disconnect(isolated_database_url):
    pool = create_database_pool(isolated_database_url, name="test-checkpoint-recovery")
    await pool.open()
    config = {"configurable": {"thread_id": "recovery-" + uuid.uuid4().hex}}
    try:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        # Regression: first request after a long idle period must start normally.
        await kill_idle_connection(pool, isolated_database_url)
        graph = build_graph(saver)
        draft = await graph.ainvoke({"draft": "Day 1: Tokyo", "approved": False}, config)
        assert draft["__interrupt__"]
        # A user may spend minutes reviewing before approving the draft.
        await kill_idle_connection(pool, isolated_database_url)
        assert (await graph.aget_state(config)).next == ("review",)
        await kill_idle_connection(pool, isolated_database_url)
        approved = await graph.ainvoke(Command(resume=True), config)
        assert approved["approved"] is True
        assert approved["draft"] == "Day 1: Tokyo"
        assert await database_pool_ready(pool)
    finally:
        await pool.close()
    assert not await database_pool_ready(pool)
    replacement = create_database_pool(isolated_database_url, name="test-checkpoint-restart")
    async with replacement:
        saved = await build_graph(AsyncPostgresSaver(replacement)).aget_state(config)
        assert saved.values["approved"] is True
        assert saved.next == ()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_artifact_ownership_and_cached_pdf_survive_disconnects(isolated_database_url):
    store = ArtifactStore(isolated_database_url)
    await store.initialize()
    assert store.persistent, "Explicit test database failed to initialize"
    thread = "artifact-recovery-" + uuid.uuid4().hex
    document = build_itinerary_document(
        thread_id=thread, markdown="### Day 1 - Tokyo\n- Visit a local market.",
        constraints={"destination": "Tokyo", "duration": "1 day"}, approved=True,
    )
    try:
        await kill_idle_connection(store._pool, isolated_database_url)
        await store.bind_thread("owner", thread)
        await kill_idle_connection(store._pool, isolated_database_url)
        assert await store.owns_thread("owner", thread)
        assert not await store.owns_thread("someone-else", thread)
        await kill_idle_connection(store._pool, isolated_database_url)
        saved = await store.save_approved("owner", document)
        await kill_idle_connection(store._pool, isolated_database_url)
        _, pdf = await store.get_or_create_pdf("owner", saved.plan_id, 1, "r1", "a1", lambda _: b"%PDF-test")
        assert pdf == b"%PDF-test"
        await kill_idle_connection(store._pool, isolated_database_url)
        _, cached = await store.get_or_create_pdf("owner", saved.plan_id, 1, "r1", "a1", lambda _: pytest.fail("PDF rendered twice"))
        assert cached == pdf
        assert await store.get_owned_plan("someone-else", saved.plan_id, 1) is None
        await kill_idle_connection(store._pool, isolated_database_url)
        assert (await store.readiness())["ready"]
    finally:
        await store.close()
    assert (await store.readiness())["ready"] is False


@pytest.mark.asyncio
async def test_failed_artifact_initialization_closes_pool():
    pool = AsyncMock()
    with patch("artifact_store.create_database_pool", return_value=pool), \
         patch.object(ArtifactStore, "_execute", AsyncMock(side_effect=OperationalError("offline"))):
        store = ArtifactStore("postgresql://example.invalid/test")
        await store.initialize()
    pool.close.assert_awaited_once()
    assert store._pool is None
    assert store.persistent is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_application_lifespan_uses_recoverable_pools(isolated_database_url):
    test_app = FastAPI()
    with patch.object(application, "DATABASE_URL", isolated_database_url), \
         patch.object(application, "start_provider_clients", AsyncMock()), \
         patch.object(application, "close_provider_clients", AsyncMock()), \
         patch.object(application, "session_security_status", return_value={"configured": True, "source": "external"}):
        async with application.lifespan(test_app):
            pool = test_app.state.checkpoint_pool
            store = test_app.state.artifact_store
            graph = test_app.state.travel_service.graph
            assert graph.checkpointer.conn is pool
            assert store.persistent
            await kill_idle_connection(pool, isolated_database_url)
            snapshot = await graph.aget_state({"configurable": {"thread_id": "new-" + uuid.uuid4().hex}})
            assert snapshot.values == {}
            await kill_idle_connection(store._pool, isolated_database_url)
            request = Request({"type": "http", "app": test_app})
            assert (await application.readiness_check(request)).status_code == 200
        assert pool.closed
        assert store._pool.closed
