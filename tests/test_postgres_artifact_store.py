import os

import pytest

from artifact_store import ArtifactStore


def _database_url() -> str | None:
    value = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not value:
        return None
    if "sslmode=" not in value:
        value += "&" if "?" in value else "?"
        value += "sslmode=require"
    return value


@pytest.mark.integration
@pytest.mark.asyncio
async def test_empty_postgres_initializes_artifact_schema_idempotently():
    database_url = _database_url()
    if not database_url:
        pytest.skip("No PostgreSQL database URL configured for integration tests.")

    store = ArtifactStore(database_url=database_url)
    await store.initialize()
    try:
        if not store.persistent:
            pytest.skip("Configured PostgreSQL database is not reachable in this environment.")
        status = await store.readiness()
        assert status["backend"] == "postgresql"
        assert status["persistent"] is True
    finally:
        await store.close()
