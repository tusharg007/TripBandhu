"""Application PostgreSQL pools with validation on every checkout.

Keep connections only while doing database work, never across LLM calls or
human review. A disconnected idle connection is replaced by psycopg_pool before
it reaches LangGraph or artifact storage; graph execution is never replayed.
"""

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


def create_database_pool(database_url: str, *, name: str) -> AsyncConnectionPool:
    if not database_url:
        raise ValueError("DATABASE_URL is not configured")
    return AsyncConnectionPool(
        conninfo=database_url,
        name=name,
        open=False,
        min_size=0,
        max_size=4,
        timeout=20,
        max_waiting=16,
        max_idle=60,
        max_lifetime=600,
        reconnect_timeout=30,
        check=AsyncConnectionPool.check_connection,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
            "prepare_threshold": None,
            "connect_timeout": 10,
            "options": "-c statement_timeout=15000",
        },
    )


async def database_pool_ready(pool: AsyncConnectionPool | None) -> bool:
    if pool is None:
        return False
    try:
        async with pool.connection(timeout=5) as connection:
            await connection.execute("SELECT 1")
        return True
    except Exception:
        # Never disclose connection strings or credentials in the health endpoint.
        return False
