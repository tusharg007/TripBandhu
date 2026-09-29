from unittest.mock import patch

from backend import get_database_url
from database import create_database_pool


def test_database_url_adds_ssl_without_changing_provider_or_credentials():
    raw = "postgresql://user@example.invalid:5432/tripbandhu"
    with patch.dict("os.environ", {"DATABASE_URL": raw}, clear=False):
        normalized = get_database_url()

    assert normalized == f"{raw}?sslmode=require"
    assert "example.invalid" in normalized
    assert "sslmode=require" in normalized


def test_database_url_preserves_existing_sslmode_and_query_parameters():
    raw = "postgresql://user@example.invalid:5432/tripbandhu?sslmode=require&channel_binding=require"
    with patch.dict("os.environ", {"DATABASE_URL": raw}, clear=False):
        assert get_database_url() == raw


def test_database_pool_does_not_send_neon_pooler_incompatible_startup_options():
    pooled_url = (
        "postgresql://example-user@example-pooler.ap-southeast-1.aws.neon.tech/"
        "neondb?sslmode=require"
    )
    with patch("database.AsyncConnectionPool") as pool_constructor:
        created = create_database_pool(pooled_url, name="test-neon-pooler")

    assert created is pool_constructor.return_value
    connection_kwargs = pool_constructor.call_args.kwargs["kwargs"]
    assert "options" not in connection_kwargs
    assert connection_kwargs["prepare_threshold"] is None
    assert connection_kwargs["connect_timeout"] == 10
