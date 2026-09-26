from unittest.mock import patch

from backend import get_database_url


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
