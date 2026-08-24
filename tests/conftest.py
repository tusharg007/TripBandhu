"""Test-session defaults that keep local and CI runs deterministic."""

import os


# Tests must never export traces or depend on a developer's local .env settings.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
