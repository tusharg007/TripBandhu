"""
agent_config.py  TF1 centralized configuration for timeouts and retry settings.

All timeout values, retry limits, and review caps are defined here.
No magic numbers should be scattered across specialist or provider code.
"""

# ---------------------------------------------------------------------------
# Provider timeouts (seconds)
# ---------------------------------------------------------------------------
# Each external provider call is wrapped in asyncio.timeout() using these values.
# Adjust per environment; these defaults are conservative for a hosted LLM/MCP stack.

PROVIDER_TIMEOUT_SECONDS: dict[str, float] = {
    "aviation": 15.0,   # AviationStack MCP (subprocess stdio)
    "tavily": 12.0,     # Tavily HTTP search
    "weather": 10.0,    # Custom weather MCP server (subprocess stdio)
    "llm": 30.0,        # Groq LLM API calls
    "llm_structured": 30.0,  # Structured output LLM calls
}


# ---------------------------------------------------------------------------
# Retry settings
# ---------------------------------------------------------------------------
# Applies only to RETRYABLE error codes (TIMEOUT, RATE_LIMITED, UNAVAILABLE).
# Authentication, configuration, and parser errors are NEVER retried.

RETRY_MAX_ATTEMPTS: int = 2          # Total attempts (1 initial + 1 retry)
RETRY_BASE_DELAY_SECONDS: float = 0.5  # Fixed delay between retries (no exponential backoff needed)


# ---------------------------------------------------------------------------
# HITL review loop
# ---------------------------------------------------------------------------
# Generous cap to prevent accidental infinite graph execution if the review
# node is reached without a human interrupt in test/mock environments.
# The system NEVER silently auto-approves when the cap is hit.

MAX_REVIEW_ITERATIONS: int = 10
