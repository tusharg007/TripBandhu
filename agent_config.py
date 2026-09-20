"""
agent_config.py — Centralized configuration for TripBandhu.

Covers: provider timeouts, retry settings, HITL caps, model IDs, token budgets,
reasoning parameters, and LLM rate-limit handling thresholds.

No magic numbers should be scattered across agent or provider code.
All tuneable knobs live here and are env-overridable where appropriate.
"""

import os

# ---------------------------------------------------------------------------
# Model IDs (env-configurable; defaults = confirmed Groq free-tier IDs)
# ---------------------------------------------------------------------------
# GROQ_CONTROL_MODEL: lighter model used for routing/classification tasks.
# GROQ_GENERATION_MODEL: larger model used for content synthesis tasks.
#
# Set in environment (or .env) to override at runtime without code changes.
GROQ_CONTROL_MODEL: str = os.getenv("GROQ_CONTROL_MODEL", "openai/gpt-oss-20b")
GROQ_GENERATION_MODEL: str = os.getenv("GROQ_GENERATION_MODEL", "openai/gpt-oss-120b")

# GPT-OSS reasoning settings — low effort, chain-of-thought hidden from output.
# Reduces token consumption while keeping routing and synthesis quality acceptable.
# Raise effort per-task only if evaluations reveal quality regressions.
LLM_REASONING_EFFORT: str = os.getenv("LLM_REASONING_EFFORT", "low")
LLM_REASONING_FORMAT: str = os.getenv("LLM_REASONING_FORMAT", "hidden")

# ---------------------------------------------------------------------------
# Per-task output token budgets
# ---------------------------------------------------------------------------
# Output budgets are sized so user-facing specialists can finish complete sections.
# Provider evidence is filtered before prompting; generated answers are not cut with
# character slicing after generation.
MAX_TOKENS_BY_TASK: dict[str, int] = {
    "guardrail":            350,
    "supervisor":           600,
    "flight_summary":      1800,
    "hotel_summary":       1400,
    "weather_summary":      900,
    "budget":              2400,
    "itinerary":           3600,
    "revision":            3600,
    "final_synthesis":     3800,
    "destination_extract":   64,   # Control model: destination name only
}

# A generation that explicitly stops because it reached max_tokens may continue in
# bounded follow-up segments. This prevents half-sentences without allowing loops.
MAX_LLM_CONTINUATIONS: int = int(os.getenv("MAX_LLM_CONTINUATIONS", "2"))

# ---------------------------------------------------------------------------
# Provider timeouts (seconds)
# ---------------------------------------------------------------------------
# Each external provider call is wrapped in asyncio.timeout() using these values.
PROVIDER_TIMEOUT_SECONDS: dict[str, float] = {
    "aviation": 22.0,          # Direct AviationStack HTTPS request
    "tavily":   15.0,          # Direct Tavily HTTPS search
    "weather":  15.0,          # Direct OpenWeather geocode/current/forecast calls
    "llm":      45.0,          # Generation model (120B) — may be slower than 70B
    "llm_structured": 30.0,    # Control model structured outputs
}

# Provider runtime controls. Cache keys never include credentials, and only payloads
# that pass adapter validation are cached.
PROVIDER_CACHE_TTL_SECONDS: dict[str, float] = {
    "aviation": 900.0,
    "tavily": 21600.0,
    "geocode": 2592000.0,
    "weather_current": 600.0,
    "weather_forecast": 1800.0,
}
PROVIDER_CONCURRENCY_LIMITS: dict[str, int] = {
    "aviation": 2,
    "tavily": 2,
    "weather": 4,
}
PROVIDER_CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 3
PROVIDER_CIRCUIT_BREAKER_COOLDOWN_SECONDS: float = 30.0

# "direct" is the production path. "mcp" remains available for compatibility
# comparisons and controlled diagnostics, not as an automatic duplicate fallback.
PROVIDER_TRANSPORT: str = os.getenv("PROVIDER_TRANSPORT", "direct").strip().casefold()

# ---------------------------------------------------------------------------
# Retry settings (all provider transports via provider_utils)
# ---------------------------------------------------------------------------
RETRY_MAX_ATTEMPTS: int = 2          # At most one retry for classified transient failures
RETRY_BASE_DELAY_SECONDS: float = 0.5

# ---------------------------------------------------------------------------
# LLM rate-limit thresholds (centralized LLM invocation via llm_utils)
# ---------------------------------------------------------------------------
# If Retry-After <= SHORT_WAIT_MAX: attempt one bounded retry.
# If Retry-After > SHORT_WAIT_MAX: degrade immediately (do not hold the user open).
RATE_LIMIT_SHORT_WAIT_MAX_SECONDS: int = 15
RATE_LIMIT_DEFAULT_RETRY_WAIT_SECONDS: float = 2.0   # When Retry-After header absent

# ---------------------------------------------------------------------------
# HITL review loop
# ---------------------------------------------------------------------------
MAX_REVIEW_ITERATIONS: int = 10

# ---------------------------------------------------------------------------
# HTTP application safeguards
# ---------------------------------------------------------------------------
# These controls prevent a single browser session from exhausting shared free-tier
# provider quotas. They are intentionally modest and env-overridable.
MAX_CONCURRENT_TRIPS: int = max(1, int(os.getenv("MAX_CONCURRENT_TRIPS", "2")))
SESSION_REQUEST_LIMIT: int = max(1, int(os.getenv("SESSION_REQUEST_LIMIT", "8")))
SESSION_REQUEST_WINDOW_SECONDS: int = max(60, int(os.getenv("SESSION_REQUEST_WINDOW_SECONDS", "3600")))
REQUIRE_PERSISTENT_STORAGE: bool = os.getenv("REQUIRE_PERSISTENT_STORAGE", "false").strip().casefold() in {"1", "true", "yes"}
