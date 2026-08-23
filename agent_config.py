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
# Values are conservative: observed outputs + ~25% safety margin.
# Guardrail/supervisor use the control model; others use the generation model.
# Do NOT lower these below observed real output lengths.
MAX_TOKENS_BY_TASK: dict[str, int] = {
    "guardrail":            350,
    "supervisor":           600,
    "flight_summary":       900,
    "budget":               900,
    "itinerary":           2200,   # gpt-oss-120b: 8k TPM limit; ~5k input + 2200 output fits
    "revision":            2200,   # same
    "final_synthesis":     2800,   # gpt-oss-20b: 12k TPM limit; ~5k input + 2800 output fits
    "destination_extract":   64,   # Control model: destination name only
}

# ---------------------------------------------------------------------------
# Provider timeouts (seconds)
# ---------------------------------------------------------------------------
# Each external provider call is wrapped in asyncio.timeout() using these values.
PROVIDER_TIMEOUT_SECONDS: dict[str, float] = {
    "aviation": 45.0,          # AviationStack MCP — uvx cold-start + API call on Render
    "tavily":   15.0,          # Tavily HTTP search
    "weather":  25.0,          # Custom weather MCP — must exceed server's internal 20s timeout
    "llm":      45.0,          # Generation model (120B) — may be slower than 70B
    "llm_structured": 30.0,    # Control model structured outputs
}

# ---------------------------------------------------------------------------
# Retry settings (MCP providers via provider_utils)
# ---------------------------------------------------------------------------
RETRY_MAX_ATTEMPTS: int = 1          # Single attempt, no retry
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
