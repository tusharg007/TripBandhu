"""
schemas.py  TF1 Typed Pydantic models for the TripBandhu agentic core.

All structured output models used by the supervisor, guardrail, providers,
and HITL review loop are defined here for reuse across the application.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Agent routing
# ---------------------------------------------------------------------------

class AgentName(str, Enum):
    """Canonical set of specialist agent identifiers.

    Only these values may appear in ``selected_agents``.  The supervisor LLM
    is prompted to return names from this set; any other value is rejected at
    validation time.
    """

    FLIGHT = "flight_agent"
    HOTEL = "hotel_agent"
    WEATHER = "weather_agent"
    BUDGET = "budget_agent"
    ITINERARY = "itinerary_agent"


# Ordered execution sequence (mirrors AGENT_ORDER in backend)
AGENT_EXECUTION_ORDER: list[AgentName] = [
    AgentName.FLIGHT,
    AgentName.HOTEL,
    AgentName.WEATHER,
    AgentName.BUDGET,
    AgentName.ITINERARY,
]


# ---------------------------------------------------------------------------
# Travel constraints
# ---------------------------------------------------------------------------

class TravelConstraints(BaseModel):
    """Structured travel constraints extracted from the user request."""

    destination: Optional[str] = Field(default=None, description="Primary travel destination")
    origin: Optional[str] = Field(default=None, description="Departure city or country")
    duration: Optional[str] = Field(default=None, description="Trip duration, e.g. '7 days'")
    budget: Optional[str] = Field(default=None, description="Budget range or limit")
    travel_style: Optional[str] = Field(default=None, description="Travel style, e.g. 'budget', 'luxury'")
    special_preferences: list[str] = Field(
        default_factory=list,
        description="Additional traveller preferences",
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict for downstream prompts and API serialisation."""
        return {
            "destination": self.destination or "",
            "origin": self.origin or "",
            "duration": self.duration or "",
            "budget": self.budget or "",
            "travel_style": self.travel_style or "",
            "special_preferences": self.special_preferences,
        }


# ---------------------------------------------------------------------------
# Supervisor decision
# ---------------------------------------------------------------------------

class SupervisorDecision(BaseModel):
    """Typed output of the supervisor routing step."""

    selected_agents: list[AgentName] = Field(
        description="Specialist agents required for this request",
    )
    constraints: TravelConstraints = Field(
        default_factory=TravelConstraints,
        description="Extracted travel constraints",
    )
    reasoning: str = Field(
        default="",
        description="Concise 1-2 sentence explanation of routing decision",
    )
    needs_clarification: bool = Field(
        default=False,
        description="True if the request is too ambiguous to route confidently",
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Constraint fields that are absent and may limit quality",
    )

    @field_validator("selected_agents", mode="before")
    @classmethod
    def validate_agents(cls, v: list[Any]) -> list[AgentName]:
        """Coerce string values to AgentName, silently dropping unknowns."""
        result: list[AgentName] = []
        seen: set[str] = set()
        for item in v:
            try:
                agent = AgentName(item) if isinstance(item, str) else AgentName(item.value)
                if agent.value not in seen:
                    seen.add(agent.value)
                    result.append(agent)
            except ValueError:
                pass
        return result


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------

class GuardrailCategory(str, Enum):
    """Category assigned by the input guardrail."""

    TRAVEL = "TRAVEL"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    MALFORMED = "MALFORMED"
    UNSAFE = "UNSAFE"


class GuardrailDecision(BaseModel):
    """Typed output of the input guardrail step."""

    allowed: bool = Field(description="Whether the request is permitted")
    reason: str = Field(default="", description="Human-readable explanation")
    category: GuardrailCategory = Field(
        default=GuardrailCategory.TRAVEL,
        description="Classification category",
    )


# ---------------------------------------------------------------------------
# HITL review
# ---------------------------------------------------------------------------

class ReviewDecision(BaseModel):
    """A single human review cycle result stored in the review history."""

    iteration: int = Field(description="0-based review cycle index")
    approved: Optional[bool] = Field(default=None, description="Approval decision")
    feedback: str = Field(default="", description="Revision instructions if not approved")


# ---------------------------------------------------------------------------
# Provider / capability result
# ---------------------------------------------------------------------------

class ErrorCode(str, Enum):
    """Provider failure classification used internally."""

    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    AUTH_CONFIGURATION = "AUTH_CONFIGURATION"
    INTERNAL = "INTERNAL"


# Whether an error code represents a transient condition worth retrying.
RETRYABLE_ERROR_CODES: frozenset[ErrorCode] = frozenset(
    {ErrorCode.TIMEOUT, ErrorCode.RATE_LIMITED, ErrorCode.UNAVAILABLE}
)


T = TypeVar("T")


class CapabilityResult(BaseModel, Generic[T]):
    """Typed wrapper returned by provider utility calls."""

    success: bool
    data: Optional[Any] = None
    error_code: Optional[ErrorCode] = None
    safe_message: Optional[str] = None
    retryable: bool = False
    latency_ms: Optional[int] = None

    @classmethod
    def ok(cls, data: Any, latency_ms: int | None = None) -> "CapabilityResult":
        return cls(success=True, data=data, latency_ms=latency_ms)

    @classmethod
    def fail(
        cls,
        error_code: ErrorCode,
        safe_message: str,
        latency_ms: int | None = None,
    ) -> "CapabilityResult":
        return cls(
            success=False,
            error_code=error_code,
            safe_message=safe_message,
            retryable=error_code in RETRYABLE_ERROR_CODES,
            latency_ms=latency_ms,
        )
