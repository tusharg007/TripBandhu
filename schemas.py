"""
schemas.py — Typed Pydantic models for the TripBandhu agentic core.

Includes all structured output models used by the supervisor, guardrail,
providers, HITL review loop, capability evidence, provenance, and evaluation layer.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Generic, Optional, TypeVar
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Agent routing
# ---------------------------------------------------------------------------

class AgentName(str, Enum):
    """Canonical set of specialist agent identifiers."""

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
# Provider / capability result & error codes
# ---------------------------------------------------------------------------

class ErrorCode(str, Enum):
    """Provider failure classification used internally."""

    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    AUTH_CONFIGURATION = "AUTH_CONFIGURATION"
    CONTRACT_MISMATCH = "CONTRACT_MISMATCH"
    INTERNAL = "INTERNAL"


RETRYABLE_ERROR_CODES: frozenset[ErrorCode] = frozenset(
    {ErrorCode.TIMEOUT, ErrorCode.RATE_LIMITED, ErrorCode.UNAVAILABLE}
)


class EvidenceKind(str, Enum):
    """Classification of evidence provenance and trustworthiness."""

    PROVIDER_DATA = "PROVIDER_DATA"      # Provider-backed structured API data
    LIVE_PROVIDER = "PROVIDER_DATA"      # Backward compatibility alias
    WEB_SOURCE = "WEB_SOURCE"            # Search engine web results
    MODEL_ESTIMATE = "MODEL_ESTIMATE"    # Model generated range / guidance (not live fare)
    DERIVED = "DERIVED"                  # Derived from verified evidence
    UNAVAILABLE = "UNAVAILABLE"          # Provider unavailable / degraded


class EvidenceFreshness(str, Enum):
    """Temporal and operational semantics of evidence."""

    LIVE = "LIVE"                  # Real-time operational data (e.g. current weather observation)
    FORECAST = "FORECAST"          # Multi-day forecast model data
    REFERENCE = "REFERENCE"        # Reference/schedule database (e.g. routes, airports, airlines)
    UNKNOWN = "UNKNOWN"            # Unspecified temporal semantics


class CapabilityHealth(str, Enum):
    """Health state of an external MCP capability."""

    AVAILABLE = "AVAILABLE"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    CONTRACT_MISMATCH = "CONTRACT_MISMATCH"


class SourceReference(BaseModel):
    """A traceable source reference for evidence provenance."""

    provider: str = Field(description="Name of the provider, e.g. 'Tavily', 'OpenWeather', 'AviationStack'")
    title: str = Field(description="Title or descriptive identifier of the source")
    url: Optional[str] = Field(default=None, description="Direct URL if exposed by provider")
    observed_at: Optional[str] = Field(default=None, description="ISO timestamp when source was observed")
    evidence_kind: EvidenceKind = Field(default=EvidenceKind.WEB_SOURCE, description="Provenance kind")
    freshness: EvidenceFreshness = Field(default=EvidenceFreshness.UNKNOWN, description="Temporal semantics")


class CapabilityEvidence(BaseModel):
    """Normalized evidence produced by an MCP tool invocation."""

    capability: str = Field(description="Logical capability name, e.g. 'HOTEL_WEB_RESEARCH'")
    provider: str = Field(description="Provider name, e.g. 'tavily', 'weather', 'aviationstack'")
    tool_name: str = Field(description="Underlying MCP tool executed")
    evidence_kind: EvidenceKind = Field(default=EvidenceKind.PROVIDER_DATA)
    freshness: EvidenceFreshness = Field(default=EvidenceFreshness.REFERENCE)
    status: CapabilityHealth = Field(default=CapabilityHealth.AVAILABLE)
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    latency_ms: Optional[int] = Field(default=None)
    data: Any = Field(default=None, description="Normalized structured payload")
    summary: str = Field(default="", description="Human/LLM readable concise factual summary")
    sources: list[SourceReference] = Field(default_factory=list, description="Preserved source references")
    error_code: Optional[ErrorCode] = Field(default=None)


class CapabilityTraceEntry(BaseModel):
    """Bounded, safe capability trace entry without secrets or sensitive filesystem paths."""

    sequence: int = Field(default=1, description="1-based invocation order within run")
    specialist: str = Field(default="specialist", description="Specialist agent that requested capability")
    capability: str = Field(default="capability", description="Logical capability name")
    server: str = Field(default="server", description="MCP server name")
    tool_name: str = Field(default="tool", description="MCP tool name")
    status: str = Field(default="AVAILABLE", description="AVAILABLE, DEGRADED, UNAVAILABLE, or CONTRACT_MISMATCH")
    latency_ms: Optional[int] = Field(default=None)
    retry_count: int = Field(default=0)
    started_at: str = Field(default="")
    completed_at: str = Field(default="")
    error_code: Optional[str] = Field(default=None)
    source_count: int = Field(default=0)

    def to_public_dict(self) -> dict[str, Any]:
        """Safe public representation stripped of internal details."""
        return {
            "specialist": self.specialist,
            "capability": self.capability,
            "server": self.server,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "source_count": self.source_count,
            "error_code": self.error_code,
        }


class CapabilityDefinition(BaseModel):
    """Specification of an approved capability in the TripBandhu Capability Registry."""

    capability_name: str = Field(description="Unique capability identifier")
    server: str = Field(description="Target MCP server name")
    tool_name: str = Field(description="Target MCP tool name")
    access_mode: str = Field(default="read_only")
    evidence_kind: EvidenceKind = Field(default=EvidenceKind.PROVIDER_DATA)
    freshness: EvidenceFreshness = Field(default=EvidenceFreshness.REFERENCE)
    description: str = Field(default="")
    required_params: list[str] = Field(default_factory=list)


T = TypeVar("T")


class CapabilityResult(BaseModel, Generic[T]):
    """Typed wrapper returned by provider utility calls."""

    success: bool
    data: Optional[Any] = None
    error_code: Optional[ErrorCode] = None
    safe_message: Optional[str] = None
    retryable: bool = False
    latency_ms: Optional[int] = None
    trace_entry: Optional[CapabilityTraceEntry] = None

    def __iter__(self):
        """Enable tuple unpacking: result, trace = await async_call_provider(...)"""
        return iter((self, self.trace_entry))

    @classmethod
    def ok(
        cls,
        data: Any,
        latency_ms: int | None = None,
        trace_entry: CapabilityTraceEntry | None = None,
    ) -> "CapabilityResult":
        return cls(
            success=True,
            data=data,
            latency_ms=latency_ms,
            trace_entry=trace_entry,
        )

    @classmethod
    def fail(
        cls,
        error_code: ErrorCode,
        safe_message: str,
        latency_ms: int | None = None,
        trace_entry: CapabilityTraceEntry | None = None,
    ) -> "CapabilityResult":
        return cls(
            success=False,
            error_code=error_code,
            safe_message=safe_message,
            retryable=error_code in RETRYABLE_ERROR_CODES,
            latency_ms=latency_ms,
            trace_entry=trace_entry,
        )
