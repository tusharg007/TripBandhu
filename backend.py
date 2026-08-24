import os
import sys
import certifi
from project_config import load_project_env

load_project_env()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from typing import Any, TypedDict, Annotated, Optional
import operator
import uuid
import asyncio

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination_async,
    forecast_mcp_search,
    weather_mcp_search,
    make_groq_llm,
)
from schemas import (
    AgentName,
    AGENT_EXECUTION_ORDER,
    GuardrailCategory,
    GuardrailDecision,
    SupervisorDecision,
    TravelConstraints,
    ReviewDecision,
    CapabilityResult,
    CapabilityEvidence,
    CapabilityTraceEntry,
    CapabilityHealth,
    ErrorCode,
    EvidenceKind,
    SourceReference,
    RunStatus,
)
from capability_registry import (
    CAPABILITY_REGISTRY,
    SPECIALIST_ALLOWED_CAPABILITIES,
    validate_specialist_capability,
    normalize_tavily_results,
    normalize_weather_results,
    normalize_flight_results,
    format_grounded_evidence_context,
)
from agent_config import (
    PROVIDER_TIMEOUT_SECONDS,
    MAX_REVIEW_ITERATIONS,
    GROQ_CONTROL_MODEL,
    GROQ_GENERATION_MODEL,
    LLM_REASONING_EFFORT,
    LLM_REASONING_FORMAT,
    MAX_TOKENS_BY_TASK,
)
from provider_utils import async_call_provider
from llm_utils import invoke_llm, invoke_llm_complete_text, merge_token_usage
from location_utils import normalize_weather_location
from tools.flight_tool import DEFAULT_ORIGIN_IATA, resolve_location_to_iata


def get_database_url() -> str | None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None
    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"
    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY") or ""
DATABASE_URL = get_database_url() or ""

# =========================
# LLM instances — two base clients + task-specific bound variants
# =========================
# Base clients (no shared HTTP client re-creation per call)
_llm_control = make_groq_llm(
    GROQ_CONTROL_MODEL,
    reasoning_effort=LLM_REASONING_EFFORT,
    reasoning_format=LLM_REASONING_FORMAT,
)
_llm_generation = make_groq_llm(
    GROQ_GENERATION_MODEL,
    reasoning_effort=LLM_REASONING_EFFORT,
    reasoning_format=LLM_REASONING_FORMAT,
)

# Task-specific bound variants (preserve base clients, control per-task output size)
_llm_guardrail = _llm_control.bind(
    max_tokens=MAX_TOKENS_BY_TASK["guardrail"]
).with_structured_output(GuardrailDecision, method="json_schema")

_llm_supervisor = _llm_control.bind(
    max_tokens=MAX_TOKENS_BY_TASK["supervisor"]
).with_structured_output(SupervisorDecision, method="json_schema")

_llm_budget   = _llm_control.bind(max_tokens=MAX_TOKENS_BY_TASK["budget"])
_llm_flight   = _llm_generation.bind(max_tokens=MAX_TOKENS_BY_TASK["flight_summary"])
_llm_hotel    = _llm_control.bind(max_tokens=MAX_TOKENS_BY_TASK["hotel_summary"])
_llm_weather  = _llm_control.bind(max_tokens=MAX_TOKENS_BY_TASK["weather_summary"])
_llm_itinerary = _llm_generation.bind(max_tokens=MAX_TOKENS_BY_TASK["itinerary"])
_llm_revision  = _llm_generation.bind(max_tokens=MAX_TOKENS_BY_TASK["revision"])
# final_synthesis uses control model (gpt-oss-20b, 12k TPM) because the full
# prompt (all specialist summaries) exceeds gpt-oss-120b's 8k TPM hard limit.
_llm_final     = _llm_control.bind(max_tokens=MAX_TOKENS_BY_TASK["final_synthesis"])


# =========================
# State
# =========================
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str
    specialist_statuses: dict[str, str]

    # Specialist textual results (for compatibility)
    flight_results: str
    flight_data_available: bool
    hotel_results: str
    hotel_data_available: bool
    weather_results: str
    weather_data_available: bool
    itinerary: str
    budget_results: str

    approval_request: str
    approved: Optional[bool]
    human_feedback: str
    final_response: str

    # Iterative HITL (TF1)
    review_iteration: int
    review_history: list[dict]
    review_limit_reached: bool

    # TF2: Capability Evidence Store & Trace
    evidence_store: dict[str, dict]
    capability_trace: list[dict]

    # LLM accounting
    llm_calls: int
    llm_token_usage: dict[str, int]  # prompt_tokens, completion_tokens, total_tokens

    # Run outcome
    run_status: str  # RunStatus enum value


# =========================
# Constants
# =========================
KNOWN_AGENTS = {a.value for a in AgentName}
AGENT_ORDER = [a.value for a in AGENT_EXECUTION_ORDER]

FLIGHT_UNAVAILABLE_MESSAGE = (
    "Live flight data is temporarily unavailable. The rest of your trip can "
    "still be prepared, but flight details should be verified before booking."
)
HOTEL_UNAVAILABLE_MESSAGE = (
    "Live hotel search is temporarily unavailable. General accommodation and "
    "neighborhood guidance will be provided based on the destination."
)
HOTEL_RATE_LIMIT_MESSAGE = (
    "Live hotel search is temporarily rate-limited by Tavily because too many requests "
    "reached the provider. This is not a hotel-agent reasoning failure. Wait briefly and "
    "start a new trip, or check a trusted accommodation platform directly."
)
WEATHER_UNAVAILABLE_MESSAGE = (
    "Live weather forecast is temporarily unavailable for this destination. "
    "General seasonal guidance will be provided."
)

# Sentinel value for itinerary generation failure — used by routing logic.
_ITINERARY_FAILURE_SENTINEL = "Draft itinerary could not be generated."

_TRAVEL_KEYWORDS = frozenset({
    "trip", "travel", "flight", "hotel", "destination", "tour", "vacation",
    "holiday", "itinerary", "airline", "airport", "visa", "passport",
    "weather", "climate", "accommodation", "booking", "sightseeing",
    "transport", "train", "bus", "cruise", "backpacking", "resort",
    "budget", "ticket", "tourism", "journey", "excursion",
})


def _deterministic_travel_check(query: str) -> GuardrailDecision:
    lower = query.lower()
    matched = any(kw in lower for kw in _TRAVEL_KEYWORDS)
    if matched:
        return GuardrailDecision(
            allowed=True,
            reason="Keyword match",
            category=GuardrailCategory.TRAVEL,
        )
    return GuardrailDecision(
        allowed=False,
        reason="Request does not appear to be travel-related.",
        category=GuardrailCategory.OUT_OF_SCOPE,
    )


def _empty_constraints() -> dict[str, Any]:
    return TravelConstraints().to_dict()


def _incr_llm(state: "TravelState", n: int = 1) -> int:
    return state.get("llm_calls", 0) + n


# =========================
# Supervisor Agent + Guardrail
# =========================
async def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)
    llm_token_usage = dict(state.get("llm_token_usage", {}))

    guardrail_system = (
        "You are the input guardrail for a travel-planning application. "
        "Determine whether the request is travel-related. "
        "Block clearly unrelated, harmful, or illegal requests. "
        "Do not block valid travel requests merely because details are missing."
    )
    guardrail_user = f"User request:\n{query}"

    try:
        gd, gd_tokens = await invoke_llm(
            _llm_guardrail,
            [SystemMessage(content=guardrail_system), HumanMessage(content=guardrail_user)],
            task_name="guardrail",
            timeout_s=PROVIDER_TIMEOUT_SECONDS["llm_structured"],
        )
        llm_calls += 1
        llm_token_usage = merge_token_usage(llm_token_usage, gd_tokens)
        allowed = gd.allowed
        guardrail_reason = gd.reason
    except Exception as exc:
        print(f"[supervisor] Guardrail structured output failed, using keyword fallback: {exc}", flush=True)
        gd_fallback = _deterministic_travel_check(query)
        allowed = gd_fallback.allowed
        guardrail_reason = gd_fallback.reason

    if not allowed:
        reason = guardrail_reason or (
            "TripBandhu can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "specialist_statuses": {agent: "NOT_SELECTED" for agent in KNOWN_AGENTS},
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "evidence_store": {},
            "capability_trace": [],
            "llm_calls": llm_calls,
            "llm_token_usage": llm_token_usage,
            "run_status": RunStatus.BLOCKED.value,
        }

    supervisor_system = (
        "You are the supervisor of a multi-agent travel-planning system. "
        "Choose ONLY the specialist agents needed for the specific request. "
        "If the user asks only for flights, select only flight_agent. "
        "If the user asks only for hotels, select only hotel_agent. "
        "If the user asks only for weather, select only weather_agent. "
        "If the user asks only for budget assessment, select only budget_agent. "
        "If the user asks for a comprehensive trip or itinerary, select all relevant specialists and itinerary_agent. "
        "Return a SupervisorDecision with selected_agents, constraints, and reasoning. "
        "Use concise, natural reasoning."
    )
    supervisor_user = (
        "Available agents:\n"
        "- flight_agent: flights, airports, airlines, routes, airfare, or booking advice\n"
        "- hotel_agent: hotels, accommodation, neighborhoods, or places to stay\n"
        "- weather_agent: weather, climate, season, forecast, or packing advice\n"
        "- budget_agent: cost, affordability, price limits, or budget feasibility\n"
        "- itinerary_agent: creates a detailed day-by-day travel itinerary\n"
        f"\nUser request:\n{query}"
    )

    try:
        decision, sv_tokens = await invoke_llm(
            _llm_supervisor,
            [SystemMessage(content=supervisor_system), HumanMessage(content=supervisor_user)],
            task_name="supervisor",
            timeout_s=PROVIDER_TIMEOUT_SECONDS["llm_structured"],
        )
        llm_calls += 1
        llm_token_usage = merge_token_usage(llm_token_usage, sv_tokens)

        raw_selected = {a.value for a in decision.selected_agents}
        selected_agents = [name for name in AGENT_ORDER if name in raw_selected]

        # Do not force itinerary_agent on specialist-only queries
        if not selected_agents:
            selected_agents = [AgentName.ITINERARY.value]

        constraints = decision.constraints.to_dict()
        reasoning = decision.reasoning.strip()

    except Exception as exc:
        print(f"[supervisor] Structured output failed, using safe defaults: {exc}", flush=True)
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = (
            "Selected travel specialists to coordinate flights, hotels, "
            "weather, budget, and itinerary planning."
        )

    statuses = {
        agent: ("SELECTED" if agent in selected_agents else "NOT_SELECTED")
        for agent in KNOWN_AGENTS
    }

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "specialist_statuses": statuses,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "evidence_store": state.get("evidence_store", {}),
        "capability_trace": state.get("capability_trace", []),
        "llm_calls": llm_calls,
        "llm_token_usage": llm_token_usage,
        "run_status": RunStatus.COMPLETED.value,
    }


async def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


# =========================
# Flight Agent — TF2 Route-Specific Capability & Evidence Normalization
# =========================
FLIGHT_AGENT_PROMPT = (
    "You are a travel flight expert.\n\n"
    "User Query:\n{query}\n\n"
    "Origin: {origin}\n"
    "Destination: {destination}\n\n"
    "Flight & Route Evidence (from AviationStack MCP):\n{flight_evidence}\n\n"
    "Generate a structured flight summary:\n"
    "1. Likely departure airport(s)\n"
    "2. Likely arrival airport(s)\n"
    "3. Operating airlines\n"
    "4. Estimated direct flight duration\n"
    "5. Estimated airfare range (MUST be explicitly marked as: '[MODEL ESTIMATE - Not Live Fare]')\n"
    "6. Booking and seasonal advice\n\n"
    "CRITICAL PRICING TRUTH:\n"
    "- AviationStack does not provide live ticket pricing. NEVER claim live airfare was verified.\n"
    "- Any dollar/rupee amount must be labeled: 'MODEL ESTIMATE - Verify live fares with airlines.'\n"
    "Return concise, professional flight intelligence."
)


async def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT (TF2 Route-Specific & Evidence Normalization)\n")
    query = state["user_query"]
    constraints = state.get("trip_constraints", {})
    dest = constraints.get("destination") or "destination"
    orig = constraints.get("origin") or ""
    statuses = dict(state.get("specialist_statuses", {}))
    llm_calls = state.get("llm_calls", 0)
    llm_token_usage = dict(state.get("llm_token_usage", {}))
    trace = list(state.get("capability_trace", []))
    evidence_store = dict(state.get("evidence_store", {}))
    origin_iata = resolve_location_to_iata(orig) or (DEFAULT_ORIGIN_IATA if not orig else None)
    destination_iata = resolve_location_to_iata(dest)
    route_args = {
        key: value
        for key, value in {
            "dep_iata": origin_iata,
            "arr_iata": destination_iata,
        }.items()
        if value
    }

    validate_specialist_capability("flight_agent", "FLIGHT_ROUTE_SEARCH")

    routes_result = await async_call_provider(
        lambda: aviation_mcp_call("list_routes", route_args),
        provider_name="aviation",
        safe_failure_message=FLIGHT_UNAVAILABLE_MESSAGE,
        specialist="flight_agent",
        capability="FLIGHT_ROUTE_SEARCH",
        server="aviationstack",
        tool_name="list_routes",
    )
    if getattr(routes_result, "trace_entry", None):
        routes_result.trace_entry.sequence = len(trace) + 1
        trace.append(routes_result.trace_entry.model_dump())

    airports_result = await async_call_provider(
        lambda: aviation_mcp_call("list_airports"),
        provider_name="aviation",
        safe_failure_message=FLIGHT_UNAVAILABLE_MESSAGE,
        specialist="flight_agent",
        capability="AIRPORT_LOOKUP",
        server="aviationstack",
        tool_name="list_airports",
    )
    if getattr(airports_result, "trace_entry", None):
        airports_result.trace_entry.sequence = len(trace) + 1
        trace.append(airports_result.trace_entry.model_dump())

    airlines_result = await async_call_provider(
        lambda: aviation_mcp_call("list_airlines"),
        provider_name="aviation",
        safe_failure_message=FLIGHT_UNAVAILABLE_MESSAGE,
        specialist="flight_agent",
        capability="AIRLINE_LOOKUP",
        server="aviationstack",
        tool_name="list_airlines",
    )
    if getattr(airlines_result, "trace_entry", None):
        airlines_result.trace_entry.sequence = len(trace) + 1
        trace.append(airlines_result.trace_entry.model_dump())

    if not (routes_result.success or airports_result.success or airlines_result.success):
        statuses["flight_agent"] = "DEGRADED"
        ev_unavailable = CapabilityEvidence(
            capability="FLIGHT_ROUTE_SEARCH",
            provider="aviationstack",
            tool_name="list_routes",
            evidence_kind=EvidenceKind.UNAVAILABLE,
            status=CapabilityHealth.UNAVAILABLE,
            summary=FLIGHT_UNAVAILABLE_MESSAGE,
            error_code=routes_result.error_code,
        )
        evidence_store["FLIGHT_ROUTE_SEARCH"] = ev_unavailable.model_dump()

        return {
            "flight_results": FLIGHT_UNAVAILABLE_MESSAGE,
            "flight_data_available": False,
            "specialist_statuses": statuses,
            "capability_trace": trace,
            "evidence_store": evidence_store,
            "messages": [AIMessage(content="Flight data unavailable.")],
            "llm_calls": llm_calls,
            "llm_token_usage": llm_token_usage,
        }

    flight_ev = normalize_flight_results(
        routes_data=routes_result.data if routes_result.success else None,
        airports_data=airports_result.data if airports_result.success else None,
        airlines_data=airlines_result.data if airlines_result.success else None,
        destination=dest,
        origin=orig,
        latency_ms=routes_result.latency_ms,
    )
    evidence_store["FLIGHT_ROUTE_SEARCH"] = flight_ev.model_dump()

    try:
        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            origin=orig or "Specified origin",
            destination=dest,
            flight_evidence=flight_ev.summary,
        )
        flight_text, tokens, generation_calls = await invoke_llm_complete_text(
            _llm_flight,
            [
                SystemMessage(content="You are an expert travel flight planner."),
                HumanMessage(content=prompt),
            ],
            task_name="flight_summary",
        )
        llm_token_usage = merge_token_usage(llm_token_usage, tokens)
        statuses["flight_agent"] = "COMPLETED"
        return {
            "flight_results": flight_text,
            "flight_data_available": True,
            "specialist_statuses": statuses,
            "capability_trace": trace,
            "evidence_store": evidence_store,
            "messages": [AIMessage(content="Flight recommendations generated.")],
            "llm_calls": llm_calls + generation_calls,
            "llm_token_usage": llm_token_usage,
        }
    except Exception as exc:
        print(f"[flight_agent] LLM call failed: {type(exc).__name__}: {exc}", flush=True)
        statuses["flight_agent"] = "DEGRADED"
        return {
            "flight_results": FLIGHT_UNAVAILABLE_MESSAGE,
            "flight_data_available": False,
            "specialist_statuses": statuses,
            "capability_trace": trace,
            "evidence_store": evidence_store,
            "messages": [AIMessage(content="Flight data unavailable.")],
            "llm_calls": llm_calls,
            "llm_token_usage": llm_token_usage,
        }


# =========================
# Hotel Agent — Tavily Web Research & Source Preservation
# =========================
_HOTEL_PRESENTATION_SYSTEM = (
    "You turn verified accommodation web evidence into an end-user hotel brief. "
    "Use only the supplied evidence. Exclude unrelated pages, raw payloads, template text, "
    "and developer diagnostics. Recommend up to six genuinely supported properties or areas, "
    "explain who each suits and its location advantage, and preserve the supplied source links. "
    "Do not invent prices, star ratings, availability, amenities, or review scores. Clearly say "
    "that availability and rates must be checked before booking. Return polished Markdown."
)


async def hotel_agent(state: TravelState):
    constraints = state.get("trip_constraints") or {}
    destination = constraints.get("destination") or state["user_query"]
    budget = constraints.get("budget") or ""
    travel_style = constraints.get("travel_style") or ""
    query = (
        f"Hotels accommodation neighborhoods and trusted places to stay in {destination}. "
        f"Traveller needs: {travel_style or 'general'}, budget: {budget or 'not specified'}. "
        "Include property names, neighborhood, and practical location details."
    )
    statuses = dict(state.get("specialist_statuses", {}))
    llm_calls = state.get("llm_calls", 0)
    llm_token_usage = dict(state.get("llm_token_usage", {}))
    trace = list(state.get("capability_trace", []))
    evidence_store = dict(state.get("evidence_store", {}))

    validate_specialist_capability("hotel_agent", "HOTEL_WEB_RESEARCH")

    result = await async_call_provider(
        lambda: tavily_mcp_search(query),
        provider_name="tavily",
        safe_failure_message=HOTEL_UNAVAILABLE_MESSAGE,
        specialist="hotel_agent",
        capability="HOTEL_WEB_RESEARCH",
        server="tavily",
        tool_name="tavily_search",
    )
    if getattr(result, "trace_entry", None):
        result.trace_entry.sequence = len(trace) + 1
        trace.append(result.trace_entry.model_dump())

    if result.success:
        hotel_ev = normalize_tavily_results(result.data, query, latency_ms=result.latency_ms)
        evidence_store["HOTEL_WEB_RESEARCH"] = hotel_ev.model_dump()

        if not hotel_ev.data:
            statuses["hotel_agent"] = "DEGRADED"
            return {
                "hotel_results": hotel_ev.summary,
                "hotel_data_available": False,
                "specialist_statuses": statuses,
                "capability_trace": trace,
                "evidence_store": evidence_store,
                "messages": [AIMessage(content="No relevant hotel sources were found.")],
                "llm_calls": llm_calls,
                "llm_token_usage": llm_token_usage,
            }

        try:
            response, tokens = await invoke_llm(
                _llm_hotel,
                [
                    SystemMessage(content=_HOTEL_PRESENTATION_SYSTEM),
                    HumanMessage(content=(
                        f"Destination: {destination}\n"
                        f"Original request: {state['user_query']}\n\n"
                        f"Filtered accommodation evidence:\n{hotel_ev.summary}"
                    )),
                ],
                task_name="hotel_summary",
            )
            hotel_text = response.content
            llm_calls += 1
            llm_token_usage = merge_token_usage(llm_token_usage, tokens)
            statuses["hotel_agent"] = "COMPLETED"
        except Exception as exc:
            print(f"[hotel_agent] Presentation LLM failed: {type(exc).__name__}: {exc}", flush=True)
            hotel_text = hotel_ev.summary
            statuses["hotel_agent"] = "DEGRADED"

        return {
            "hotel_results": hotel_text,
            "hotel_data_available": True,
            "specialist_statuses": statuses,
            "capability_trace": trace,
            "evidence_store": evidence_store,
            "messages": [AIMessage(content="Hotel information processed.")],
            "llm_calls": llm_calls,
            "llm_token_usage": llm_token_usage,
        }
    else:
        statuses["hotel_agent"] = "DEGRADED"
        hotel_failure_message = (
            HOTEL_RATE_LIMIT_MESSAGE
            if result.error_code == ErrorCode.RATE_LIMITED
            else (result.safe_message or HOTEL_UNAVAILABLE_MESSAGE)
        )
        ev_unavailable = CapabilityEvidence(
            capability="HOTEL_WEB_RESEARCH",
            provider="tavily",
            tool_name="tavily_search",
            evidence_kind=EvidenceKind.UNAVAILABLE,
            status=CapabilityHealth.UNAVAILABLE,
            summary=hotel_failure_message,
            error_code=result.error_code,
        )
        evidence_store["HOTEL_WEB_RESEARCH"] = ev_unavailable.model_dump()

        return {
            "hotel_results": hotel_failure_message,
            "hotel_data_available": False,
            "specialist_statuses": statuses,
            "capability_trace": trace,
            "evidence_store": evidence_store,
            "messages": [AIMessage(content="Hotel data unavailable.")],
            "llm_calls": llm_calls,
            "llm_token_usage": llm_token_usage,
        }


# =========================
# Weather Agent — OpenWeather MCP & Normalization
# Phase 3: Destination priority — supervisor constraints > LLM fallback
# =========================
_WEATHER_PRESENTATION_SYSTEM = (
    "You turn normalized weather evidence into concise, friendly travel guidance. "
    "Use only the facts supplied. Never show raw JSON, Python dictionaries, tool errors, URLs, "
    "API details, stack traces, or developer diagnostics. Summarize current conditions and the "
    "forecast, then add practical packing or timing advice supported by those conditions. "
    "Do not imply that a short forecast covers dates outside its stated range. "
    "Your entire response must be weather-only: never include flights, hotels, an itinerary, "
    "budgets, costs, restaurants, or sightseeing. Return concise Markdown."
)


_WEATHER_OUT_OF_SCOPE_MARKERS = (
    "budget", "cost breakdown", "estimated cost", "itinerary", "day 1", "day1",
    "suggested hotel", "accommodation", "flight", "sightseeing", "food & dining",
)


def _is_weather_only_presentation(text: str) -> bool:
    """Reject cross-specialist content before it can reach the Weather tab."""
    normalized = " ".join(str(text or "").casefold().split())
    if not normalized:
        return False
    return not any(marker in normalized for marker in _WEATHER_OUT_OF_SCOPE_MARKERS)


async def weather_agent(state: TravelState):
    statuses = dict(state.get("specialist_statuses", {}))
    llm_calls = state.get("llm_calls", 0)
    llm_token_usage = dict(state.get("llm_token_usage", {}))
    trace = list(state.get("capability_trace", []))
    evidence_store = dict(state.get("evidence_store", {}))

    validate_specialist_capability("weather_agent", "WEATHER_CURRENT")

    # Priority 1: Use supervisor-extracted destination (avoids redundant LLM call on normal trips)
    city = (state.get("trip_constraints") or {}).get("destination") or ""

    # Priority 2: LLM fallback — only when destination is genuinely absent from constraints
    if not city:
        try:
            async with asyncio.timeout(PROVIDER_TIMEOUT_SECONDS["llm_structured"]):
                city = await extract_destination_async(state["user_query"])
            llm_calls += 1
        except Exception as exc:
            print(f"[weather_agent] Destination extraction failed: {exc}", flush=True)
            city = "your destination"

    city = normalize_weather_location(city) or "your destination"

    weather_result = await async_call_provider(
        lambda: weather_mcp_search(city),
        provider_name="weather",
        safe_failure_message=WEATHER_UNAVAILABLE_MESSAGE,
        specialist="weather_agent",
        capability="WEATHER_CURRENT",
        server="weather",
        tool_name="get_current_weather",
    )
    if getattr(weather_result, "trace_entry", None):
        weather_result.trace_entry.sequence = len(trace) + 1
        trace.append(weather_result.trace_entry.model_dump())

    forecast_result = await async_call_provider(
        lambda: forecast_mcp_search(city),
        provider_name="weather",
        safe_failure_message=WEATHER_UNAVAILABLE_MESSAGE,
        specialist="weather_agent",
        capability="WEATHER_FORECAST",
        server="weather",
        tool_name="get_forecast",
    )
    if getattr(forecast_result, "trace_entry", None):
        forecast_result.trace_entry.sequence = len(trace) + 1
        trace.append(forecast_result.trace_entry.model_dump())

    if weather_result.success or forecast_result.success:
        weather_ev = normalize_weather_results(
            current_data=weather_result.data if weather_result.success else None,
            forecast_data=forecast_result.data if forecast_result.success else None,
            city=city,
            latency_ms=(weather_result.latency_ms or 0) + (forecast_result.latency_ms or 0),
        )
        evidence_store["WEATHER_FORECAST"] = weather_ev.model_dump()
        statuses["weather_agent"] = (
            "COMPLETED" if weather_result.success and forecast_result.success else "DEGRADED"
        )

        try:
            response, tokens = await invoke_llm(
                _llm_weather,
                [
                    SystemMessage(content=_WEATHER_PRESENTATION_SYSTEM),
                    HumanMessage(content=(
                        f"Destination: {city}\n\n"
                        f"Weather evidence:\n{weather_ev.summary}\n\n"
                        "Produce only: current conditions, near-term forecast, and packing/timing advice."
                    )),
                ],
                task_name="weather_summary",
            )
            llm_calls += 1
            llm_token_usage = merge_token_usage(llm_token_usage, tokens)
            if _is_weather_only_presentation(response.content):
                weather_text = response.content
            else:
                print(
                    "[weather_agent] Rejected cross-specialist presentation; using normalized weather fallback.",
                    flush=True,
                )
                weather_text = weather_ev.summary
        except Exception as exc:
            print(f"[weather_agent] Presentation LLM failed: {type(exc).__name__}: {exc}", flush=True)
            weather_text = weather_ev.summary
            statuses["weather_agent"] = "DEGRADED"

        return {
            "weather_results": weather_text,
            "weather_data_available": True,
            "specialist_statuses": statuses,
            "capability_trace": trace,
            "evidence_store": evidence_store,
            "messages": [AIMessage(content="Weather information processed.")],
            "llm_calls": llm_calls,
            "llm_token_usage": llm_token_usage,
        }
    else:
        dest = (state.get("trip_constraints") or {}).get("destination") or city or "your destination"
        weather_text = (
            f"Live weather information for {dest} is temporarily unavailable. "
            "General seasonal guidance will be provided; check current forecasts closer to departure."
        )
        statuses["weather_agent"] = "DEGRADED"
        ev_unavailable = CapabilityEvidence(
            capability="WEATHER_FORECAST",
            provider="weather",
            tool_name="get_current_weather,get_forecast",
            evidence_kind=EvidenceKind.UNAVAILABLE,
            status=CapabilityHealth.UNAVAILABLE,
            summary=weather_text,
            error_code=weather_result.error_code or forecast_result.error_code,
        )
        evidence_store["WEATHER_FORECAST"] = ev_unavailable.model_dump()

        return {
            "weather_results": weather_text,
            "weather_data_available": False,
            "specialist_statuses": statuses,
            "capability_trace": trace,
            "evidence_store": evidence_store,
            "messages": [AIMessage(content="Weather data unavailable.")],
            "llm_calls": llm_calls,
            "llm_token_usage": llm_token_usage,
        }


# =========================
# Budget Agent
# Phase 4+5: Control model, task-specific token budget, stable instructions first
# =========================
_BUDGET_SYSTEM = (
    "You are a practical travel budget analyst.\n\n"
    "CRITICAL GROUNDEDNESS RULES:\n"
    "- DO NOT invent exact flight fares or prices if flight data is unavailable or schedule-only.\n"
    "- All flight cost estimations MUST be explicitly marked: '[MODEL ESTIMATE - Not Live Fare]'.\n"
    "- AviationStack does not supply live ticket prices. If flight data is unavailable or schedule-only, "
    "state clearly that airfare is estimated and must be verified.\n"
    "- Web hotel rates are search-based approximations ([WEB_SOURCE]); do not present them as locked booking rates.\n"
    "- Clearly differentiate verified provider data from model estimation categories.\n"
    "- Do not invent the traveller's salary or monthly income. If no explicit budget amount was provided, "
    "state that affordability against the user's limit cannot be determined and provide a planning range.\n"
    "- Complete every requested section and table. Never stop mid-row, mid-bullet, or mid-sentence.\n"
    "- Use INR as the primary currency for an India-origin trip; label any exchange rate as an estimate."
)


async def budget_agent(state: TravelState):
    statuses = dict(state.get("specialist_statuses", {}))
    evidence_context = format_grounded_evidence_context(state.get("evidence_store", {}))
    llm_token_usage = dict(state.get("llm_token_usage", {}))

    prompt = (
        f"Analyze whether this trip is realistic for the user's budget.\n\n"
        f"User Query:\n{state['user_query']}\n\n"
        f"Trip Constraints:\n{state.get('trip_constraints', {})}\n\n"
        f"Grounded Provider & Web Evidence:\n{evidence_context}\n\n"
        "Return:\n1. Estimated cost breakdown by category\n"
        "2. Budget feasibility assessment\n"
        "3. Cost-saving recommendations\n"
        "4. Risk areas\n"
        "5. Budget assumptions and verification checklist\n\n"
        "Finish all five sections. Keep tables compact enough to complete the response."
    )

    try:
        budget_text, tokens, generation_calls = await invoke_llm_complete_text(
            _llm_budget,
            [
                SystemMessage(content=_BUDGET_SYSTEM),
                HumanMessage(content=prompt),
            ],
            task_name="budget",
        )
        llm_token_usage = merge_token_usage(llm_token_usage, tokens)
        statuses["budget_agent"] = "COMPLETED"
        return {
            "budget_results": budget_text,
            "specialist_statuses": statuses,
            "messages": [AIMessage(content="Budget assessment generated.")],
            "llm_calls": state.get("llm_calls", 0) + generation_calls,
            "llm_token_usage": llm_token_usage,
        }
    except Exception as exc:
        print(f"[budget_agent] LLM call failed: {exc}", flush=True)
        statuses["budget_agent"] = "DEGRADED"
        return {
            "budget_results": "Budget estimation could not be generated.",
            "specialist_statuses": statuses,
            "messages": [AIMessage(content="Budget estimation failed.")],
            "llm_calls": state.get("llm_calls", 0),
            "llm_token_usage": llm_token_usage,
        }


# =========================
# Itinerary Agent
# Phase 4+5: Generation model; Phase 6: DEGRADED does NOT set approval_request
# =========================
_ITINERARY_SYSTEM = (
    "You are an expert travel planner.\n\n"
    "CRITICAL GROUNDEDNESS RULES:\n"
    "- DO NOT invent or fabricate flight prices or live flight schedules if flight data is unavailable.\n"
    "- If flight data is marked [UNAVAILABLE], state clearly: "
    "'Live flight data was unavailable; check schedules with airlines.' "
    "Do NOT fabricate fake flight numbers or fake price ranges.\n"
    "- For hotel recommendations from web sources ([WEB_SOURCE]), preserve neighborhood and accommodation names.\n"
    "- Never invent factual data that contradicts the Grounded Evidence section.\n"
    "- Complete every day requested by the user. Never end mid-day or mid-sentence. If space is "
    "tight, make each day more concise instead of omitting later days.\n"
    "- End with a short 'Before you book' checklist so completion is unambiguous."
)


async def itinerary_agent(state: TravelState):
    statuses = dict(state.get("specialist_statuses", {}))
    evidence_context = format_grounded_evidence_context(state.get("evidence_store", {}))
    llm_token_usage = dict(state.get("llm_token_usage", {}))

    prompt = (
        f"Create a complete travel itinerary for human review.\n\n"
        f"User Query:\n{state['user_query']}\n\n"
        f"Trip Constraints:\n{state.get('trip_constraints', {})}\n\n"
        f"Grounded Evidence & Provenance:\n{evidence_context}\n\n"
        f"Budget Analysis:\n{state.get('budget_results', '')}\n\n"
        "Make the itinerary practical, budget-aware, and ready for human review. "
        "Include all requested days and finish the complete plan."
    )

    try:
        itinerary_content, tokens, generation_calls = await invoke_llm_complete_text(
            _llm_itinerary,
            [
                SystemMessage(content=_ITINERARY_SYSTEM),
                HumanMessage(content=prompt),
            ],
            task_name="itinerary",
        )
        llm_token_usage = merge_token_usage(llm_token_usage, tokens)
        statuses["itinerary_agent"] = "COMPLETED"
        llm_calls = state.get("llm_calls", 0) + generation_calls
        approval_request = (
            "Please review the generated draft itinerary. Approve it to create the "
            "final polished plan, or provide feedback for revision."
        )
    except Exception as exc:
        print(f"[itinerary_agent] LLM call failed: {exc}", flush=True)
        statuses["itinerary_agent"] = "DEGRADED"
        itinerary_content = _ITINERARY_FAILURE_SENTINEL
        llm_calls = state.get("llm_calls", 0)
        # PHASE 6: Do NOT set approval_request when DEGRADED — route_after_itinerary
        # will bypass human_review and send the run directly to itinerary_degraded_agent.
        approval_request = ""

    return {
        "itinerary": itinerary_content,
        "approval_request": approval_request,
        "specialist_statuses": statuses,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": llm_calls,
        "llm_token_usage": llm_token_usage,
    }


# =========================
# Itinerary Revision Agent
# Phase 4+5: Generation model; Phase 6: DEGRADED also routes away from human_review
# =========================
_REVISION_SYSTEM = (
    "You are an expert travel planner revising an itinerary based on feedback.\n\n"
    "CRITICAL GROUNDEDNESS RULES:\n"
    "- Never fabricate specific fares or live booking details when specialist data is marked unavailable.\n"
    "- Apply the feedback accurately while maintaining grounded facts."
)


async def itinerary_revision_agent(state: TravelState):
    statuses = dict(state.get("specialist_statuses", {}))
    feedback = state.get("human_feedback", "").strip() or "Please improve the draft itinerary."
    iteration = state.get("review_iteration", 0)
    evidence_context = format_grounded_evidence_context(state.get("evidence_store", {}))
    llm_token_usage = dict(state.get("llm_token_usage", {}))

    prompt = (
        f"You previously drafted a travel itinerary. The human reviewer has requested a revision.\n\n"
        f"Original User Query:\n{state['user_query']}\n\n"
        f"Human Reviewer Feedback (apply carefully):\n{feedback}\n\n"
        f"Previous Draft Itinerary:\n{state.get('itinerary', '')}\n\n"
        f"Grounded Evidence & Provenance:\n{evidence_context}\n\n"
        "Produce a revised travel itinerary that directly addresses the feedback."
    )

    try:
        revised_itinerary, tokens, generation_calls = await invoke_llm_complete_text(
            _llm_revision,
            [
                SystemMessage(content=_REVISION_SYSTEM),
                HumanMessage(content=prompt),
            ],
            task_name="revision",
        )
        llm_token_usage = merge_token_usage(llm_token_usage, tokens)
        statuses["itinerary_agent"] = "COMPLETED"
        llm_calls = state.get("llm_calls", 0) + generation_calls
        approval_request = (
            f"Revised itinerary (revision {iteration + 1}). "
            "Please review and approve, or provide further feedback."
        )
    except Exception as exc:
        print(f"[itinerary_revision_agent] LLM call failed: {exc}", flush=True)
        statuses["itinerary_agent"] = "DEGRADED"
        revised_itinerary = state.get("itinerary", _ITINERARY_FAILURE_SENTINEL)
        llm_calls = state.get("llm_calls", 0)
        approval_request = ""  # PHASE 6: bypass human_review on failure

    return {
        "itinerary": revised_itinerary,
        "approval_request": approval_request,
        "specialist_statuses": statuses,
        "review_iteration": iteration + 1,
        "messages": [AIMessage(content=f"Itinerary revised (iteration {iteration + 1}).")],
        "llm_calls": llm_calls,
        "llm_token_usage": llm_token_usage,
    }


# =========================
# Itinerary Degraded Agent (Phase 6 — new node)
# Handles failed itinerary generation: truthful message, no approval gate.
# =========================
async def itinerary_degraded_agent(state: TravelState):
    """
    Called when itinerary_agent (or itinerary_revision_agent) fails (DEGRADED).
    Produces a truthful degraded response without entering human_review.
    Never invents a fallback itinerary.
    """
    statuses = dict(state.get("specialist_statuses", {}))
    statuses["itinerary_agent"] = "DEGRADED"

    degraded_msg = (
        "The draft itinerary could not be generated in this run — "
        "the AI generation step did not complete successfully. "
        "Please retry your request. If the problem persists, it may be due to a "
        "temporary service interruption or provider resource limit.\n\n"
        "All other specialist results (flights, hotels, weather, budget) remain available above."
    )
    return {
        "final_response": degraded_msg,
        "approval_request": "",
        "run_status": RunStatus.DEGRADED.value,
        "specialist_statuses": statuses,
        "messages": [AIMessage(content=degraded_msg)],
    }


# =========================
# Human Review Node
# =========================
async def human_review_node(state: TravelState):
    iteration = state.get("review_iteration", 0)
    review_history = list(state.get("review_history") or [])

    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "specialist_statuses": state.get("specialist_statuses", {}),
            "review_iteration": iteration,
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    review_entry = ReviewDecision(
        iteration=iteration,
        approved=approved,
        feedback=human_feedback,
    )
    review_history.append(review_entry.model_dump())

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "review_history": review_history,
        "messages": [AIMessage(content=f"Human review {iteration}: {'approved' if approved else 'revision requested'}.")],
    }


# =========================
# Final Response Agent
# Phase 5: Compact context — no raw evidence_context; specialist summaries + provider status block
# Phase 4: Generation model with final_synthesis token budget
# =========================
def _build_provider_status_block(state: TravelState) -> str:
    """Compact, grounding-preserving provider status block for final_agent prompt."""
    statuses = state.get("specialist_statuses", {})
    lines = []

    if statuses.get("flight_agent") not in (None, "NOT_SELECTED"):
        avail = "AVAILABLE" if state.get("flight_data_available") else "DEGRADED — data not retrieved"
        lines.append(f"- Flights: {avail} [PROVIDER_DATA · REFERENCE — verify schedules/fares before booking]")

    if statuses.get("hotel_agent") not in (None, "NOT_SELECTED"):
        avail = "AVAILABLE" if state.get("hotel_data_available") else "DEGRADED — data not retrieved"
        lines.append(f"- Hotels: {avail} [WEB_SOURCE — web search results, not locked booking rates]")

    if statuses.get("weather_agent") not in (None, "NOT_SELECTED"):
        avail = "AVAILABLE" if state.get("weather_data_available") else "DEGRADED — data not retrieved"
        lines.append(f"- Weather: {avail} [PROVIDER_DATA · LIVE/FORECAST]")

    if statuses.get("budget_agent") not in (None, "NOT_SELECTED"):
        budget_status = statuses.get("budget_agent", "COMPLETED")
        lines.append(f"- Budget: {budget_status} [MODEL_ESTIMATE — verify all price ranges]")

    return "\n".join(lines) if lines else "No specialist provider data retrieved."


_FINAL_SYSTEM = (
    "You are a professional AI travel research assistant.\n\n"
    "CRITICAL GROUNDEDNESS & TRUTHFULNESS RULES:\n"
    "- When presenting flight options, describe them as 'Flight Route Reference' "
    "(Provider data · Reference — verify current schedule before booking). "
    "Never describe route reference data as confirmed flights or live bookings.\n"
    "- When presenting the approved itinerary, label the section 'Approved Day-by-Day Itinerary' "
    "or 'Day-by-Day Itinerary' — NEVER 'Confirmed Schedule' or 'Confirmed Itinerary' "
    "since reservations are not made.\n"
    "- If flight data is unavailable or degraded, state: 'Live flight information was unavailable for this run. Verify schedules and fares with an airline or booking provider before booking.'\n"
    "- All estimated price ranges must be explicitly marked '[MODEL ESTIMATE - Not Live Fare]'.\n"
    "- Format the final answer beautifully with markdown sections suited to the query.\n"
    "Be clear, realistic, and practical."
)


async def final_agent(state: TravelState):
    is_specialist_only = "itinerary_agent" not in state.get("selected_agents", [])
    llm_token_usage = dict(state.get("llm_token_usage", {}))

    if is_specialist_only:
        review_instruction = (
            "The user asked a focused travel query. Provide a clear, comprehensive, and helpful summary "
            "focusing specifically on the requested information without generating an unrequested full itinerary."
        )
    elif state.get("approved", False):
        review_instruction = "The user approved the draft itinerary. Preserve its decisions while polishing it."
    else:
        review_instruction = (
            "Apply this human feedback carefully:\n"
            f"{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}"
        )

    provider_status = _build_provider_status_block(state)

    # Phase 5: compact context — specialist summaries already distill the evidence_context.
    # Sending raw evidence_context on top of them repeats the same provider facts.
    # Provenance labels (PROVIDER_DATA, WEB_SOURCE, MODEL_ESTIMATE) are preserved via
    # provider_status block and grounding rules in the specialist summaries themselves.
    # These are already normalized specialist summaries. Preserve them in full:
    # character slicing used to cut flight records and approved itineraries midway.
    _hotel_summary = state.get("hotel_results", "") or "(not retrieved)"
    _weather_summary = state.get("weather_results", "") or "(not retrieved)"
    _flight_summary = state.get("flight_results", "") or "(not retrieved)"
    _budget_summary = state.get("budget_results", "") or "(not retrieved)"
    _itinerary_draft = state.get("itinerary", "") or "(none)"

    final_prompt = (
        f"## Instructions\n{review_instruction}\n\n"
        f"## User Request\n{state['user_query']}\n\n"
        f"## Trip Constraints\n{state.get('trip_constraints', {})}\n\n"
        f"## Provider Status\n{provider_status}\n\n"
        f"## Flight Summary\n{_flight_summary}\n\n"
        f"## Hotel Summary\n{_hotel_summary}\n\n"
        f"## Weather Summary\n{_weather_summary}\n\n"
        f"## Budget Analysis\n{_budget_summary}\n\n"
        f"## Draft Itinerary\n{_itinerary_draft}"
    )

    try:
        response, tokens = await invoke_llm(
            _llm_final,
            [
                SystemMessage(content=_FINAL_SYSTEM),
                HumanMessage(content=final_prompt),
            ],
            task_name="final_synthesis",
        )
        llm_token_usage = merge_token_usage(llm_token_usage, tokens)
        return {
            "final_response": response.content,
            "messages": [response],
            "llm_calls": _incr_llm(state),
            "llm_token_usage": llm_token_usage,
            "run_status": RunStatus.COMPLETED.value,
        }
    except Exception as exc:
        import traceback as _tb
        print(f"[final_agent] LLM call FAILED ({type(exc).__name__}): {exc}", flush=True)
        print(_tb.format_exc(), flush=True)
        return {
            "final_response": "The final travel plan could not be generated. Please try again.",
            "messages": [AIMessage(content="Final plan generation failed.")],
            "llm_calls": state.get("llm_calls", 0),
            "llm_token_usage": llm_token_usage,
            "run_status": RunStatus.DEGRADED.value,
        }


# =========================
# Review Limit Reached Node
# =========================
async def review_limit_reached_agent(state: TravelState):
    iteration = state.get("review_iteration", 0)
    message = (
        f"Review limit reached ({iteration} revision iterations). "
        "No final approved travel plan was produced. "
        "Please start a new planning run with updated requirements or revise your request."
    )
    return {
        "final_response": message,
        "approved": False,
        "review_limit_reached": True,
        "approval_request": "",
        "run_status": RunStatus.DEGRADED.value,
        "messages": [AIMessage(content=message)],
    }


# =========================
# Dynamic Routing Logic
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
    "itinerary_degraded": "itinerary_degraded",
    "final_agent": "final_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"
    selected = _selected_agents(state)
    return selected[0] if selected else "final_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)
        for next_agent in AGENT_ORDER[current_index + 1:]:
            if next_agent in selected:
                return next_agent
        if "itinerary_agent" in selected:
            return "itinerary_agent"
        return "final_agent"
    return route


def route_after_itinerary(state: TravelState) -> str:
    """
    Phase 6: Route itinerary_agent (and itinerary_revision) output.
    DEGRADED or missing itinerary → itinerary_degraded (no human_review gate).
    Valid itinerary → human_review as normal.
    """
    statuses = state.get("specialist_statuses", {})
    itinerary = (state.get("itinerary") or "").strip()
    if (
        statuses.get("itinerary_agent") == "DEGRADED"
        or not itinerary
        or itinerary == _ITINERARY_FAILURE_SENTINEL
    ):
        return "itinerary_degraded"
    return "human_review"


def route_after_review(state: TravelState) -> str:
    iteration = state.get("review_iteration", 0)
    approved = state.get("approved", False)

    if approved:
        return "final_agent"

    if (iteration + 1) >= MAX_REVIEW_ITERATIONS:
        print(
            f"[HITL] Review limit reached ({iteration + 1} rejections >= {MAX_REVIEW_ITERATIONS}). "
            "Routing to review_limit_reached node. NEVER auto-approving.",
            flush=True,
        )
        return "review_limit_reached"

    return "itinerary_revision"


# =========================
# Graph Construction & Service Factory
# =========================
def build_travel_graph(checkpointer=None):
    g = StateGraph(TravelState)

    g.add_node("supervisor", supervisor_agent)
    g.add_node("guardrail_blocked", guardrail_blocked_agent)
    g.add_node("flight_agent", flight_agent)
    g.add_node("hotel_agent", hotel_agent)
    g.add_node("weather_agent", weather_agent)
    g.add_node("budget_agent", budget_agent)
    g.add_node("itinerary_agent", itinerary_agent)
    g.add_node("itinerary_revision", itinerary_revision_agent)
    g.add_node("itinerary_degraded", itinerary_degraded_agent)  # Phase 6
    g.add_node("human_review", human_review_node)
    g.add_node("final_agent", final_agent)
    g.add_node("review_limit_reached", review_limit_reached_agent)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)
    g.add_conditional_edges("flight_agent", route_after_agent("flight_agent"), ROUTE_MAP)
    g.add_conditional_edges("hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP)
    g.add_conditional_edges("weather_agent", route_after_agent("weather_agent"), ROUTE_MAP)
    g.add_conditional_edges("budget_agent", route_after_agent("budget_agent"), ROUTE_MAP)

    # Phase 6: itinerary_agent routes conditionally — DEGRADED bypasses human_review
    g.add_conditional_edges(
        "itinerary_agent",
        route_after_itinerary,
        {"human_review": "human_review", "itinerary_degraded": "itinerary_degraded"},
    )
    # Phase 6: revision failures also bypass human_review
    g.add_conditional_edges(
        "itinerary_revision",
        route_after_itinerary,
        {"human_review": "human_review", "itinerary_degraded": "itinerary_degraded"},
    )

    g.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "final_agent": "final_agent",
            "itinerary_revision": "itinerary_revision",
            "review_limit_reached": "review_limit_reached",
        },
    )
    g.add_edge("final_agent", END)
    g.add_edge("guardrail_blocked", END)
    g.add_edge("review_limit_reached", END)
    g.add_edge("itinerary_degraded", END)  # Phase 6

    return g.compile(checkpointer=checkpointer)


class TravelAgentService:
    def __init__(self, compiled_graph):
        self.graph = compiled_graph

    async def run(self, user_input: str, thread_id: str | None = None) -> dict[str, Any]:
        if not thread_id:
            thread_id = f"user_{uuid.uuid4().hex}"
        config = {"configurable": {"thread_id": thread_id}}
        result = await self.graph.ainvoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "user_query": user_input,
                "guardrail_allowed": True,
                "guardrail_reason": "",
                "selected_agents": [],
                "specialist_statuses": {},
                "trip_constraints": _empty_constraints(),
                "supervisor_reasoning": "",
                "flight_results": "",
                "flight_data_available": True,
                "hotel_results": "",
                "hotel_data_available": True,
                "weather_results": "",
                "weather_data_available": True,
                "budget_results": "",
                "itinerary": "",
                "approval_request": "",
                "approved": None,
                "human_feedback": "",
                "final_response": "",
                "review_iteration": 0,
                "review_history": [],
                "review_limit_reached": False,
                "evidence_store": {},
                "capability_trace": [],
                "llm_calls": 0,
                "llm_token_usage": {},
                "run_status": RunStatus.COMPLETED.value,
            },
            config=config,
        )
        return _serialize_result(result, thread_id)

    async def resume(self, thread_id: str, approved: bool, feedback: str = "") -> dict[str, Any]:
        if not thread_id:
            raise ValueError("thread_id is required to resume a travel plan.")
        config = {"configurable": {"thread_id": thread_id}}
        result = await self.graph.ainvoke(
            Command(
                resume={
                    "approved": approved,
                    "feedback": feedback.strip(),
                }
            ),
            config=config,
        )
        return _serialize_result(result, thread_id)


def create_travel_service(checkpointer=None) -> TravelAgentService:
    compiled = build_travel_graph(checkpointer=checkpointer)
    return TravelAgentService(compiled)


_default_service: Optional[TravelAgentService] = None


def get_default_service() -> TravelAgentService:
    global _default_service
    if _default_service is None:
        _default_service = create_travel_service(checkpointer=InMemorySaver())
    return _default_service


# =========================
# Serialization Helpers
# =========================
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None
    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(result: dict[str, Any], thread_id: str) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    requires_approval = False
    if interrupt_payload:
        itinerary_in_interrupt = interrupt_payload.get("draft_itinerary") or result.get("itinerary", "")
        # Phase 6: never present approval gate for a missing/failed itinerary
        if itinerary_in_interrupt and itinerary_in_interrupt.strip() != _ITINERARY_FAILURE_SENTINEL:
            requires_approval = True
            answer = itinerary_in_interrupt
        else:
            # Interrupt with degraded itinerary — treat as degraded, no approval
            answer = result.get("final_response") or last_message

    # Safe public capability trace
    raw_trace = result.get("capability_trace", [])
    safe_trace = []
    for item in raw_trace:
        if isinstance(item, dict):
            safe_trace.append({
                "specialist": item.get("specialist", ""),
                "capability": item.get("capability", ""),
                "server": item.get("server", ""),
                "status": item.get("status", ""),
                "latency_ms": item.get("latency_ms"),
                "source_count": item.get("source_count", 0),
                "error_code": item.get("error_code"),
            })

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": requires_approval,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if requires_approval and interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if requires_approval and interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "specialist_statuses": result.get("specialist_statuses", {}),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "review_iteration": result.get("review_iteration", 0),
        "review_limit_reached": result.get("review_limit_reached", False),
        "capability_trace": safe_trace,
        "llm_calls": result.get("llm_calls", 0),
        "llm_token_usage": result.get("llm_token_usage", {}),  # Phase 8: internal trace
        "run_status": result.get("run_status", RunStatus.COMPLETED.value),
    }


# =========================
# Public Async API Functions
# =========================
async def run_travel_agent(
    user_input: str,
    thread_id: str | None = None,
    service: TravelAgentService | None = None,
) -> dict[str, Any]:
    svc = service or get_default_service()
    return await svc.run(user_input, thread_id=thread_id)


async def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
    service: TravelAgentService | None = None,
) -> dict[str, Any]:
    svc = service or get_default_service()
    return await svc.resume(thread_id, approved=approved, feedback=feedback)
