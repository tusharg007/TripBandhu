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
from langchain_groq import ChatGroq

from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination_async,
    forecast_mcp_search,
    weather_mcp_search,
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
    EvidenceKind,
    SourceReference,
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
from agent_config import PROVIDER_TIMEOUT_SECONDS, MAX_REVIEW_ITERATIONS
from provider_utils import async_call_provider


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
# LLM instances
# =========================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY or "dummy_ci_groq_key",
)

_llm_supervisor = llm.with_structured_output(SupervisorDecision)
_llm_guardrail = llm.with_structured_output(GuardrailDecision)


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

    llm_calls: int


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
WEATHER_UNAVAILABLE_MESSAGE = (
    "Live weather forecast is temporarily unavailable for this destination. "
    "General seasonal guidance will be provided."
)

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

    guardrail_system = (
        "You are the input guardrail for a travel-planning application. "
        "Determine whether the request is travel-related. "
        "Block clearly unrelated, harmful, or illegal requests. "
        "Do not block valid travel requests merely because details are missing."
    )
    guardrail_user = f"User request:\n{query}"

    try:
        async with asyncio.timeout(PROVIDER_TIMEOUT_SECONDS["llm_structured"]):
            gd: GuardrailDecision = await _llm_guardrail.ainvoke(
                [SystemMessage(content=guardrail_system), HumanMessage(content=guardrail_user)]
            )
        llm_calls += 1
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
        async with asyncio.timeout(PROVIDER_TIMEOUT_SECONDS["llm_structured"]):
            decision: SupervisorDecision = await _llm_supervisor.ainvoke(
                [SystemMessage(content=supervisor_system), HumanMessage(content=supervisor_user)]
            )
        llm_calls += 1

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
    trace = list(state.get("capability_trace", []))
    evidence_store = dict(state.get("evidence_store", {}))

    # Validate permission
    validate_specialist_capability("flight_agent", "FLIGHT_ROUTE_SEARCH")

    # 1. Attempt Route-Specific Search (list_routes)
    routes_result = await async_call_provider(
        lambda: aviation_mcp_call("list_routes"),
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

    # 2. Reference Airport & Airline Lookups
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

    # If all AviationStack calls failed -> Degraded
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
        }

    # Normalize evidence
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
        async with asyncio.timeout(PROVIDER_TIMEOUT_SECONDS["llm"]):
            response = await llm.ainvoke(
                [
                    SystemMessage(content="You are an expert travel flight planner."),
                    HumanMessage(content=prompt),
                ]
            )
        statuses["flight_agent"] = "COMPLETED"
        return {
            "flight_results": response.content,
            "flight_data_available": True,
            "specialist_statuses": statuses,
            "capability_trace": trace,
            "evidence_store": evidence_store,
            "messages": [AIMessage(content="Flight recommendations generated.")],
            "llm_calls": llm_calls + 1,
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
        }


# =========================
# Hotel Agent — Tavily Web Research & Source Preservation
# =========================
async def hotel_agent(state: TravelState):
    query = f"Best hotels, neighborhoods, and places to stay for {state['user_query']}"
    statuses = dict(state.get("specialist_statuses", {}))
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
        statuses["hotel_agent"] = "COMPLETED"
        hotel_ev = normalize_tavily_results(result.data, query, latency_ms=result.latency_ms)
        evidence_store["HOTEL_WEB_RESEARCH"] = hotel_ev.model_dump()

        return {
            "hotel_results": hotel_ev.summary,
            "hotel_data_available": True,
            "specialist_statuses": statuses,
            "capability_trace": trace,
            "evidence_store": evidence_store,
            "messages": [AIMessage(content="Hotel information processed.")],
            "llm_calls": state.get("llm_calls", 0),
        }
    else:
        statuses["hotel_agent"] = "DEGRADED"
        ev_unavailable = CapabilityEvidence(
            capability="HOTEL_WEB_RESEARCH",
            provider="tavily",
            tool_name="tavily_search",
            evidence_kind=EvidenceKind.UNAVAILABLE,
            status=CapabilityHealth.UNAVAILABLE,
            summary=HOTEL_UNAVAILABLE_MESSAGE,
            error_code=result.error_code,
        )
        evidence_store["HOTEL_WEB_RESEARCH"] = ev_unavailable.model_dump()

        return {
            "hotel_results": result.safe_message,
            "hotel_data_available": False,
            "specialist_statuses": statuses,
            "capability_trace": trace,
            "evidence_store": evidence_store,
            "messages": [AIMessage(content="Hotel data unavailable.")],
            "llm_calls": state.get("llm_calls", 0),
        }


# =========================
# Weather Agent — OpenWeather MCP & Normalization
# =========================
async def weather_agent(state: TravelState):
    statuses = dict(state.get("specialist_statuses", {}))
    llm_calls = state.get("llm_calls", 0)
    trace = list(state.get("capability_trace", []))
    evidence_store = dict(state.get("evidence_store", {}))

    validate_specialist_capability("weather_agent", "WEATHER_CURRENT")

    try:
        async with asyncio.timeout(PROVIDER_TIMEOUT_SECONDS["llm"]):
            city = await extract_destination_async(state["user_query"])
        llm_calls += 1
    except Exception as exc:
        print(f"[weather_agent] Destination extraction failed: {exc}", flush=True)
        city = state.get("trip_constraints", {}).get("destination") or "your destination"

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

    if weather_result.success and forecast_result.success:
        weather_ev = normalize_weather_results(
            current_data=weather_result.data,
            forecast_data=forecast_result.data,
            city=city,
            latency_ms=(weather_result.latency_ms or 0) + (forecast_result.latency_ms or 0),
        )
        evidence_store["WEATHER_FORECAST"] = weather_ev.model_dump()
        statuses["weather_agent"] = "COMPLETED"

        return {
            "weather_results": weather_ev.summary,
            "weather_data_available": True,
            "specialist_statuses": statuses,
            "capability_trace": trace,
            "evidence_store": evidence_store,
            "messages": [AIMessage(content="Weather information processed.")],
            "llm_calls": llm_calls,
        }
    else:
        dest = state.get("trip_constraints", {}).get("destination") or city or "your destination"
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
        }


# =========================
# Budget Agent — Grounded Budget Feasibility & Pricing Truth
# =========================
async def budget_agent(state: TravelState):
    statuses = dict(state.get("specialist_statuses", {}))
    evidence_context = format_grounded_evidence_context(state.get("evidence_store", {}))

    prompt = (
        "Analyze whether this trip is realistic for the user's budget.\n\n"
        f"User Query:\n{state['user_query']}\n\n"
        f"Trip Constraints:\n{state.get('trip_constraints', {})}\n\n"
        f"Grounded Provider & Web Evidence:\n{evidence_context}\n\n"
        "CRITICAL GROUNDEDNESS RULES:\n"
        "- DO NOT invent exact flight fares or prices if flight data is unavailable or schedule-only.\n"
        "- All flight cost estimations MUST be explicitly marked: '[MODEL ESTIMATE - Not Live Fare]'.\n"
        "- AviationStack does not supply live ticket prices. If flight data is unavailable or schedule-only, state clearly that airfare is estimated and must be verified.\n"
        "- Web hotel rates are search-based approximations ([WEB_SOURCE]); do not present them as locked booking rates.\n"
        "- Clearly differentiate verified provider data from model estimation categories.\n\n"
        "Return:\n1. Estimated cost breakdown by category\n2. Budget feasibility assessment\n3. Cost-saving recommendations\n4. Risk areas"
    )

    try:
        async with asyncio.timeout(PROVIDER_TIMEOUT_SECONDS["llm"]):
            response = await llm.ainvoke(
                [
                    SystemMessage(content="You are a practical travel budget analyst."),
                    HumanMessage(content=prompt),
                ]
            )
        statuses["budget_agent"] = "COMPLETED"
        return {
            "budget_results": response.content,
            "specialist_statuses": statuses,
            "messages": [AIMessage(content="Budget assessment generated.")],
            "llm_calls": _incr_llm(state),
        }
    except Exception as exc:
        print(f"[budget_agent] LLM call failed: {exc}", flush=True)
        statuses["budget_agent"] = "DEGRADED"
        return {
            "budget_results": "Budget estimation could not be generated.",
            "specialist_statuses": statuses,
            "messages": [AIMessage(content="Budget estimation failed.")],
            "llm_calls": state.get("llm_calls", 0),
        }


# =========================
# Itinerary Agent
# =========================
async def itinerary_agent(state: TravelState):
    statuses = dict(state.get("specialist_statuses", {}))
    evidence_context = format_grounded_evidence_context(state.get("evidence_store", {}))

    prompt = (
        "Create a complete travel itinerary for human review.\n\n"
        f"User Query:\n{state['user_query']}\n\n"
        f"Trip Constraints:\n{state.get('trip_constraints', {})}\n\n"
        f"Grounded Evidence & Provenance:\n{evidence_context}\n\n"
        f"Budget Analysis:\n{state.get('budget_results', '')}\n\n"
        "CRITICAL GROUNDEDNESS RULES:\n"
        "- DO NOT invent or fabricate flight prices or live flight schedules if flight data is unavailable.\n"
        "- If flight data is marked [UNAVAILABLE], state clearly: 'Live flight data was unavailable; check schedules with airlines.' Do NOT fabricate fake flight numbers or fake price ranges.\n"
        "- For hotel recommendations from web sources ([WEB_SOURCE]), preserve neighborhood and accommodation names.\n"
        "- Never invent factual data that contradicts the Grounded Evidence section.\n\n"
        "Make the itinerary practical, budget-aware, and ready for human review."
    )

    try:
        async with asyncio.timeout(PROVIDER_TIMEOUT_SECONDS["llm"]):
            response = await llm.ainvoke(
                [
                    SystemMessage(content="You are an expert travel planner."),
                    HumanMessage(content=prompt),
                ]
            )
        statuses["itinerary_agent"] = "COMPLETED"
        itinerary_content = response.content
        llm_calls = _incr_llm(state)
    except Exception as exc:
        print(f"[itinerary_agent] LLM call failed: {exc}", flush=True)
        statuses["itinerary_agent"] = "DEGRADED"
        itinerary_content = "Draft itinerary could not be generated."
        llm_calls = state.get("llm_calls", 0)

    return {
        "itinerary": itinerary_content,
        "approval_request": (
            "Please review the generated draft itinerary. Approve it to create the "
            "final polished plan, or provide feedback for revision."
        ),
        "specialist_statuses": statuses,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": llm_calls,
    }


# =========================
# Itinerary Revision Agent
# =========================
async def itinerary_revision_agent(state: TravelState):
    statuses = dict(state.get("specialist_statuses", {}))
    feedback = state.get("human_feedback", "").strip() or "Please improve the draft itinerary."
    iteration = state.get("review_iteration", 0)
    evidence_context = format_grounded_evidence_context(state.get("evidence_store", {}))

    prompt = (
        "You previously drafted a travel itinerary. The human reviewer has requested a revision.\n\n"
        f"Original User Query:\n{state['user_query']}\n\n"
        f"Human Reviewer Feedback (apply carefully):\n{feedback}\n\n"
        f"Previous Draft Itinerary:\n{state.get('itinerary', '')}\n\n"
        f"Grounded Evidence & Provenance:\n{evidence_context}\n\n"
        "CRITICAL GROUNDEDNESS RULES:\n"
        "- Never fabricate specific fares or live booking details when specialist data is marked unavailable.\n"
        "- Apply the feedback accurately while maintaining grounded facts.\n\n"
        "Produce a revised travel itinerary that directly addresses the feedback."
    )

    try:
        async with asyncio.timeout(PROVIDER_TIMEOUT_SECONDS["llm"]):
            response = await llm.ainvoke(
                [
                    SystemMessage(content="You are an expert travel planner revising an itinerary based on feedback."),
                    HumanMessage(content=prompt),
                ]
            )
        statuses["itinerary_agent"] = "COMPLETED"
        revised_itinerary = response.content
        llm_calls = _incr_llm(state)
    except Exception as exc:
        print(f"[itinerary_revision_agent] LLM call failed: {exc}", flush=True)
        statuses["itinerary_agent"] = "DEGRADED"
        revised_itinerary = state.get("itinerary", "Draft itinerary could not be revised.")
        llm_calls = state.get("llm_calls", 0)

    return {
        "itinerary": revised_itinerary,
        "approval_request": (
            f"Revised itinerary (revision {iteration + 1}). "
            "Please review and approve, or provide further feedback."
        ),
        "specialist_statuses": statuses,
        "review_iteration": iteration + 1,
        "messages": [AIMessage(content=f"Itinerary revised (iteration {iteration + 1}).")],
        "llm_calls": llm_calls,
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
# =========================
async def final_agent(state: TravelState):
    iteration = state.get("review_iteration", 0)
    is_specialist_only = "itinerary_agent" not in state.get("selected_agents", [])
    evidence_context = format_grounded_evidence_context(state.get("evidence_store", {}))

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

    final_prompt = (
        "Generate the final travel response for the user.\n\n"
        f"Context / Instructions:\n{review_instruction}\n\n"
        f"User Request:\n{state['user_query']}\n\n"
        f"Supervisor Constraints:\n{state.get('trip_constraints', {})}\n\n"
        f"Grounded Evidence & Provenance:\n{evidence_context}\n\n"
        f"Flight Summary:\n{state.get('flight_results', '')}\n\n"
        f"Hotel Summary:\n{state.get('hotel_results', '')}\n\n"
        f"Weather Summary:\n{state.get('weather_results', '')}\n\n"
        f"Budget Analysis:\n{state.get('budget_results', '')}\n\n"
        f"Draft Itinerary:\n{state.get('itinerary', '')}\n\n"
        "CRITICAL GROUNDEDNESS & TRUTHFULNESS RULES:\n"
        "- When presenting flight options, describe them as 'Flight Route Reference' (Provider data · Reference — verify current schedule before booking). Never describe route reference data as confirmed flights or live bookings.\n"
        "- When presenting the approved itinerary, label the section 'Approved Day-by-Day Itinerary' or 'Day-by-Day Itinerary' — NEVER 'Confirmed Schedule' or 'Confirmed Itinerary' since reservations are not made.\n"
        "- If flight data is unavailable or degraded, state: 'Live flight information was unavailable for this run. Verify schedules and fares with an airline or booking provider before booking.'\n"
        "- All estimated price ranges must be explicitly marked '[MODEL ESTIMATE - Not Live Fare]'.\n"
        "- Format the final answer beautifully with markdown sections suited to the query.\n"
        "Be clear, realistic, and practical."
    )

    try:
        async with asyncio.timeout(PROVIDER_TIMEOUT_SECONDS["llm"]):
            response = await llm.ainvoke(
                [
                    SystemMessage(content="You are a professional AI travel research assistant."),
                    HumanMessage(content=final_prompt),
                ]
            )
        return {
            "final_response": response.content,
            "messages": [response],
            "llm_calls": _incr_llm(state),
        }
    except Exception as exc:
        print(f"[final_agent] LLM call failed: {exc}", flush=True)
        return {
            "final_response": "The final travel plan could not be generated. Please try again.",
            "messages": [AIMessage(content="Final plan generation failed.")],
            "llm_calls": state.get("llm_calls", 0),
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
    g.add_node("human_review", human_review_node)
    g.add_node("final_agent", final_agent)
    g.add_node("review_limit_reached", review_limit_reached_agent)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)
    g.add_conditional_edges("flight_agent", route_after_agent("flight_agent"), ROUTE_MAP)
    g.add_conditional_edges("hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP)
    g.add_conditional_edges("weather_agent", route_after_agent("weather_agent"), ROUTE_MAP)
    g.add_conditional_edges("budget_agent", route_after_agent("budget_agent"), ROUTE_MAP)

    g.add_edge("itinerary_agent", "human_review")
    g.add_edge("itinerary_revision", "human_review")
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

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get("itinerary", "")

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
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
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
