import os
import certifi
from project_config import load_project_env

load_project_env()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

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
)
from agent_config import PROVIDER_TIMEOUT_SECONDS, MAX_REVIEW_ITERATIONS
from provider_utils import async_call_provider


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )
    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"
    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

DATABASE_URL = get_database_url()

# =========================
# LLM instances
# =========================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
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
        # Deterministic keyword fallback — does NOT count as an LLM call
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

        # Do NOT force itinerary_agent on specialist-only queries!
        # If no specialist was selected at all, default to itinerary_agent
        if not selected_agents:
            selected_agents = [AgentName.ITINERARY.value]

        constraints = decision.constraints.to_dict()
        reasoning = decision.reasoning.strip()

    except Exception as exc:
        print(f"[supervisor] Structured output failed, using safe defaults: {exc}", flush=True)
        # Fallback selects full trip specialists
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
# Flight Agent
# =========================
FLIGHT_AGENT_PROMPT = (
    "You are a travel flight expert.\n\n"
    "User Query:\n{query}\n\n"
    "Airport Information:\n{airport_data}\n\n"
    "Airline Information:\n{airline_data}\n\n"
    "Generate:\n"
    "1. Likely departure airport\n"
    "2. Likely arrival airport\n"
    "3. Airlines serving this route\n"
    "4. Typical flight duration\n"
    "5. Estimated airfare range (clearly labeled as estimate)\n"
    "6. Peak season pricing warning\n"
    "7. Booking advice\n\n"
    "Return concise travel guidance."
)


async def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")
    query = state["user_query"]
    statuses = dict(state.get("specialist_statuses", {}))
    llm_calls = state.get("llm_calls", 0)

    airports_result: CapabilityResult = await async_call_provider(
        lambda: aviation_mcp_call("list_airports"),
        provider_name="aviation",
        safe_failure_message=FLIGHT_UNAVAILABLE_MESSAGE,
    )
    if not airports_result.success:
        statuses["flight_agent"] = "DEGRADED"
        return {
            "flight_results": airports_result.safe_message,
            "flight_data_available": False,
            "specialist_statuses": statuses,
            "messages": [AIMessage(content="Flight data unavailable.")],
            "llm_calls": llm_calls,
        }

    airlines_result: CapabilityResult = await async_call_provider(
        lambda: aviation_mcp_call("list_airlines"),
        provider_name="aviation",
        safe_failure_message=FLIGHT_UNAVAILABLE_MESSAGE,
    )
    if not airlines_result.success:
        statuses["flight_agent"] = "DEGRADED"
        return {
            "flight_results": airlines_result.safe_message,
            "flight_data_available": False,
            "specialist_statuses": statuses,
            "messages": [AIMessage(content="Flight data unavailable.")],
            "llm_calls": llm_calls,
        }

    try:
        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports_result.data)[:3000],
            airline_data=str(airlines_result.data)[:3000],
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
            "messages": [AIMessage(content="Flight data unavailable.")],
            "llm_calls": llm_calls,
        }


# =========================
# Hotel Agent — Tavily only, NO LLM call
# =========================
async def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    statuses = dict(state.get("specialist_statuses", {}))
    # hotel_agent does NOT call the LLM — Tavily search only.
    # llm_calls is NOT incremented here.

    result: CapabilityResult = await async_call_provider(
        lambda: tavily_mcp_search(query),
        provider_name="tavily",
        safe_failure_message=HOTEL_UNAVAILABLE_MESSAGE,
    )
    if result.success:
        statuses["hotel_agent"] = "COMPLETED"
        return {
            "hotel_results": result.data,
            "hotel_data_available": True,
            "specialist_statuses": statuses,
            "messages": [AIMessage(content="Hotel information processed.")],
            "llm_calls": state.get("llm_calls", 0),
        }
    else:
        statuses["hotel_agent"] = "DEGRADED"
        return {
            "hotel_results": result.safe_message,
            "hotel_data_available": False,
            "specialist_statuses": statuses,
            "messages": [AIMessage(content="Hotel data unavailable.")],
            "llm_calls": state.get("llm_calls", 0),
        }


# =========================
# Weather Agent
# =========================
async def weather_agent(state: TravelState):
    statuses = dict(state.get("specialist_statuses", {}))
    llm_calls = state.get("llm_calls", 0)

    # extract_destination_async is an LLM call — count it
    try:
        async with asyncio.timeout(PROVIDER_TIMEOUT_SECONDS["llm"]):
            city = await extract_destination_async(state["user_query"])
        llm_calls += 1
    except Exception as exc:
        print(f"[weather_agent] Destination extraction failed: {exc}", flush=True)
        city = state.get("trip_constraints", {}).get("destination") or "your destination"

    weather_result: CapabilityResult = await async_call_provider(
        lambda: weather_mcp_search(city),
        provider_name="weather",
        safe_failure_message=WEATHER_UNAVAILABLE_MESSAGE,
    )
    forecast_result: CapabilityResult = await async_call_provider(
        lambda: forecast_mcp_search(city),
        provider_name="weather",
        safe_failure_message=WEATHER_UNAVAILABLE_MESSAGE,
    )

    if weather_result.success and forecast_result.success:
        weather_text = (
            f"Current Weather:\n{weather_result.data}\n\n"
            f"Forecast:\n{forecast_result.data}\n"
        )
        statuses["weather_agent"] = "COMPLETED"
        return {
            "weather_results": weather_text,
            "weather_data_available": True,
            "specialist_statuses": statuses,
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
        return {
            "weather_results": weather_text,
            "weather_data_available": False,
            "specialist_statuses": statuses,
            "messages": [AIMessage(content="Weather data unavailable.")],
            "llm_calls": llm_calls,
        }


# =========================
# Budget Agent
# =========================
async def budget_agent(state: TravelState):
    statuses = dict(state.get("specialist_statuses", {}))
    prompt = (
        "Analyze whether this trip is realistic for the user's budget.\n\n"
        f"User Query:\n{state['user_query']}\n\n"
        f"Trip Constraints:\n{state.get('trip_constraints', {})}\n\n"
        f"Flight Results:\n{state.get('flight_results', '')}\n\n"
        f"Hotel Results:\n{state.get('hotel_results', '')}\n\n"
        f"Weather Results:\n{state.get('weather_results', '')}\n\n"
        "CRITICAL GROUNDEDNESS RULES:\n"
        "- If flight results indicate live flight data is unavailable, DO NOT invent exact flight fares or prices. Explicitly note that airfare must be checked and budgeted separately.\n"
        "- Clearly differentiate between estimated category ranges and verified live prices.\n"
        "- If exact prices are unavailable, provide realistic category ranges and explicitly state they are estimates.\n\n"
        "Return:\n1. Estimated cost categories\n2. Budget risk areas\n3. Money-saving suggestions\n4. Overall feasibility"
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
    prompt = (
        "Create a complete travel itinerary for human review.\n\n"
        f"User Query:\n{state['user_query']}\n\n"
        f"Trip Constraints:\n{state.get('trip_constraints', {})}\n\n"
        f"Flight Results:\n{state.get('flight_results', '')}\n\n"
        f"Hotel Results:\n{state.get('hotel_results', '')}\n\n"
        f"Weather Results:\n{state.get('weather_results', '')}\n\n"
        f"Budget Results:\n{state.get('budget_results', '')}\n\n"
        "CRITICAL GROUNDEDNESS RULES:\n"
        "- If flight results indicate live flight information is unavailable, DO NOT invent or fabricate flight prices (e.g. specific currency ranges), flight numbers, or specific airline schedules. Instead, state clearly: 'Live flight data is unavailable; check schedules and fares directly with airlines or booking portals.'\n"
        "- If hotel information is general or unavailable, do not claim specific live room rates.\n"
        "- If weather is unavailable, provide seasonal tips and remind the traveler to check live forecasts.\n"
        "- Never fabricate specific numbers, fares, or live booking details when specialist data is marked unavailable.\n\n"
        "Make the itinerary practical, budget-aware, and easy to follow.\nCreate a clear draft that is ready for human review."
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

    prompt = (
        "You previously drafted a travel itinerary. The human reviewer has requested a revision.\n\n"
        f"Original User Query:\n{state['user_query']}\n\n"
        f"Human Reviewer Feedback (apply carefully):\n{feedback}\n\n"
        f"Previous Draft Itinerary:\n{state.get('itinerary', '')}\n\n"
        f"Trip Constraints:\n{state.get('trip_constraints', {})}\n\n"
        f"Flight Results:\n{state.get('flight_results', '')}\n\n"
        "CRITICAL GROUNDEDNESS RULES:\n"
        "- If flight results indicate live flight information is unavailable, DO NOT invent or fabricate flight prices, flight numbers, or specific airline schedules.\n"
        "- Never fabricate specific numbers, fares, or live booking details when specialist data is marked unavailable.\n"
        "- Apply the feedback accurately and completely.\n\n"
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
        f"Flights:\n{state.get('flight_results', '')}\n\n"
        f"Hotels:\n{state.get('hotel_results', '')}\n\n"
        f"Weather:\n{state.get('weather_results', '')}\n\n"
        f"Budget Analysis:\n{state.get('budget_results', '')}\n\n"
        f"Draft Itinerary:\n{state.get('itinerary', '')}\n\n"
        "CRITICAL GROUNDEDNESS & TRUTHFULNESS RULES:\n"
        "- If flight data indicates live flight data was unavailable, DO NOT fabricate flight prices, fares (e.g. \u20b9X to \u20b9Y), flight numbers, or specific airlines from imagination. In the Flight Information section, state clearly: 'Live flight information was unavailable for this run. Verify schedules and fares with an airline or booking provider before booking.'\n"
        "- If hotel data was unavailable or non-live, provide neighborhood guidance and state that prices must be confirmed.\n"
        "- If weather was unavailable, provide seasonal advice and advise checking forecasts before departure.\n"
        "- Do not fabricate live booking codes or specific unverified fares.\n\n"
        "Format the final answer clearly and beautifully using relevant sections suited to the query.\n"
        "Be clear and practical. Keep the response useful for real travel planning."
    )

    try:
        async with asyncio.timeout(PROVIDER_TIMEOUT_SECONDS["llm"]):
            response = await llm.ainvoke(
                [
                    SystemMessage(content="You are a professional AI travel booking assistant."),
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
# Review Limit Reached Node (TF1 architectural correction)
# =========================
async def review_limit_reached_agent(state: TravelState):
    """Terminal node reached when max revision iterations are exhausted without approval.

    NEVER produces a normal approved travel plan. Returns a stable REVIEW_LIMIT_REACHED state.
    """
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
        # If no more selected specialists:
        if "itinerary_agent" in selected:
            return "itinerary_agent"
        return "final_agent"
    return route


def route_after_review(state: TravelState) -> str:
    iteration = state.get("review_iteration", 0)
    approved = state.get("approved", False)

    if approved:
        return "final_agent"

    # If the user rejects and we have reached the max review iterations (e.g. 10th rejection on iteration 9)
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
    """Build and compile the LangGraph travel graph with an optional checkpointer."""
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
    """Application-scoped service wrapping the compiled graph and its checkpointer lifecycle."""

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
    """Factory to create a TravelAgentService with the given checkpointer."""
    compiled = build_travel_graph(checkpointer=checkpointer)
    return TravelAgentService(compiled)


# Default fallback service using InMemorySaver (for tests or standalone scripts)
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
        "llm_calls": result.get("llm_calls", 0),
    }


# =========================
# Public Async API Functions (delegates to service)
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
