"""
capability_registry.py — TF2 MCP Capability Registry, Normalization & Provenance Layer.

Defines approved capability definitions, specialist permission maps, deterministic
output normalizers, and grounding context packagers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from schemas import (
    CapabilityDefinition,
    CapabilityEvidence,
    CapabilityHealth,
    EvidenceKind,
    SourceReference,
    ErrorCode,
)


# ===========================================================================
# 1. Canonical Approved Capability Registry
# ===========================================================================

CAPABILITY_REGISTRY: dict[str, CapabilityDefinition] = {
    "FLIGHT_ROUTE_SEARCH": CapabilityDefinition(
        capability_name="FLIGHT_ROUTE_SEARCH",
        server="aviationstack",
        tool_name="list_routes",
        access_mode="read_only",
        evidence_kind=EvidenceKind.LIVE_PROVIDER,
        description="Search commercial flight routes between departure and arrival IATA airport codes.",
        required_params=[],
    ),
    "AIRPORT_LOOKUP": CapabilityDefinition(
        capability_name="AIRPORT_LOOKUP",
        server="aviationstack",
        tool_name="list_airports",
        access_mode="read_only",
        evidence_kind=EvidenceKind.LIVE_PROVIDER,
        description="Lookup airport metadata and IATA codes.",
        required_params=[],
    ),
    "AIRLINE_LOOKUP": CapabilityDefinition(
        capability_name="AIRLINE_LOOKUP",
        server="aviationstack",
        tool_name="list_airlines",
        access_mode="read_only",
        evidence_kind=EvidenceKind.LIVE_PROVIDER,
        description="Lookup airline names and IATA codes.",
        required_params=[],
    ),
    "HOTEL_WEB_RESEARCH": CapabilityDefinition(
        capability_name="HOTEL_WEB_RESEARCH",
        server="tavily",
        tool_name="tavily_search",
        access_mode="read_only",
        evidence_kind=EvidenceKind.WEB_SOURCE,
        description="Web search for hotels, accommodations, neighborhoods, and reviews.",
        required_params=["query"],
    ),
    "WEATHER_CURRENT": CapabilityDefinition(
        capability_name="WEATHER_CURRENT",
        server="weather",
        tool_name="get_current_weather",
        access_mode="read_only",
        evidence_kind=EvidenceKind.LIVE_PROVIDER,
        description="Real-time current meteorological observations for a destination city.",
        required_params=["city"],
    ),
    "WEATHER_FORECAST": CapabilityDefinition(
        capability_name="WEATHER_FORECAST",
        server="weather",
        tool_name="get_forecast",
        access_mode="read_only",
        evidence_kind=EvidenceKind.LIVE_PROVIDER,
        description="Multi-day meteorological forecast entries for a destination city.",
        required_params=["city"],
    ),
}


# ===========================================================================
# 2. Specialist Allowed Capability Mapping (Strict Verification)
# ===========================================================================

SPECIALIST_ALLOWED_CAPABILITIES: dict[str, frozenset[str]] = {
    "flight_agent": frozenset({"FLIGHT_ROUTE_SEARCH", "AIRPORT_LOOKUP", "AIRLINE_LOOKUP"}),
    "hotel_agent": frozenset({"HOTEL_WEB_RESEARCH"}),
    "weather_agent": frozenset({"WEATHER_CURRENT", "WEATHER_FORECAST"}),
}


def validate_specialist_capability(specialist: str, capability_name: str) -> CapabilityDefinition:
    """Validate that a specialist is permitted to execute the given capability.

    Raises:
        ValueError: If capability is unknown or not permitted for the specialist.
    """
    if capability_name not in CAPABILITY_REGISTRY:
        raise ValueError(f"Unknown capability '{capability_name}' is not in the TripBandhu Capability Registry.")

    allowed = SPECIALIST_ALLOWED_CAPABILITIES.get(specialist, frozenset())
    if capability_name not in allowed:
        raise ValueError(
            f"Specialist '{specialist}' is not permitted to execute capability '{capability_name}'. "
            f"Allowed capabilities: {sorted(allowed)}"
        )

    return CAPABILITY_REGISTRY[capability_name]


# ===========================================================================
# 3. Output Normalizers with Source Provenance
# ===========================================================================

def normalize_tavily_results(raw_data: Any, query: str, latency_ms: Optional[int] = None) -> CapabilityEvidence:
    """Normalize raw Tavily search output into bounded structured evidence with source links."""
    now_iso = datetime.now(timezone.utc).isoformat()
    sources: list[SourceReference] = []
    items: list[dict[str, Any]] = []

    # Handle string, list, or dict raw formats safely
    raw_results = []
    if isinstance(raw_data, dict):
        raw_results = raw_data.get("results", []) or raw_data.get("data", [])
        if not raw_results and "content" in raw_data:
            raw_results = [{"title": f"Tavily Search: {query}", "url": "", "content": raw_data["content"]}]
    elif isinstance(raw_data, list):
        raw_results = raw_data
    elif isinstance(raw_data, str):
        try:
            parsed = json.loads(raw_data)
            if isinstance(parsed, dict):
                raw_results = parsed.get("results", [parsed])
            elif isinstance(parsed, list):
                raw_results = parsed
            else:
                raw_results = [{"title": f"Search: {query}", "url": "", "content": str(raw_data)}]
        except Exception:
            raw_results = [{"title": f"Search: {query}", "url": "", "content": raw_data}]

    # Bound top results to 5-8 items to prevent prompt bloat
    bounded_results = raw_results[:6]
    summary_lines = []

    for item in bounded_results:
        if isinstance(item, dict):
            title = str(item.get("title") or item.get("name") or "Web Search Result").strip()
            url = str(item.get("url") or "").strip()
            content = str(item.get("content") or item.get("snippet") or item.get("body") or "").strip()
        else:
            title = f"Web Search Result for {query}"
            url = ""
            content = str(item).strip()

        sources.append(
            SourceReference(
                provider="Tavily Web Search",
                title=title,
                url=url if url else None,
                observed_at=now_iso,
                evidence_kind=EvidenceKind.WEB_SOURCE,
            )
        )
        items.append({"title": title, "url": url, "snippet": content[:350]})
        url_md = f" ([Source]({url}))" if url else ""
        summary_lines.append(f"- **{title}**{url_md}: {content[:250]}")

    formatted_summary = "\n".join(summary_lines) if summary_lines else str(raw_data)[:1000]

    return CapabilityEvidence(
        capability="HOTEL_WEB_RESEARCH",
        provider="tavily",
        tool_name="tavily_search",
        evidence_kind=EvidenceKind.WEB_SOURCE,
        status=CapabilityHealth.AVAILABLE,
        retrieved_at=now_iso,
        latency_ms=latency_ms,
        data=items,
        summary=formatted_summary,
        sources=sources,
    )


def normalize_weather_results(
    current_data: Any,
    forecast_data: Any,
    city: str,
    latency_ms: Optional[int] = None,
) -> CapabilityEvidence:
    """Normalize OpenWeather observations and forecast entries into typed evidence."""
    now_iso = datetime.now(timezone.utc).isoformat()
    sources = [
        SourceReference(
            provider="OpenWeather API",
            title=f"Live Weather Observation for {city}",
            url="https://openweathermap.org",
            observed_at=now_iso,
            evidence_kind=EvidenceKind.LIVE_PROVIDER,
        )
    ]

    structured_data = {
        "city": city,
        "current": current_data,
        "forecast": forecast_data,
    }

    # Format concise summary
    curr_str = current_data if isinstance(current_data, str) else json.dumps(current_data, indent=2)
    fore_str = forecast_data if isinstance(forecast_data, str) else json.dumps(forecast_data, indent=2)
    summary = f"**Current Weather for {city}:**\n{curr_str}\n\n**Forecast:**\n{fore_str}"

    return CapabilityEvidence(
        capability="WEATHER_FORECAST",
        provider="weather",
        tool_name="get_current_weather,get_forecast",
        evidence_kind=EvidenceKind.LIVE_PROVIDER,
        status=CapabilityHealth.AVAILABLE,
        retrieved_at=now_iso,
        latency_ms=latency_ms,
        data=structured_data,
        summary=summary,
        sources=sources,
    )


def normalize_flight_results(
    routes_data: Any,
    airports_data: Any,
    airlines_data: Any,
    destination: str,
    origin: str = "",
    latency_ms: Optional[int] = None,
) -> CapabilityEvidence:
    """Normalize AviationStack route and airport data into typed evidence."""
    now_iso = datetime.now(timezone.utc).isoformat()
    sources = [
        SourceReference(
            provider="AviationStack Schedule & Route API",
            title=f"Flight Routes & Airports for {origin or 'Origin'} -> {destination}",
            url="https://aviationstack.com",
            observed_at=now_iso,
            evidence_kind=EvidenceKind.LIVE_PROVIDER,
        )
    ]

    structured_data = {
        "destination": destination,
        "origin": origin,
        "routes": routes_data,
        "airports": airports_data,
        "airlines": airlines_data,
    }

    routes_str = str(routes_data)[:800] if routes_data else "No specific routes returned."
    airports_str = str(airports_data)[:600] if airports_data else ""
    airlines_str = str(airlines_data)[:600] if airlines_data else ""

    summary = (
        f"**Route Information ({origin or 'Origin'} -> {destination}):**\n{routes_str}\n\n"
        f"**Serving Airports:**\n{airports_str}\n\n"
        f"**Operating Airlines:**\n{airlines_str}"
    )

    return CapabilityEvidence(
        capability="FLIGHT_ROUTE_SEARCH",
        provider="aviationstack",
        tool_name="list_routes,list_airports,list_airlines",
        evidence_kind=EvidenceKind.LIVE_PROVIDER,
        status=CapabilityHealth.AVAILABLE,
        retrieved_at=now_iso,
        latency_ms=latency_ms,
        data=structured_data,
        summary=summary,
        sources=sources,
    )


# ===========================================================================
# 4. Grounding Context Packager (Evidence Kind Tagging)
# ===========================================================================

def format_grounded_evidence_context(evidence_store: dict[str, dict]) -> str:
    """Pack normalized evidence into a deterministic, clearly labeled prompt context.

    Explicitly tags each section with [LIVE_PROVIDER], [WEB_SOURCE], [MODEL_ESTIMATE],
    or [UNAVAILABLE] to prevent model fabrication and maintain truthful grounding.
    """
    if not evidence_store:
        return "No external provider evidence was retrieved for this run."

    sections: list[str] = []

    for cap_name, ev_dict in evidence_store.items():
        ev = CapabilityEvidence(**ev_dict) if isinstance(ev_dict, dict) else ev_dict
        kind_tag = f"[{ev.evidence_kind.value}]"
        status_tag = f"Status: {ev.status.value}"
        provider_info = f"Provider: {ev.provider} ({ev.tool_name})"

        source_links = []
        for src in ev.sources:
            if src.url:
                source_links.append(f"[{src.title}]({src.url})")
            else:
                source_links.append(src.title)
        sources_str = f"Sources: {', '.join(source_links)}" if source_links else "Sources: Direct API"

        section = (
            f"### {kind_tag} {cap_name} ({provider_info})\n"
            f"- {status_tag} | {sources_str}\n"
            f"{ev.summary.strip()}\n"
        )
        sections.append(section)

    return "\n\n".join(sections)
