"""
capability_registry.py — MCP Capability Registry, Normalization & Provenance Layer.

Defines approved capability definitions, specialist permission maps, deterministic
output normalizers, and grounding context packagers.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Optional

from schemas import (
    CapabilityDefinition,
    CapabilityEvidence,
    CapabilityHealth,
    EvidenceKind,
    EvidenceFreshness,
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
        evidence_kind=EvidenceKind.PROVIDER_DATA,
        freshness=EvidenceFreshness.REFERENCE,
        description="Search commercial flight routes between departure and arrival IATA airport codes.",
        required_params=[],
    ),
    "AIRPORT_LOOKUP": CapabilityDefinition(
        capability_name="AIRPORT_LOOKUP",
        server="aviationstack",
        tool_name="list_airports",
        access_mode="read_only",
        evidence_kind=EvidenceKind.PROVIDER_DATA,
        freshness=EvidenceFreshness.REFERENCE,
        description="Lookup airport metadata and IATA codes.",
        required_params=[],
    ),
    "AIRLINE_LOOKUP": CapabilityDefinition(
        capability_name="AIRLINE_LOOKUP",
        server="aviationstack",
        tool_name="list_airlines",
        access_mode="read_only",
        evidence_kind=EvidenceKind.PROVIDER_DATA,
        freshness=EvidenceFreshness.REFERENCE,
        description="Lookup airline names and IATA codes.",
        required_params=[],
    ),
    "HOTEL_WEB_RESEARCH": CapabilityDefinition(
        capability_name="HOTEL_WEB_RESEARCH",
        server="tavily",
        tool_name="tavily_search",
        access_mode="read_only",
        evidence_kind=EvidenceKind.WEB_SOURCE,
        freshness=EvidenceFreshness.REFERENCE,
        description="Web search for hotels, accommodations, neighborhoods, and reviews.",
        required_params=["query"],
    ),
    "WEATHER_CURRENT": CapabilityDefinition(
        capability_name="WEATHER_CURRENT",
        server="weather",
        tool_name="get_current_weather",
        access_mode="read_only",
        evidence_kind=EvidenceKind.PROVIDER_DATA,
        freshness=EvidenceFreshness.LIVE,
        description="Real-time current meteorological observations for a destination city.",
        required_params=["city"],
    ),
    "WEATHER_FORECAST": CapabilityDefinition(
        capability_name="WEATHER_FORECAST",
        server="weather",
        tool_name="get_forecast",
        access_mode="read_only",
        evidence_kind=EvidenceKind.PROVIDER_DATA,
        freshness=EvidenceFreshness.FORECAST,
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

def _unwrap_mcp(raw: Any) -> Any:
    """Strip MCP protocol envelope [{type:'text', text:'...JSON...'}] → plain data.

    langchain_mcp_adapters returns tool results wrapped in the MCP content-block
    format. This helper unwraps that envelope and JSON-parses the inner text so
    downstream normalizers always receive plain dicts / lists / strings.
    """
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and raw[0].get("type") == "text":
        combined = " ".join(item.get("text", "") for item in raw if isinstance(item, dict))
        try:
            return json.loads(combined)
        except (json.JSONDecodeError, TypeError):
            return combined  # fallback: plain string
    return raw  # already unwrapped — pass through unchanged


_HOTEL_TERMS = frozenset({
    "hotel", "hotels", "resort", "hostel", "accommodation", "lodging",
    "guesthouse", "guest house", "homestay", "room", "rooms", "suite",
    "suites", "inn", "stay", "stays", "neighborhood", "neighbourhood",
    "booking", "property",
})
_HOTEL_NOISE_TERMS = frozenset({
    "best buy", "dictionary", "definition", "merriam-webster", "cambridge",
    "google play", "app store", "weatherapi", "openweather", "word of the day",
})


def _clean_web_text(value: Any) -> str:
    """Reduce scraped-page noise while retaining useful prose."""
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _hotel_relevance_score(title: str, url: str, content: str) -> int:
    haystack = f"{title} {url} {content}".casefold()
    if any(term in haystack for term in _HOTEL_NOISE_TERMS):
        return -100
    return sum(2 if term in title.casefold() else 1 for term in _HOTEL_TERMS if term in haystack)


def _number(value: Any, suffix: str = "") -> str:
    if value is None or value == "":
        return "Not reported"
    return f"{value}{suffix}"


def _usable_weather_payload(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and not value.get("error")


def _select_provider_records(value: Any, max_records: int = 30) -> Any:
    """Select complete provider records without cutting any record's text midway."""
    unwrapped = _unwrap_mcp(value)
    if isinstance(unwrapped, list):
        return unwrapped[:max_records]
    if isinstance(unwrapped, dict):
        records = unwrapped.get("data")
        if isinstance(records, list):
            selected = dict(unwrapped)
            selected["data"] = records[:max_records]
            selected["records_returned"] = len(records)
            selected["records_selected_for_route_summary"] = min(len(records), max_records)
            return selected
    return unwrapped


def normalize_tavily_results(raw_data: Any, query: str, latency_ms: Optional[int] = None) -> CapabilityEvidence:
    """Keep hotel-relevant Tavily evidence and discard unrelated search noise."""
    now_iso = datetime.now(timezone.utc).isoformat()
    sources: list[SourceReference] = []
    items: list[dict[str, Any]] = []

    raw_data = _unwrap_mcp(raw_data)  # strip MCP {type, text} envelope if present

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

    ranked_results: list[tuple[int, int, dict[str, str]]] = []
    for index, item in enumerate(raw_results):
        if isinstance(item, dict):
            title = _clean_web_text(item.get("title") or item.get("name") or "Accommodation result")
            url = str(item.get("url") or "").strip()
            content = _clean_web_text(item.get("content") or item.get("snippet") or item.get("body"))
        else:
            title = "Accommodation result"
            url = ""
            content = _clean_web_text(item)

        score = _hotel_relevance_score(title, url, content)
        if score > 0:
            ranked_results.append((score, index, {"title": title, "url": url, "content": content}))

    # Six relevant sources are enough for synthesis; ranking prevents unrelated high-
    # position results from crowding actual accommodation sources out of the prompt.
    bounded_results = [entry[2] for entry in sorted(ranked_results, key=lambda value: (-value[0], value[1]))[:6]]
    summary_lines = []

    for item in bounded_results:
        title = item["title"]
        url = item["url"]
        content = item["content"]

        sources.append(
            SourceReference(
                provider="Tavily Web Search",
                title=title,
                url=url if url else None,
                observed_at=now_iso,
                evidence_kind=EvidenceKind.WEB_SOURCE,
                freshness=EvidenceFreshness.REFERENCE,
            )
        )
        # A snippet is deliberately bounded at the evidence boundary so full scraped
        # pages never reach an LLM or the browser. This does not limit generated output.
        snippet = content[:700].rstrip()
        items.append({"title": title, "url": url, "snippet": snippet})
        url_md = f" ([Source]({url}))" if url else ""
        summary_lines.append(f"- **{title}**{url_md}: {snippet or 'Accommodation source found; details require verification.'}")

    formatted_summary = "\n".join(summary_lines) if summary_lines else (
        "No reliable hotel-specific sources were found in the search results. "
        "Try a more specific destination or check a trusted accommodation platform directly."
    )

    return CapabilityEvidence(
        capability="HOTEL_WEB_RESEARCH",
        provider="tavily",
        tool_name="tavily_search",
        evidence_kind=EvidenceKind.WEB_SOURCE,
        freshness=EvidenceFreshness.REFERENCE,
        status=CapabilityHealth.AVAILABLE if items else CapabilityHealth.DEGRADED,
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

    # Unwrap MCP protocol envelope on both inputs
    current_data = _unwrap_mcp(current_data)
    forecast_data = _unwrap_mcp(forecast_data)

    sources = [
        SourceReference(
            provider="OpenWeather API",
            title=f"Live Weather & Forecast Observation for {city}",
            url="https://openweathermap.org",
            observed_at=now_iso,
            evidence_kind=EvidenceKind.PROVIDER_DATA,
            freshness=EvidenceFreshness.LIVE,
        )
    ]

    current_ok = _usable_weather_payload(current_data)
    forecast_ok = _usable_weather_payload(forecast_data)
    safe_current = current_data if current_ok else None
    safe_forecast = forecast_data if forecast_ok else None
    structured_data = {
        "city": city,
        "current": safe_current,
        "forecast": safe_forecast,
    }

    sections: list[str] = []
    if current_ok:
        reported_city = current_data.get("city") or city
        sections.append(
            f"## Current weather in {reported_city}\n"
            f"- **Conditions:** {current_data.get('condition') or 'Not reported'}\n"
            f"- **Temperature:** {_number(current_data.get('temperature_c'), '°C')}"
            + (
                f" (feels like {_number(current_data.get('feels_like_c'), '°C')})"
                if current_data.get("feels_like_c") is not None else ""
            )
            + f"\n- **Humidity:** {_number(current_data.get('humidity'), '%')}"
            + f"\n- **Wind:** {_number(current_data.get('wind_speed'), ' m/s')}"
        )
    else:
        sections.append(
            f"## Current weather in {city}\n"
            "Live conditions are temporarily unavailable. Check a current forecast before departure."
        )

    forecast_entries = forecast_data.get("forecast", []) if forecast_ok else []
    if isinstance(forecast_entries, list) and forecast_entries:
        rows = []
        for entry in forecast_entries:
            if not isinstance(entry, dict):
                continue
            rows.append(
                f"- **{entry.get('datetime') or 'Upcoming'}:** "
                f"{entry.get('condition') or 'Conditions not reported'}, "
                f"{_number(entry.get('temperature_c', entry.get('temp')), '°C')}"
            )
        if rows:
            sections.append("## Near-term forecast\n" + "\n".join(rows))
    if not forecast_entries:
        sections.append(
            "## Near-term forecast\n"
            "A provider forecast is temporarily unavailable. Recheck closer to the travel date."
        )

    summary = "\n\n".join(sections)

    return CapabilityEvidence(
        capability="WEATHER_FORECAST",
        provider="weather",
        tool_name="get_current_weather,get_forecast",
        evidence_kind=EvidenceKind.PROVIDER_DATA,
        freshness=EvidenceFreshness.FORECAST,
        status=CapabilityHealth.AVAILABLE if current_ok and forecast_ok else CapabilityHealth.DEGRADED,
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
    """Normalize AviationStack route and airport data into typed reference evidence."""
    now_iso = datetime.now(timezone.utc).isoformat()
    sources = [
        SourceReference(
            provider="AviationStack Schedule & Route Database",
            title=f"Flight Routes & Airports for {origin or 'Origin'} -> {destination}",
            url="https://aviationstack.com",
            observed_at=now_iso,
            evidence_kind=EvidenceKind.PROVIDER_DATA,
            freshness=EvidenceFreshness.REFERENCE,
        )
    ]

    selected_routes = _select_provider_records(routes_data)
    selected_airports = _select_provider_records(airports_data)
    selected_airlines = _select_provider_records(airlines_data)
    structured_data = {
        "destination": destination,
        "origin": origin,
        "routes": selected_routes,
        "airports": selected_airports,
        "airlines": selected_airlines,
    }

    # Do not character-slice provider evidence: that can cut a flight record halfway
    # through and subsequently cause a generated specialist answer to stop abruptly.
    routes_str = json.dumps(selected_routes, ensure_ascii=False, default=str) if routes_data else "No specific routes returned."
    airports_str = json.dumps(selected_airports, ensure_ascii=False, default=str) if airports_data else "No airport records returned."
    airlines_str = json.dumps(selected_airlines, ensure_ascii=False, default=str) if airlines_data else "No airline records returned."

    summary = (
        f"**Route Information ({origin or 'Origin'} -> {destination}) [REFERENCE]:**\n{routes_str}\n\n"
        f"**Serving Airports [REFERENCE]:**\n{airports_str}\n\n"
        f"**Operating Airlines [REFERENCE]:**\n{airlines_str}"
    )

    return CapabilityEvidence(
        capability="FLIGHT_ROUTE_SEARCH",
        provider="aviationstack",
        tool_name="list_routes,list_airports,list_airlines",
        evidence_kind=EvidenceKind.PROVIDER_DATA,
        freshness=EvidenceFreshness.REFERENCE,
        status=CapabilityHealth.AVAILABLE,
        retrieved_at=now_iso,
        latency_ms=latency_ms,
        data=structured_data,
        summary=summary,
        sources=sources,
    )


# ===========================================================================
# 4. Grounding Context Packager (Evidence Kind & Freshness Tagging)
# ===========================================================================

def format_grounded_evidence_context(evidence_store: dict[str, dict]) -> str:
    """Pack normalized evidence into a deterministic, clearly labeled prompt context.

    Explicitly tags each section with [PROVIDER_DATA - REFERENCE], [PROVIDER_DATA - LIVE],
    [WEB_SOURCE], [MODEL_ESTIMATE], or [UNAVAILABLE] to maintain truthful grounding.
    """
    if not evidence_store:
        return "No external provider evidence was retrieved for this run."

    sections: list[str] = []

    for cap_name, ev_dict in evidence_store.items():
        ev = CapabilityEvidence(**ev_dict) if isinstance(ev_dict, dict) else ev_dict
        freshness_tag = f" - {ev.freshness.value}" if ev.freshness != EvidenceFreshness.UNKNOWN else ""
        kind_tag = f"[{ev.evidence_kind.value}{freshness_tag}]"
        status_tag = f"Status: {ev.status.value}"
        provider_info = f"Provider: {ev.provider} ({ev.tool_name})"

        source_links = []
        for src in ev.sources:
            if src.url:
                source_links.append(f"[{src.title}]({src.url})")
            else:
                source_links.append(src.title)
        sources_str = f"Sources: {', '.join(source_links)}" if source_links else "Sources: Direct Provider API"

        section = (
            f"### {kind_tag} {cap_name} ({provider_info})\n"
            f"- {status_tag} | {sources_str}\n"
            f"{ev.summary.strip()}\n"
        )
        sections.append(section)

    return "\n\n".join(sections)
