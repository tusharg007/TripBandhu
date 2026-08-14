# TripBandhu — TF2: MCP Capability Contracts, Evidence & Tool Provenance

This document details the MCP capability layer, tool provenance, evidence normalization, schema drift safety, and execution contracts implemented in **TripBandhu TF2**.

---

## 1. Architectural Philosophy: MCP as Transport

MCP (Model Context Protocol) is treated strictly as **transport**, never as an ungrounded or unchecked source of truth.

```text
LangGraph Specialist
       ↓
Capability Client / Registry Validation
       ↓
MCP Session & Tool Execution (Pinned Version)
       ↓
External Provider Response
       ↓
Normalized Domain Evidence (with Source Provenance)
       ↓
Bounded & Labeled Grounding Context
       ↓
Downstream Graph State & Synthesis
```

- Raw MCP envelopes are never passed directly to synthesis prompts.
- Specialists are forbidden from invoking arbitrary or model-invented tool names.
- All retrieved data is tagged with its provenance category (`LIVE_PROVIDER`, `WEB_SOURCE`, `MODEL_ESTIMATE`, `DERIVED`, `UNAVAILABLE`).

---

## 2. MCP Server Inventory & Transports

| Server Name | Transport | Invocation / Endpoint | Authentication Source | Pinned Version | Lifecycle & Concurrency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AviationStack MCP** | `stdio` | `uvx aviationstack-mcp==1.6.0` | `AVIATION_STACK_API_KEY` | `aviationstack-mcp==1.6.0` | Reusable Async MultiServer MCP client |
| **Tavily MCP** | `streamable_http` | `https://mcp.tavily.com/mcp/?tavilyApiKey=...` | `TAVILY_API_KEY` | Managed SaaS endpoint | Reusable Async HTTP streamable session |
| **Weather MCP** | `stdio` | `python custom_weather_mcp_server.py` | `OPENWEATHER_API_KEY` | Local custom FastMCP | Reusable Subprocess stdio transport |

---

## 3. Canonical Capability Registry & Permissions

TripBandhu maintains a canonical typed registry in [`capability_registry.py`](../capability_registry.py):

| Capability Name | MCP Server | Underlying MCP Tool | Evidence Kind | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `FLIGHT_ROUTE_SEARCH` | `aviationstack` | `list_routes` | `LIVE_PROVIDER` | Search flight routes between departure/arrival IATA codes |
| `AIRPORT_LOOKUP` | `aviationstack` | `list_airports` | `LIVE_PROVIDER` | Lookup airport metadata and IATA codes |
| `AIRLINE_LOOKUP` | `aviationstack` | `list_airlines` | `LIVE_PROVIDER` | Lookup airline names and IATA codes |
| `HOTEL_WEB_RESEARCH` | `tavily` | `tavily_search` | `WEB_SOURCE` | Search web for accommodation, neighborhoods, reviews |
| `WEATHER_CURRENT` | `weather` | `get_current_weather` | `LIVE_PROVIDER` | Real-time weather observation |
| `WEATHER_FORECAST` | `weather` | `get_forecast` | `LIVE_PROVIDER` | Multi-day meteorological forecast |

### Specialist Permission Map

Specialists are strictly constrained to their approved capability set:
- **`flight_agent`**: `["FLIGHT_ROUTE_SEARCH", "AIRPORT_LOOKUP", "AIRLINE_LOOKUP"]`
- **`hotel_agent`**: `["HOTEL_WEB_RESEARCH"]`
- **`weather_agent`**: `["WEATHER_CURRENT", "WEATHER_FORECAST"]`

Any attempt to call an unauthorized or arbitrary tool is rejected with `CONTRACT_MISMATCH` / `UNAVAILABLE`.

---

## 4. Route-Specific Flight Capability & Pricing Truth

### Aviation Tool Selection: `list_routes`
In `aviationstack-mcp==1.6.0`, the primary flight discovery tool is `list_routes` (accepting `dep_iata`, `arr_iata`, and `airline_iata` filters), complemented by `list_airports` and `list_airlines`. Reference-data-only tools are no longer the sole source of flight data.

### Flight Pricing Truth
- **AviationStack is operational/schedule data, not a live ticket pricing source.**
- When flight prices or budget ranges are estimated by the model, they are strictly marked `[MODEL ESTIMATE - Not Live Fare]`.
- Prompts explicitly instruct agents never to claim that live booking airfares have been verified without a live ticketing provider.

---

## 5. Normalized Evidence Models

Defined in [`schemas.py`](../schemas.py):

### `EvidenceKind`
- `LIVE_PROVIDER`: Real-time operational data from verified APIs (e.g. OpenWeather).
- `WEB_SOURCE`: Search engine results (e.g. Tavily search).
- `MODEL_ESTIMATE`: Model-generated planning ranges (budget allowances).
- `DERIVED`: Computed from verified evidence.
- `UNAVAILABLE`: Provider failed or degraded gracefully.

### `SourceReference`
Preserves source metadata for full traceability:
- `provider`: Underlying provider name (`"Tavily"`, `"OpenWeather"`, `"AviationStack"`)
- `title`: Result title or entity descriptor
- `url`: Direct source link (when exposed by provider)
- `observed_at`: ISO timestamp of observation
- `evidence_kind`: Provenance category

### `CapabilityEvidence`
- `capability`: Logical capability identifier
- `provider`: Underlying provider
- `tool_name`: MCP tool name
- `status`: `AVAILABLE` | `DEGRADED` | `UNAVAILABLE` | `CONTRACT_MISMATCH`
- `retrieved_at`: ISO timestamp
- `latency_ms`: Execution latency
- `data`: Structured normalized payload
- `summary`: Human/LLM-readable concise summary
- `sources`: List of `SourceReference` objects

---

## 6. Capability Trace & Public Safety Contract

TripBandhu records an append-only `CapabilityTraceEntry` per invocation:
- `sequence`: 1-based order in run
- `specialist`: Calling specialist name
- `capability`: Logical capability name
- `server`: MCP server name
- `tool_name`: Tool invoked
- `status`: Health status (`AVAILABLE`, `DEGRADED`, etc.)
- `latency_ms`: Invocation latency
- `retry_count`: Transient retries performed
- `started_at` / `completed_at`: Timestamps
- `source_count`: Number of sources preserved

### Security & Sanitization
The public trace contract (`CapabilityTraceEntry.to_public_dict()`) exposes only high-level metadata. It **strictly excludes**:
- API keys, credentials, and authorization headers
- Raw subprocess environment variables
- `DATABASE_URL`
- Local filesystem paths
- Full internal stack traces

---

## 7. MCP Session Lifecycle, Reconnect & Cancellation

- **Reusable Client**: Managed through `MultiServerMCPClient` attached to application state.
- **Cancellation Safety**: `provider_utils.py` uses `asyncio.CancelledError` pass-through, ensuring task cancellation propagates immediately without being converted into generic provider failures.
- **Schema Drift Detection**: `validate_mcp_capability_health()` verifies registered tools against runtime server tool lists, assigning `CONTRACT_MISMATCH` if schemas drift or disappear.
- **Bounded Retry**: Only transient network/timeout errors (`TIMEOUT`, `RATE_LIMITED`, `UNAVAILABLE`) are retried up to 2 times with backoff; configuration and contract errors fail fast.

---

## 8. Remaining Limitations & Next Steps

1. **Free-Tier Constraints**: The free AviationStack API tier restricts certain route queries; the graph degrades gracefully with clear user messaging.
2. **Deterministic Context Packing**: Evidence context is bounded to prevent token bloat during itinerary revision turns.
3. **Prerequisites for TF3**: Fully frozen agent graph, structured evidence layer, and deterministic test suite ready for benchmark evaluation.
