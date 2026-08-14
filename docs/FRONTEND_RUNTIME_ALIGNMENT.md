# TripBandhu TF0 Frontend Runtime Alignment

## Authoritative Backend

The local backend is the authoritative implementation for this phase. It preserves the current supervisor, guardrail, flight, hotel, weather, budget, itinerary, PostgreSQL checkpointing, MCP client, custom weather MCP server, and LangGraph `interrupt()` / `Command(resume=...)` human-review flow.

TF0 does not replace the agent graph or redesign the backend routing model.

## Path and Configuration Behavior

Project-relative configuration now derives from `Path(__file__).resolve()` through `project_config.py`.

`load_project_env()` loads the repository `.env` with `override=False`, so local development can read project secrets while production-provided environment variables continue to win.

No Windows drive, username, Conda path, or Python executable path is hardcoded. The weather MCP server is launched through the active Python executable, and the AviationStack MCP server uses `uvx` from the runtime path.

## API Contract

`POST /api/travel` starts a planning run and returns the normalized public contract:

- `thread_id`
- `answer`
- `requires_approval`
- `approval_request`
- `flight_results`
- `hotel_results`
- `weather_results`
- `budget_results`
- `itinerary`
- `selected_agents`
- `trip_constraints`
- `supervisor_reasoning`
- `guardrail_allowed`
- `guardrail_reason`
- `approved`
- `human_feedback`
- `llm_calls`

`POST /api/travel/resume` accepts `thread_id`, `approved`, and `feedback`, delegates to `resume_travel_agent()`, and returns the same normalized public contract.

Unexpected API exceptions are logged server-side and returned as stable public errors such as `TRAVEL_PLANNING_FAILED` and `RESUME_FAILED`, without stack traces, local paths, database URLs, or provider credentials.

## Browser HITL Flow

The browser submits the initial request to `/api/travel`. When `requires_approval` is true, the UI presents the draft itinerary as a review item rather than a final answer.

Approving sends:

```json
{
  "thread_id": "...",
  "approved": true,
  "feedback": ""
}
```

Requesting a revision sends:

```json
{
  "thread_id": "...",
  "approved": false,
  "feedback": "..."
}
```

Both actions call `/api/travel/resume` and render the normalized resumed response. TF0 exposes the current backend semantics exactly; a repeated review loop remains TF1 work.

## Frontend Alignment

The FastAPI/Jinja/HTML/CSS/vanilla JavaScript interface now centers the real agentic workflow:

- Trip request with quick prompts and New Trip control
- Execution panel derived from `selected_agents` and response fields
- Supervisor decision with guardrail status, constraints, and reasoning
- Specialist outputs for flights, hotels, weather, budget, and draft itinerary
- Human review panel for approval or revision feedback
- Final response with copy and PDF export after resume

The New Trip control clears the stored `thread_id`, visible agent state, old errors, review panels, and final output. Refresh still preserves an active thread unless the user explicitly starts a new trip.

Markdown output is rendered through pinned `marked` and sanitized through pinned `DOMPurify` before insertion into the page.

## Docker Runtime Requirements

The Docker image installs pinned `uv` and verifies `uvx --version` during build because the AviationStack MCP client launches `uvx aviationstack-mcp`.

The Docker build context excludes `.env`, `.env.*`, Git metadata, virtual environments, caches, compiled Python files, logs, local databases, and runtime artifacts while allowing `.env.example`.

## TF1 Status — RESOLVED in `feat/agentic-core-hardening`

TF1 has been implemented:

- [x] Clean async execution model (no `nest_asyncio`, no `asyncio.run` in graph nodes)
- [x] `AsyncPostgresSaver` via FastAPI lifespan context manager
- [x] Structured supervisor output via `SupervisorDecision` Pydantic model
- [x] Structured guardrail via `GuardrailDecision` Pydantic model
- [x] Structured `TravelConstraints` extraction
- [x] `AgentName` enum validation on supervisor output
- [x] Iterative HITL review loop with revision history and defensive cap
- [x] Centralized provider timeouts (`asyncio.timeout`)
- [x] Exception classification and bounded retry via `provider_utils.async_call_provider`
- [x] LLM call accounting fix (hotel_agent was over-counting)
- [x] `extract_destination_async` LLM call now counted

## TF2 Debt

- Streaming SSE for real-time specialist progress
- Evaluation framework for LLM output quality
- Observability / OpenTelemetry tracing
- Rate-limit awareness and exponential backoff for Groq API
- Multi-destination / multi-segment trip planning
