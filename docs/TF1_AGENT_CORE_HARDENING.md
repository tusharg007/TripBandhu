# TripBandhu TF1 — Agent Core Hardening

## Summary

TF1 hardens the agent execution core of TripBandhu across four primary goals with strict architectural correctness:

1. **Eliminate `nest_asyncio` / nested event-loop execution** — all graph nodes and provider adapters are native `async def` functions.
2. **Structured and validated supervisor + constraint extraction** — Pydantic models with `with_structured_output`, preserving dynamic specialist routing.
3. **True iterative HITL review loop with defensive cap** — users can request multiple revisions; reaching the cap **never auto-finalizes or auto-approves**.
4. **Explicit provider/tool timeout and failure semantics** — centralized timeouts, retry classification, and precise LLM call accounting.

---

## Architectural Decisions & Invariants

### 1. Application-Scoped Async Postgres / Graph Lifecycle
- FastAPI `lifespan` manages the `AsyncPostgresSaver` context manager.
- `create_travel_service(checkpointer)` builds and compiles the LangGraph instance bound to the active checkpointer and stores `TravelAgentService` on `app.state.travel_service`.
- On shutdown, `lifespan` cleanly exits the context manager and closes the database connection.
- No import-time permanent database connections.
- Graph construction (`build_travel_graph`) is cleanly separated from execution, allowing tests to run with `InMemorySaver` without database dependencies.

### 2. Dynamic Specialist Routing (No Forced Itinerary)
- The supervisor dynamically selects only the specialists required for the query (`FULL_TRIP`, `FLIGHT_ONLY`, `HOTEL_ONLY`, `WEATHER_ONLY`, `BUDGET_ONLY`, `ITINERARY_ONLY`).
- Invariant: `itinerary_agent` is NOT unconditionally appended.
- For specialist-only queries (e.g. flight-only or hotel-only):
  - Relevant specialist(s) execute.
  - `route_after_agent` routes directly to `final_agent` to format the specialist summary.
  - No draft itinerary is created, and execution does NOT pause for human review of an unrequested itinerary.
- When `itinerary_agent` is selected:
  - Draft itinerary is generated and execution pauses at `human_review_node` (`interrupt`).

### 3. Defensive Review Cap Semantics (Never Auto-Finalize)
- Reaching `MAX_REVIEW_ITERATIONS` (default: 10 rejections) routes to `review_limit_reached_agent`.
- **Never auto-finalizes or auto-approves**.
- Returns a stable state with `approved: False`, `review_limit_reached: True`, and a safe user message advising to start a new planning run.
- Review history and checkpoints remain preserved.

### 4. Precise LLM Call Accounting
- `llm_calls` counts **actual model invocations**:
  - Supervisor structured output (+1)
  - Model-backed input guardrail (+1)
  - Destination extraction (+1)
  - Specialist prompts for flight, budget, itinerary, itinerary_revision, and final response (+1 each on actual invocation)
- Excluded from `llm_calls`:
  - Tavily-only searches (`hotel_agent` makes 0 LLM calls)
  - MCP subprocess tool calls
  - Deterministic keyword fallbacks (0 LLM calls)
  - Routing and serialization

### 5. Lightweight Capability Result & Anti-Fabrication
- `CapabilityResult[T]` wraps provider calls cleanly with timeout and error classification.
- Error codes: `TIMEOUT`, `RATE_LIMITED`, `UNAVAILABLE`, `INVALID_RESPONSE`, `AUTH_CONFIGURATION`, `INTERNAL`.
- Retries are bounded to max 2 attempts for transient errors (`TIMEOUT`, `RATE_LIMITED`, `UNAVAILABLE`). Auth/configuration errors are NEVER retried.
- Strict anti-fabrication prompts: when specialist data is marked unavailable/degraded, exact prices or schedules are never hallucinated.

---

## File Manifest

| File | Purpose |
|------|---------|
| `schemas.py` | Pydantic models: `AgentName`, `SupervisorDecision`, `TravelConstraints`, `GuardrailDecision`, `ReviewDecision`, `CapabilityResult`, `ErrorCode` |
| `agent_config.py` | Centralized timeout constants and retry configuration |
| `provider_utils.py` | `async_call_provider` utility wrapping `asyncio.timeout` and bounded retry |
| `backend.py` | Async LangGraph state graph, specialist agents, iterative HITL review loop, `TravelAgentService` |
| `app.py` | FastAPI application with lifespan-scoped `AsyncPostgresSaver`, API contract endpoints |
| `mcp_client.py` | Async MCP adapters including `extract_destination_async` |

---

## Test Manifest (76 Tests, 100% Pass Rate)

| Test File | Description |
|-----------|-------------|
| `test_structured_supervisor.py` | Dynamic routing (flight-only, hotel-only, weather-only, budget-only, full-trip), AgentName validation, fallback |
| `test_structured_constraints.py` | Typed constraint extraction with partial/missing fields |
| `test_guardrail.py` | Allow/block behavior, deterministic travel-keyword fallback |
| `test_async_execution.py` | Static check for no `nest_asyncio`, no `asyncio.run` in nodes, async function definitions, lifespan validation |
| `test_hitl_loop.py` | Approval routing, revision routing, review cap routing, API validation (feedback required for revision) |
| `test_hitl_multi_turn_e2e.py` | Full multi-turn revision loop (v0 -> v1 -> v2 -> approved), 10-rejection cap without approval |
| `test_specialist_only_e2e.py` | End-to-end flight-only and hotel-only queries without itinerary or interrupt |
| `test_llm_accounting.py` | Exact LLM call counts for all nodes, fallback accounting, hotel_agent 0-call check |
| `test_provider_failure.py` | Classification, no retry on auth errors, degraded specialist status, anti-fabrication |
| `test_concurrency.py` | Thread isolation, independent state across threads |
| `test_api_contract.py` | Preserved TF0 public contract on `/api/travel` and `/api/travel/resume` |
| `test_truthfulness_and_degradation.py` | TF0.5 regression tests updated for async runtime |
| `test_release_safety.py` | Release safety, Docker configuration, and template checks |
