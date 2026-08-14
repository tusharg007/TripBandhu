# TripBandhu — TF3 / TF3.1: Evaluation, Reliability Benchmarks & Trace Observability

This document details the quantitative evaluation harness, reliability benchmarks, critical safety invariants, and trace observability model implemented in **TripBandhu TF3 / TF3.1**.

---

## 1. Separation of the Three Evidence Layers

To prevent ambiguity or conflated metric claims, TripBandhu strictly separates its verification into three distinct, complementary layers:

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                   TRIPBANDHU THREE-TIER EVIDENCE MODEL                   │
├──────────────────────────────────────────────────────────────────────────┤
│ Layer A: VERSIONED EVALUATION DATASET (v3.1.0, 50 Scenarios)             │
│   - Behavioral contract benchmarks evaluated with real numerators/denoms │
│   - Tests routing, constraints, guardrails, capabilities, HITL, drift    │
├──────────────────────────────────────────────────────────────────────────┤
│ Layer B: PYTEST REGRESSION SUITE (88 Fast Unit / Contract Tests)         │
│   - Granular unit assertions on classes, schemas, state graphs, and mocks │
│   - Runs in ~4 seconds offline with 0 paid/network calls                 │
├──────────────────────────────────────────────────────────────────────────┤
│ Layer C: POSTGRES CHECKPOINT INTEGRATION (1 Multi-Turn Persistence Test) │
│   - Real multi-turn LangGraph interrupt/resume persistence               │
│   - Proves state survival across service disposals and process restart   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Benchmark Dataset Specification (Version 3.1.0)

The evaluation suite ([`eval/eval_dataset.py`](../eval/eval_dataset.py)) contains **50 versioned test cases** across **14 distinct categories**:

| Category | Cases | Evaluated Behavioral Contract |
| :--- | :--- | :--- |
| **`routing`** | 6 | Specialist-only routes (Flight, Hotel, Weather, Budget) vs multi-specialist & full trip coordination |
| **`constraints`** | 5 | Structured extraction of destination, origin, duration, budget, travel style, and preferences |
| **`guardrails`** | 5 | In-domain travel, out-of-scope coding/general queries, prompt injection, and embedded tasks |
| **`capability_selection`** | 5 | Approved specialist tool permissions, unknown capability rejection, cross-specialist denial |
| **`evidence_semantics`** | 6 | `PROVIDER_DATA` (`REFERENCE`, `LIVE`, `FORECAST`), `WEB_SOURCE`, `MODEL_ESTIMATE`, `UNAVAILABLE` |
| **`groundedness`** | 4 | Degraded provider handling, pricing truth (`[MODEL ESTIMATE - Not Live Fare]`), zero fake data |
| **`hitl`** | 5 | Multi-turn interrupts, approval routing, revision feedback increment, review cap safe stop |
| **`provider_failure`** | 4 | Classification of `TIMEOUT`, `AUTH_CONFIGURATION`, `UNAVAILABLE`, `INVALID_RESPONSE`; backoff bounds |
| **`schema_drift`** | 2 | Missing or altered MCP tool schema detection as `CONTRACT_MISMATCH` with safe continuation |
| **`cancellation`** | 1 | Immediate `asyncio.CancelledError` pass-through with 0 retries |
| **`llm_accounting`** | 2 | Exact model invocation accounting (`hotel_agent` $= 0$, `extract_destination_async` $= 1$) |
| **`thread_isolation`** | 1 | Thread A (Japan full trip) vs Thread B (Dubai flight-only) maintain disjoint state |
| **`persistence`** | 1 | Multi-turn checkpoint survival across service destruction and recreation |
| **`negative_controls`** | 3 | Evaluator sensitivity testing (catches ungrounded fares, unauthorized tools, auto-approvals) |

---

## 3. Evaluator Modes

1. **`FAST` (Default Offline Mode)**:
   - Evaluates **49 deterministic cases** in $<1$ second with 0 network or database dependencies.
   - Case 47 (`PERSISTENCE_POSTGRES_RESTART`) is marked as `SKIPPED` without inflating the success score.
2. **`POSTGRES` (Integration Mode: `python -m eval.evaluator --postgres`)**:
   - Executes all **50 cases** including the live PostgreSQL multi-turn recreation flow against a real database.
3. **`OPTIONAL LIVE`**:
   - Small non-destructive live MCP discovery and server compatibility check.

---

## 4. Real Metric Denominators & Scorecard (Version 3.1.0)

### Category Breakdown (Numerator / Denominator)

```text
  - routing               : 6/6 (total in category: 6)
  - constraints           : 5/5 (total in category: 5)
  - guardrails            : 5/5 (total in category: 5)
  - capability_selection  : 5/5 (total in category: 5)
  - evidence_semantics    : 6/6 (total in category: 6)
  - groundedness          : 4/4 (total in category: 4)
  - hitl                  : 5/5 (total in category: 5)
  - provider_failure      : 4/4 (total in category: 4)
  - schema_drift          : 2/2 (total in category: 2)
  - cancellation          : 1/1 (total in category: 1)
  - llm_accounting        : 2/2 (total in category: 2)
  - thread_isolation      : 1/1 (total in category: 1)
  - persistence           : 1/1 in POSTGRES mode (0/0 in FAST mode, 1 skipped)
  - negative_controls     : 3/3 (total in category: 3)
```

### Measured Reliability Scores
- **Routing Accuracy**: **100.0%** ($6/6$)
- **Constraint Extraction Accuracy**: **100.0%** ($5/5$)
- **Guardrail Precision & Recall**: **100.0%** ($5/5$)
- **Capability Permission Compliance**: **100.0%** ($5/5$)
- **Evidence Semantics Accuracy**: **100.0%** ($6/6$)
- **Groundedness & Anti-Fabrication Rate**: **100.0%** ($4/4$)
- **HITL Protocol Adherence**: **100.0%** ($5/5$)
- **Provider Resilience**: **100.0%** ($4/4$)
- **Schema Drift Detection**: **100.0%** ($2/2$)
- **Cancellation Safety**: **100.0%** ($1/1$)
- **LLM Accounting Accuracy**: **100.0%** ($2/2$)
- **Thread Isolation Integrity**: **100.0%** ($1/1$)
- **Persistence Pass Rate**: **100.0%** ($1/1$ in POSTGRES mode)
- **Negative Controls Sensitivity**: **100.0%** ($3/3$)
- **Critical Safety Invariants (100% Target)**: **100.0%** ($5/5$ Invariants Maintained)
- **Trace Safety Compliance (Zero Secrets)**: **100.0%** (Clean)

---

## 5. Trace Observability & Secret Scanning

The trace observability module ([`eval/trace_observability.py`](../eval/trace_observability.py)) verifies that all capability trace entries adhere to the public schema:
```json
{
  "specialist": "flight_agent",
  "capability": "FLIGHT_ROUTE_SEARCH",
  "server": "aviationstack",
  "status": "AVAILABLE",
  "latency_ms": 120,
  "source_count": 1,
  "error_code": null
}
```

Traces are automatically scanned with regex filters detecting API keys (`gsk_...`, `tvly-...`), tokens, passwords, database credentials, and local directory paths.

---

## 6. Critical Safety Invariants

1. **Defensive Review Cap**: 10 rejections strictly route to `review_limit_reached` with `approved=False`. Reaching the review cap **never** produces an approved travel plan.
2. **Dynamic Specialist Routing**: Specialist-only queries execute directly to final response without forcing `itinerary_agent` or triggering interrupts.
3. **Flight Pricing Truth**: Flight fares are strictly tagged `[MODEL ESTIMATE - Not Live Fare]`; agents never claim live fares were verified without a live ticketing provider.
4. **Cancellation Safety**: `asyncio.CancelledError` immediately bubbles up without being caught or masked as a generic error.
5. **Thread Isolation**: Concurrent or separate thread IDs maintain completely disjoint checkpoint state.

---

## 7. Prerequisites for TF4

- Agent core, evidence layer, and evaluation harness fully verified and frozen.
- 89 automated tests passing with 100% invariant scores.
- PostgreSQL checkpoint persistence verified across restart.
- Ready for production hardening, Docker build verification, and deployment packaging.
