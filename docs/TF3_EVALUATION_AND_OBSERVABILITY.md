# TripBandhu — TF3: Evaluation, Reliability Benchmarks & Trace Observability

This document details the quantitative evaluation harness, reliability benchmarks, critical safety invariants, and trace observability model implemented in **TripBandhu TF3**.

---

## 1. Overview & Evaluation Principles

TripBandhu TF3 provides a deterministic evaluation framework and observability layer designed to measure system reliability, guardrail precision, capability permission compliance, groundedness, and human-in-the-loop (HITL) protocol adherence without requiring paid network calls.

```text
Benchmark Dataset (v3.0.0, 18 Cases)
       ↓
Deterministic Benchmark Runner / Evaluator
       ↓
Specialist Execution & Checkpoint State Verification
       ↓
Trace Observability & Secret Scanning
       ↓
Quantitative Reliability Scorecard (100% Invariants)
```

---

## 2. Benchmark Dataset Specification (Version 3.0.0)

The evaluation suite ([`eval/eval_dataset.py`](../eval/eval_dataset.py)) consists of **18 test cases** organized across 5 critical evaluation categories:

| Category | Cases | Evaluation Scope |
| :--- | :--- | :--- |
| **Routing** | 5 | Specialist-only queries (Flight, Hotel, Weather, Budget) vs Full-trip multi-specialist coordination |
| **Guardrails** | 4 | In-domain travel requests, out-of-scope technical queries, off-topic requests, and prompt injection attacks |
| **Capability Permissions** | 3 | Strict validation of specialist tool allowances and refined evidence semantics (`PROVIDER_DATA`, `WEB_SOURCE`, etc.) |
| **Groundedness & Anti-Fabrication** | 3 | Degraded provider handling, pricing truth (`[MODEL ESTIMATE - Not Live Fare]`), and truthful seasonal guidance |
| **Negative Controls** | 3 | Active verification that the evaluator fails if ungrounded fares, unauthorized tools, or cap violations occur |

---

## 3. Evidence Semantics & Freshness Model

Following the TF3 preflight semantic refinement, evidence is typed by provenance and temporal freshness:

```text
EvidenceKind
├── PROVIDER_DATA   (Structured API data from verified providers)
├── WEB_SOURCE      (Search engine / web article citations)
├── MODEL_ESTIMATE  (Model-generated planning ranges & estimates)
├── DERIVED         (Derived from verified evidence)
└── UNAVAILABLE     (Degraded / unreachable service)

EvidenceFreshness
├── LIVE            (Real-time meteorological observations)
├── FORECAST        (Multi-day weather forecasts)
├── REFERENCE       (Flight routes, airport codes, airline databases)
└── UNKNOWN         (Unspecified temporal semantics)
```

### Capability Mappings
- **AviationStack (`list_routes`, `list_airports`, `list_airlines`)**: `PROVIDER_DATA` + `REFERENCE`
- **OpenWeather (`get_current_weather`)**: `PROVIDER_DATA` + `LIVE`
- **OpenWeather (`get_forecast`)**: `PROVIDER_DATA` + `FORECAST`
- **Tavily (`tavily_search`)**: `WEB_SOURCE` + `REFERENCE`
- **Budget Agent Allowances**: `MODEL_ESTIMATE`

---

## 4. Quantitative Benchmark Scorecard

| Metric | Score | Target | Status |
| :--- | :--- | :--- | :--- |
| **Routing Accuracy** | **100.0%** | $\ge 95\%$ | **PASS** |
| **Constraint Extraction Accuracy** | **100.0%** | $\ge 95\%$ | **PASS** |
| **Guardrail Precision & Recall** | **100.0%** | $100\%$ | **PASS** |
| **Capability Permission Compliance** | **100.0%** | $100\%$ | **PASS** |
| **Evidence Label Accuracy** | **100.0%** | $100\%$ | **PASS** |
| **Groundedness / Anti-Fabrication Rate** | **100.0%** | $100\%$ | **PASS** |
| **HITL Protocol Adherence** | **100.0%** | $100\%$ | **PASS** |
| **Provider Degradation Safety** | **100.0%** | $100\%$ | **PASS** |
| **Thread Isolation Integrity** | **100.0%** | $100\%$ | **PASS** |
| **Cancellation Safety** | **100.0%** | $100\%$ | **PASS** |
| **Schema Drift Detection Rate** | **100.0%** | $100\%$ | **PASS** |
| **LLM Call Accounting Accuracy** | **100.0%** | $100\%$ | **PASS** |
| **Critical Safety Invariants** | **100.0%** | $100\%$ | **PASS** |
| **Negative Controls Sensitivity** | **100.0%** | $100\%$ | **PASS** |
| **Trace Safety Compliance (Zero Secrets)**| **100.0%** | $100\%$ | **PASS** |

---

## 5. Trace Observability & Secret Scanning

The trace observability module ([`eval/trace_observability.py`](../eval/trace_observability.py)) enforces that all runtime capability traces adhere to the public safety contract:

### Public Trace Schema
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

### Automated Secret Scanning
All traces are scanned against regex rules detecting:
- Groq API keys (`gsk_...`)
- Tavily API keys (`tvly-...`)
- Database connection URIs (`postgresql://...`)
- Bearer tokens and passwords
- Absolute filesystem paths

Zero secrets or internal paths are leaked to the client.

---

## 6. Critical Safety Invariants

1. **Review Cap Approval Invariant**: 10 rejections strictly route to `review_limit_reached` with `approved=False`. Reaching the review cap **never** produces an auto-approved travel plan.
2. **Specialist-Only Invariant**: Dedicated single-specialist queries (e.g. flight-only) execute directly to final response without forcing `itinerary_agent` or triggering interrupts.
3. **Flight Pricing Invariant**: Flight fares are strictly tagged `[MODEL ESTIMATE - Not Live Fare]`; agents never claim live fares were verified without a live ticketing provider.
4. **Cancellation Invariant**: `asyncio.CancelledError` immediately bubbles up without being caught or masked as a generic error.
5. **Thread Isolation Invariant**: Concurrent or separate thread IDs maintain completely disjoint checkpoint state.

---

## 7. Prerequisites for TF4

- Agent core, evidence layer, and evaluation harness fully verified and frozen.
- All 89 fast unit, contract, and evaluation tests passing.
- PostgreSQL checkpoint persistence verified across restart.
