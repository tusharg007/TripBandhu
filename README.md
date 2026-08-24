# TripBandhu

*A stateful multi-agent travel research system that coordinates specialist agents, live external MCP capabilities, evidence-aware synthesis, and iterative human-in-the-loop review — built on LangGraph, FastAPI, and PostgreSQL.*

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0+-1C3C3C?style=for-the-badge)](https://www.langchain.com/langgraph)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-AsyncPostgresSaver-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Groq](https://img.shields.io/badge/Groq-GPT--OSS%2020B%20%2F%20120B-F55036?style=for-the-badge)](https://console.groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-F4B942?style=for-the-badge)](LICENSE)

> **Latest release:** `v1.0.2` — GPT-OSS Migration & Runtime Reliability
> **Live demo:** [https://tripbandhu.onrender.com](https://tripbandhu.onrender.com) *(free tier — cold starts; local run is recommended)*

---

## 📸 System in Action

| **1. Agent Execution & Supervisor Routing** | **2. Interactive Human-in-the-Loop Review** |
| :---: | :---: |
| ![Agent Panel](docs/assets/portfolio/agent-supervisor.png) | ![Human Review Gate](docs/assets/portfolio/human-review.png) |
| *Supervisor validates constraints and activates relevant specialists.* | *Graph pauses via LangGraph `interrupt()` for traveler approval.* |

| **3. Grounded Specialist Intelligence** | **4. Final Travel Plan** |
| :---: | :---: |
| ![Specialist Tabs](docs/assets/portfolio/evidence-specialists.png) | ![Final Plan](docs/assets/portfolio/final-plan.png) |
| *Live AviationStack, Tavily, and OpenWeather typed evidence tabs.* | *Traveler-approved synthesis with copyable markdown & PDF export.* |

---

## 🎯 Why TripBandhu Exists

Most AI travel assistants suffer from three fundamental flaws:

1. **Single-Prompt Hallucination** — Fabricating flights, stale hotel rates, and inaccurate weather from static model weights with no live grounding.
2. **Brittle One-Shot Generation** — Producing entire multi-day itineraries in a single unconstrained LLM pass without modular domain validation.
3. **Black-Box Automation** — No mechanism for the traveler to inspect intermediate findings, refine preferences, or approve the schedule before finalizing.

**TripBandhu** treats travel planning as an **evidence-based distributed coordination problem**: deterministic routing, strict typed evidence contracts, schema-drift fallbacks, durable PostgreSQL checkpointing, and first-class Human-in-the-Loop (HITL) review.

---

## 🏛️ Core Architectural Principles

- **Deterministic State Machine** — Orchestrated via LangGraph DAG with explicit routing, state reducers, and defensive bounds
- **Truthful Evidence & Groundedness** — Every output tagged with semantic classification (`PROVIDER_DATA`, `WEB_SOURCE`, `MODEL_ESTIMATE`, `DERIVED`, `UNAVAILABLE`) and freshness (`LIVE`, `FORECAST`, `REFERENCE`)
- **True Async HITL** — Execution suspends via `interrupt()` and resumes with `Command(resume=...)`; review cap terminates safely without auto-finalizing
- **Resilient MCP Capability Layer** — All external tools pass through validated contracts with schema-drift detection and graceful degradation
- **Checkpoint Persistence** — `AsyncPostgresSaver` ensures graph runs survive complete service restarts

### Current Reliability Hardening

- **Clean specialist presentation** — Hotel and weather evidence is filtered, normalized, and converted into traveler-facing Markdown instead of exposing raw provider payloads or developer errors.
- **Strict specialist boundaries** — Weather output is rejected if it leaks budget, hotel, flight, or itinerary content; unsupported hotel search results are excluded before synthesis.
- **Complete long-form output** — Flight, budget, draft itinerary, and revision tasks use bounded continuation when a model stops at its token limit, avoiding half-finished tables and truncated plans.
- **Typed provider failures** — Errors embedded inside successful MCP envelopes are detected centrally. Tavily HTTP `429` responses are reported as temporary provider rate limits rather than misleadingly described as poor hotel-search results.
- **Location-aware tools** — Common city aliases are normalized for weather lookups, while flight requests resolve origin and destination locations to IATA codes before calling AviationStack.

---

## 📐 System Architecture

```mermaid
flowchart TB
    subgraph ClientLayer ["Client Layer"]
        UI["Retro-Editorial Web UI\n(Vanilla HTML5 / CSS / JS)"]
    end

    subgraph FastAPILayer ["FastAPI Application Core"]
        API["FastAPI App\n(/api/travel, /api/travel/resume, /api/travel/download-pdf)"]
        Guard["Input Guardrail\n(Scope, Safety & Prompt Injection)"]
        Pool["Application-Scoped\nAsyncPostgresSaver Pool"]
    end

    subgraph LangGraphCore ["LangGraph Stateful Execution Core"]
        Supervisor["Structured Supervisor\n(Constraint Extraction & Dispatch)"]
        subgraph Specialists ["Specialist Agents"]
            Flight["Flight\n(AviationStack MCP via uvx)"]
            Hotel["Hotel\n(Tavily Web Search MCP)"]
            Weather["Weather\n(OpenWeather Custom MCP)"]
            Budget["Budget\n(Cost Allocation Engine)"]
            Itinerary["Itinerary\n(Schedule Coordinator)"]
        end
        HITL["Human Review Gate\n(interrupt / Command resume)"]
        Synthesis["Final Plan Synthesis\n(Evidence-Aware Aggregator)"]
    end

    subgraph External ["External APIs & Storage"]
        AS["AviationStack\n(Live Routes & Status)"]
        TV["Tavily Search\n(Hotel & Attraction Data)"]
        OW["OpenWeather\n(Current & Forecast)"]
        PG[("PostgreSQL 15+\n(Checkpoints & Thread States)")]
    end

    UI --> API --> Guard --> Supervisor
    Supervisor --> Flight & Hotel & Weather & Budget --> Itinerary --> HITL
    HITL -->|Approved| Synthesis --> API --> UI
    HITL -->|Revision| Supervisor
    Flight -.->|MCP| AS
    Hotel -.->|MCP| TV
    Weather -.->|MCP| OW
    Pool <--> PG
    LangGraphCore <--> Pool
```

---

## 🤖 LLM Architecture (v1.0.2)

TripBandhu uses **task-specific Groq models** via the `make_groq_llm` factory with `max_retries=0`:

| Task | Model | Rationale |
| :--- | :--- | :--- |
| Guardrail, Supervisor, Hotel/Weather Presentation, Budget, Final Synthesis | `openai/gpt-oss-20b` | 12k TPM limit; structured routing and concise specialist presentation |
| Flight Summary, Itinerary, Revision | `openai/gpt-oss-120b` | Higher quality generation; 8k TPM |

> **Why different models for different tasks?**
> `gpt-oss-120b` has a hard 8,000 TPM limit on the Groq free tier. The final synthesis prompt (all specialist summaries + draft itinerary) exceeds this, so it uses `gpt-oss-20b` which supports 12,000 TPM.

---

## ⚡ Local vs Render — Why Local Run is Strongly Recommended

| Factor | Local Run | Render Free Tier |
| :--- | :--- | :--- |
| **AviationStack startup** | `uvx` cache: starts in ~1-2s | Cold-downloads MCP package on each deploy: 10-20s extra |
| **Weather API response** | Direct network: ~1-3s | Render NAT overhead: often 20s+, hits 25s timeout |
| **Service sleep** | Never sleeps | Spins down after 15 min; ~30-60s wake-up |
| **MCP subprocess stability** | Stable | Container memory pressure can crash subprocesses |
| **Timeout reliability** | Rarely triggers | Frequently triggers even with 45s client limits |
| **Flight data** | Consistently retrieves data | May time out during cold-start window |

**TL;DR:** Run locally for reliable flight data, weather, and complete hotel results. Use Render for quick demos only.

---

## 🚀 Setup & Demo — Step by Step

### Prerequisites

| Tool | Version | Notes |
| :--- | :--- | :--- |
| Python | 3.11+ | Required |
| PostgreSQL | 15+ | Optional — app uses in-memory if not configured |
| `uv` (provides `uvx`) | Latest | Required for AviationStack MCP |
| Git | Any | Clone repo |

```bash
# Install uv (if not already installed)
pip install uv
```

---

### Step 1 — Clone & Install

```bash
git clone https://github.com/tusharg007/TripBandhu.git
cd TripBandhu

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

### Step 2 — Configure `.env`

Create `.env` in the project root:

```env
# LLM (required)
GROQ_API_KEY=gsk_your_groq_key_here

# Specialist Data Providers (all optional — agents degrade gracefully without them)
TAVILY_API_KEY=tvly-your_tavily_key_here
AVIATION_STACK_API_KEY=your_aviationstack_key_here
OPENWEATHER_API_KEY=your_openweather_key_here

# PostgreSQL (optional — in-memory used if blank)
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/tripbandhu

# LangSmith Observability (optional; enable only after verifying the key/region)
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=lsv2_your_langsmith_key
LANGSMITH_PROJECT=TripBandhu
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

---

### Step 3 — Create PostgreSQL Database (Optional)

```sql
-- In psql or pgAdmin:
CREATE DATABASE tripbandhu;
```

---

### Step 4 — Verify AviationStack MCP

```bash
# Confirm uvx works
uvx --version

# Test AviationStack MCP starts without errors (Ctrl+C after it starts)
uvx --with mcp==1.28.1 aviationstack-mcp==1.6.0
```

If you see a pydantic warning but no `ModuleNotFoundError` — you're good.

---

### Step 5 — Start the Server

```bash
python app.py
```

Opens at **http://localhost:8080**

> Port conflict? Override it:
> ```powershell
> # Windows PowerShell
> $env:PORT="9000"; python app.py
>
> # macOS / Linux
> PORT=9000 python app.py
> ```

---

### Step 6 — Demo Queries

Open **http://localhost:8080** and try these in order:

#### Demo A — Focused Flight Query
```
What flights are available from Delhi to Bangkok?
```
Verifies: AviationStack MCP starts correctly, route data appears in Flights tab, no fake fares.

#### Demo B — Full 7-Day Budget Trip
```
Plan a 7-day budget trip from Delhi to Thailand for 2 people.
Include flights, hotels, weather, and a day-by-day itinerary.
```

**Expected flow:**
1. 5 specialist tabs fill in: Flights → Hotels → Weather → Budget → Draft Itinerary
2. **"Review Itinerary"** section appears — read the draft
3. Click **"Approve"** or type feedback and click **"Request Changes"**
4. After approval → **Final Travel Plan** generates
5. **"Copy"** for markdown export · **"Download PDF"** for PDF

#### Demo C — Cultural + Budget Focus
```
Plan a 7-day cultural and food trip from Delhi to Japan
with flights, hotels, weather, sightseeing, and a mid-range budget of ₹2.5 lakh.
```

#### Demo D — Focused Weather Query (no itinerary)
```
What is the weather like in Switzerland in late August?
```
Verifies: Supervisor activates only weather_agent (no itinerary gate).

---

### Step 7 — Running Tests

```bash
# Install the CI-pinned development test dependencies
pip install -r requirements-dev.txt

# Fast unit and contract tests (no external API calls)
pytest -m "not integration" -v

# Full benchmark — FAST mode (49 deterministic cases)
python -m eval.evaluator

# Full benchmark — POSTGRES mode (50 cases, requires DATABASE_URL)
python -m eval.evaluator --postgres
```

---

## ⚙️ Environment Variables Reference

| Variable | Required | Default | Description |
| :--- | :---: | :---: | :--- |
| `GROQ_API_KEY` | **Yes** | — | Groq API key |
| `TAVILY_API_KEY` | Optional | — | Hotel & web search |
| `AVIATION_STACK_API_KEY` | Optional | — | Live flight routes (also accepted as `AVIATIONSTACK_API_KEY`) |
| `OPENWEATHER_API_KEY` | Optional | — | Weather data |
| `DATABASE_URL` | Optional | `""` | PostgreSQL connection string |
| `PORT` | Optional | `8080` | Server port (Render sets this automatically) |
| `MAX_REVIEW_ITERATIONS` | Optional | `10` | Max HITL revision cycles |
| `LANGSMITH_TRACING` | Optional | `false` | Enable LangSmith tracing |
| `LANGSMITH_API_KEY` | Optional | — | LangSmith API key |
| `LANGSMITH_PROJECT` | Optional | `TripBandhu` | LangSmith project name |

> **Free-tier limits:**
> - Groq `gpt-oss-120b`: 8,000 TPM hard limit
> - Groq `gpt-oss-20b`: 12,000 TPM hard limit
> - AviationStack free: 100 API calls/month
> - OpenWeather free: 60 calls/minute

---

## 🔄 Human-in-the-Loop Workflow

```
User Query
    │
    ▼
[Guardrail] ── blocked ──► "Out of scope" response
    │ allowed
    ▼
[Supervisor] extracts trip constraints
    │
    ├──► [Flight Agent]    → AviationStack MCP
    ├──► [Hotel Agent]     → Tavily web search
    ├──► [Weather Agent]   → OpenWeather MCP
    └──► [Budget Agent]    → LLM cost estimation
    │
    ▼
[Itinerary Agent] assembles draft plan
    │
    ▼
[Human Review Gate] ──interrupt()──► UI shows draft to traveler
    │
    ├── Approve ────────────────────► [Final Synthesis] → Final plan shown
    │
    └── Revision + feedback
              │
              ▼
         [Revision Agent] refines draft
              │
              ▼
         [Human Review Gate] again (up to MAX_REVIEW_ITERATIONS)
              │
              └── Cap reached ──► Safe termination, no auto-finalization
```

---

## 📊 Evaluation Results (v1.0.2)

| Suite | Result |
| :--- | :--- |
| Pytest regression | ✅ 137 fast + 1 PostgreSQL integration test |
| Benchmark FAST mode (49 cases) | ✅ 49 / 49 passed |
| Benchmark POSTGRES mode (50 cases) | ✅ 50 / 50 passed |
| Live HITL validation on Render | ✅ Passed |

---

## 🛠️ Troubleshooting

| Error | Cause | Fix |
| :--- | :--- | :--- |
| `[WinError 10013]` on startup | The selected port is already in use | Set `$env:PORT="9000"` |
| `ModuleNotFoundError: mcp.server.fastmcp` | Old uvx cache / wrong mcp version | Fixed in v1.0.2 via `--with mcp==1.28.1` in uvx args |
| `413 Request too large` | Groq TPM limit exceeded | Fixed in v1.0.2 — final synthesis uses `gpt-oss-20b` |
| `No module named 'reportlab'` when downloading a PDF | Dependencies were installed from an older or malformed requirements file | Pull the latest code and run `pip install -r requirements.txt` in the active environment |
| Flights always "unavailable" | AviationStack key missing or uvx crash | Check `AVIATION_STACK_API_KEY` in `.env`; test uvx manually |
| Hotels/weather show raw provider text | Provider evidence reached the UI without relevance and presentation layers | Hotel relevance filtering, safe weather normalization, and dedicated presentation passes now produce end-user Markdown |
| Hotels show `DEGRADED` and Tavily reports HTTP `429` | Tavily rejected the live search because the API key exceeded its request-rate allowance | Wait briefly and start a new trip. If it happens frequently, check Tavily usage and use a production/PAYGO key; TripBandhu does not blindly retry when MCP omits `Retry-After` |
| Final plan: "could not be generated" | Groq rate limit / TPM exceeded | Wait 1 min (TPM reset) then retry |
| LangSmith `403 Forbidden` or multipart ingest failures | Tracing key, workspace, or regional endpoint does not match | Keep `LANGSMITH_TRACING=false`, or configure the endpoint/workspace for the key before enabling tracing |
| Browser still shows an older UI/PDF behavior | Cached static assets from an earlier deploy | Hard-refresh the page; deployed assets are commit-versioned |

---

## 🗺️ Repository Structure

```
TripBandhu/
├── .github/workflows/ci.yml          # GitHub Actions CI
├── eval/
│   ├── eval_dataset.py               # 50-case benchmark dataset (v3.1.0)
│   └── evaluator.py                  # FAST / POSTGRES benchmark harness
├── tests/                            # 137 fast tests + PostgreSQL integration test
├── docs/assets/portfolio/            # UI screenshots
├── static/
│   ├── script.js                     # Frontend logic, HITL, PDF export
│   └── style.css                     # Retro-editorial design system
├── templates/index.html              # Single-page app shell
├── app.py                            # FastAPI server & endpoints
├── backend.py                        # LangGraph graph + all agent functions
├── mcp_client.py                     # MCP server configs & tool adapters
├── capability_registry.py            # Evidence normalizers (Tavily, Weather, Flight)
├── provider_utils.py                 # Async MCP call wrapper (timeout + retry)
├── pdf_generator.py                  # Validated server-side itinerary PDF renderer
├── agent_config.py                   # Timeouts, token budgets, model assignments
├── llm_utils.py                      # LLM invocation with rate-limit handling
├── schemas.py                        # Typed evidence, error codes, state schemas
├── custom_weather_mcp_server.py      # OpenWeather MCP server (FastMCP stdio)
├── project_config.py                 # Path resolution
├── requirements.txt
├── requirements-dev.txt              # CI-pinned test dependencies
├── .env.example
└── README.md
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
