<div align="center">

# ✈️ TripBandhu

### Your AI travel companion—from a single idea to a practical journey

TripBandhu turns a natural-language travel request into a grounded travel brief with live flight-status data, hotel research, a budget-aware itinerary, and a polished final plan. A stateful LangGraph workflow coordinates specialist stages while FastAPI, PostgreSQL, and a responsive web experience make the planner usable beyond a notebook demo.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-1C3C3C?style=for-the-badge)](https://www.langchain.com/langgraph)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Stateful-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-F4B942?style=for-the-badge)](LICENSE)

[![Live on Render](https://img.shields.io/badge/Live_on_Render-Open_TripBandhu-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://tripbandhu.onrender.com)

### [🌐 Launch the live application](https://tripbandhu.onrender.com)

[Try it live](#-try-tripbandhu-live) · [Explore the workflow](#-how-tripbandhu-thinks) · [Run locally](#-run-it-locally) · [Use the API](#-api-reference) · [Deploy with Docker](#-run-with-docker)

</div>

---

## 🌐 Try TripBandhu live

TripBandhu is deployed on Render and ready to use in the browser—no local installation required.

<div align="center">

### **[Open TripBandhu AI →](https://tripbandhu.onrender.com)**

[Live application](https://tripbandhu.onrender.com) · [Health check](https://tripbandhu.onrender.com/health) · [Interactive API docs](https://tripbandhu.onrender.com/docs)

</div>

Submit a request such as:

> *Plan a complete 7-day Japan trip from India with flights, hotels, sightseeing, and a budget under 2 lakh INR.*

## 🌍 What is TripBandhu?

Planning a trip is rarely one task. It means interpreting an open-ended request, identifying a route, checking flights, researching places to stay, balancing time and budget, and turning scattered information into something a traveler can actually follow.

TripBandhu models that process as a coordinated graph:

- a **flight specialist** translates natural language into airport routes and retrieves live operational flight data;
- a **hotel researcher** searches the web for relevant accommodation ideas;
- an **itinerary planner** combines the collected context into a practical schedule;
- a **final travel assistant** transforms everything into one clear, structured response;
- a **PostgreSQL checkpointer** preserves graph state under a reusable conversation thread.

> [!IMPORTANT]
> TripBandhu is a travel research and itinerary assistant—not a booking engine. AviationStack provides flight schedules and status information, not live ticket fares. Always verify availability, prices, visa rules, and local guidance before booking.

## ✨ Experience highlights

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>🗣️ Natural-language route intelligence</h3>
      Understands requests such as <em>“7 days in Japan from India”</em>, explicit routes such as <code>DAC to NRT</code>, destination-only searches, country names, city names, and global-flight queries.
    </td>
    <td width="50%" valign="top">
      <h3>✈️ Grounded flight research</h3>
      Resolves cities and countries to IATA airports, then returns airline, flight number, status, terminals, gates, schedules, and delays from AviationStack.
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>🏨 Web-powered hotel discovery</h3>
      Uses Tavily to collect focused hotel suggestions with source titles, URLs, and concise search snippets instead of relying only on model memory.
    </td>
    <td width="50%" valign="top">
      <h3>🧠 Two-stage LLM synthesis</h3>
      Llama 3.3 70B through Groq first builds a practical itinerary, then produces a traveler-friendly answer with flights, hotels, daily plans, budget guidance, and recommendations.
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>💾 Thread-scoped memory</h3>
      LangGraph checkpoints are stored in PostgreSQL. The browser retains the active <code>thread_id</code> so follow-up requests can continue under the same graph thread.
    </td>
    <td width="50%" valign="top">
      <h3>📄 A result you can take with you</h3>
      The responsive glassmorphism UI renders Markdown, includes quick prompts and keyboard submission, and lets travelers copy a plan or export it as a styled PDF.
    </td>
  </tr>
</table>

## 🧭 How TripBandhu thinks

```mermaid
flowchart LR
    U["Traveler request"] --> API["FastAPI /api/travel"]
    API --> G["LangGraph workflow"]

    subgraph FLOW["Specialist workflow"]
        direction LR
        F["✈️ Flight agent<br/>route parsing + AviationStack"] --> H["🏨 Hotel agent<br/>Tavily research"]
        H --> I["🗓️ Itinerary agent<br/>Groq + Llama 3.3 70B"]
        I --> R["✨ Final response agent<br/>structured travel brief"]
    end

    G --> F
    R --> API
    G <--> DB[("PostgreSQL<br/>checkpoints")]
    API --> UI["Responsive web UI<br/>Markdown · Copy · PDF"]
```

### Route-resolution pipeline

The flight tool does more than forward a search string. It progressively resolves the traveler’s intent:

1. Detect direct IATA pairs such as `DEL to NRT`.
2. Parse phrases such as `from Dhaka to Tokyo`, `to Japan from India`, or `flights from Delhi`.
3. Match preferred city airports and country aliases.
4. Search the local `airportsdata` catalog when there is no curated match.
5. Use `DEFAULT_ORIGIN_IATA` for destination-only trip requests.
6. Query AviationStack with the resolved departure and/or arrival filters.

This keeps route interpretation deterministic while reserving the LLM for itinerary reasoning and final presentation.

## 🧱 Technology map

| Layer | Technology | Responsibility |
|---|---|---|
| Web API | FastAPI + Uvicorn | Serves the UI, health check, and travel-planning endpoint |
| Agent orchestration | LangGraph | Defines state, specialist nodes, execution order, and checkpoint integration |
| LLM integration | LangChain + ChatGroq | Calls `llama-3.3-70b-versatile` for itinerary and response synthesis |
| Flight intelligence | AviationStack + airportsdata + pycountry | Retrieves live flight data and resolves human locations to IATA codes |
| Hotel research | Tavily | Finds current hotel-related web results and source links |
| State persistence | PostgreSQL + `PostgresSaver` | Stores LangGraph checkpoints by `thread_id` |
| Frontend | HTML + CSS + JavaScript | Responsive planner, quick prompts, loading/error states, and result rendering |
| Client-side output | marked + html2pdf.js | Renders the Markdown answer and exports it to PDF |
| Packaging | Docker | Provides a reproducible Python 3.11 runtime |
| Optional observability | LangSmith | Captures LangChain/LangGraph traces when explicitly enabled |

## 📁 Project anatomy

```text
TripBandhu/
├── app.py                  # FastAPI application and HTTP routes
├── backend.py              # LangGraph state, agents, edges, LLM, and checkpointer
├── tools/
│   ├── flight_tool.py      # Route parsing, airport resolution, AviationStack client
│   ├── tavily_tool.py      # Hotel-oriented Tavily search and result formatting
│   └── __init__.py
├── templates/
│   └── index.html          # Main planner interface
├── static/
│   ├── style.css           # Responsive glassmorphism design and print styling
│   └── script.js           # API calls, thread persistence, Markdown, copy, PDF export
├── test.py                 # Interactive command-line smoke test
├── requirements.txt        # Pinned Python dependencies
├── Dockerfile              # Container image definition
├── .dockerignore           # Docker build-context exclusions
├── .gitignore              # Local and secret-file exclusions
├── LICENSE                 # MIT License
└── README.md
```

## 🚀 Run it locally

### Prerequisites

- Python **3.11** recommended
- PostgreSQL database
- [Groq API key](https://console.groq.com/keys)
- [Tavily API key](https://app.tavily.com/)
- [AviationStack API key](https://aviationstack.com/)
- Optional: a [LangSmith API key](https://smith.langchain.com/) for tracing

### 1. Clone the repository

```bash
git clone https://github.com/tusharg007/TripBandhu.git
cd TripBandhu
```

### 2. Create an isolated environment

<details open>
<summary><strong>Using venv</strong></summary>

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

</details>

<details>
<summary><strong>Using Conda</strong></summary>

```bash
conda create -n tripbandhu python=3.11 -y
conda activate tripbandhu
```

</details>

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Using `python -m pip` ensures packages are installed into the interpreter that will run the application.

### 4. Configure the environment

Create a `.env` file in the repository root:

```dotenv
# Required
GROQ_API_KEY="your_groq_api_key"
TAVILY_API_KEY="your_tavily_api_key"
AVIATIONSTACK_API_KEY="your_aviationstack_api_key"
DEFAULT_ORIGIN_IATA="DEL"

# Cloud PostgreSQL / Render-style connection
DATABASE_URL="postgresql://user:password@host:5432/database?sslmode=require"

# Optional LangSmith tracing
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="your_langsmith_api_key"
LANGSMITH_PROJECT="TripBandhu"
# LANGSMITH_WORKSPACE_ID="your_workspace_id"
```

For a local PostgreSQL server that does not use TLS, explicitly set `sslmode=disable`:

```dotenv
DATABASE_URL="postgresql://postgres:password@localhost:5432/tripbandhu?sslmode=disable"
```

> [!CAUTION]
> Never commit `.env`. It is already excluded by this repository’s `.gitignore`. If a key is ever exposed, revoke and rotate it immediately.

### 5. Start the application

```bash
python app.py
```

Open **http://127.0.0.1:8000**. FastAPI also provides interactive API documentation at **http://127.0.0.1:8000/docs**.

> [!NOTE]
> The application connects to PostgreSQL and initializes the LangGraph checkpoint tables during startup. An invalid database URL or missing Groq key will prevent the server from starting.

## 🐳 Run with Docker

Build the image:

```bash
docker build -t tripbandhu .
```

Start a container with your environment file:

```bash
docker run --rm --env-file .env -p 8000:8000 tripbandhu
```

Then visit **http://localhost:8000**.

If PostgreSQL is running on your host machine, `localhost` inside the container points to the container itself. On Docker Desktop, use `host.docker.internal` in `DATABASE_URL`, or connect both services through a Docker network.

## 🔌 API reference

### `GET /`

Serves the TripBandhu planner interface.

### `GET /health`

```json
{
  "status": "ok",
  "message": "AI Travel Planner API is running"
}
```

### `POST /api/travel`

Creates a travel plan or continues an existing thread.

#### Request

```json
{
  "message": "Plan a 7-day Japan trip from India under 2 lakh INR",
  "thread_id": null
}
```

`thread_id` is optional. Omit it to create a new UUID-backed thread; reuse the returned value for subsequent requests in the same conversation.

#### cURL example

```bash
curl --request POST http://127.0.0.1:8000/api/travel \
  --header "Content-Type: application/json" \
  --data '{"message":"Plan a 5-day Dubai trip from Dhaka with hotels and sightseeing."}'
```

#### Response shape

```json
{
  "success": true,
  "thread_id": "user_7e8f...",
  "answer": "# Your Travel Plan...",
  "flight_results": "Live flights from DAC to DXB...",
  "hotel_results": "1. Hotel result...",
  "itinerary": "Day 1...",
  "llm_calls": 4
}
```

Validation failures return HTTP `400`; unexpected tool, model, or database failures return HTTP `500` with `success: false`.

## 🧪 Interactive smoke test

With PostgreSQL running and `.env` configured:

```bash
python test.py
```

Enter a travel request at the prompt. The script invokes the same graph used by FastAPI under the fixed thread ID `test_user` and prints the final answer. This is an interactive smoke test, not an automated test suite.

## 🛠️ Troubleshooting

| Symptom | Likely cause | Resolution |
|---|---|---|
| `ModuleNotFoundError: No module named 'tavily'` | Requirements were installed into another Python environment | Activate the intended environment and compare `python -m pip --version` with `python -c "import sys; print(sys.executable)"` |
| `python-dotenv could not parse statement` | An `.env` key contains spaces or invalid syntax | Use identifiers such as `LANGSMITH_API_KEY`, never `LANGSMITH API KEY` |
| LangSmith `/runs/multipart` returns `403` | Revoked key, wrong LangSmith region, or missing workspace context | Rotate the key, match the regional endpoint, or set `LANGSMITH_WORKSPACE_ID`; disable tracing if unused |
| Server fails during import/startup | PostgreSQL is unreachable or `GROQ_API_KEY` is missing | Verify `DATABASE_URL`, SSL mode, credentials, firewall, and Groq configuration |
| No flight prices appear | AviationStack supplies operational flight data rather than fares | Treat results as schedule/status research and verify prices with a fare or booking provider |
| Destination-only request uses the wrong departure | `DEFAULT_ORIGIN_IATA` does not match the traveler’s home airport | Set a valid three-letter IATA code in `.env` |

## 🔭 Current boundaries and roadmap

TripBandhu currently prioritizes a transparent, understandable workflow. That also leaves clear opportunities for evolution:

- [ ] Integrate a fare-search or booking API for real ticket prices
- [ ] Run independent flight and hotel research stages in parallel
- [ ] Add dates, traveler count, currency, and budget as structured inputs
- [ ] Add automated unit, integration, and graph evaluation suites
- [ ] Stream agent progress and partial responses to the browser
- [ ] Add user accounts and a saved-trip dashboard
- [ ] Expand airport ranking beyond curated city/country mappings
- [ ] Add maps, weather, activities, visa guidance, and local transport research

## 🤝 Contributing

Contributions that improve reliability, travel-data quality, UX, evaluation, or documentation are welcome.

1. Fork the repository.
2. Create a branch: `git checkout -b feature/your-improvement`.
3. Make and test your changes.
4. Commit with a clear message.
5. Push the branch and open a pull request.

Please avoid committing API keys, database credentials, generated environments, or personal travel data.

## 📜 License

TripBandhu is released under the [MIT License](LICENSE).

---

<div align="center">

Built by [Tushar Ghosh](https://github.com/tusharg007) with a belief that good travel technology should turn scattered information into confident decisions.

**If TripBandhu helps or inspires you, consider giving the repository a ⭐.**

</div>
