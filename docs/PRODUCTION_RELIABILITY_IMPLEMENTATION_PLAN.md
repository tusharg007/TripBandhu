# TripBandhu: reliable evidence and professional itinerary exports

Prepared 15 September 2026; implementation status updated 21 September 2026. This plan is based on inspected code, the supplied PDF and trace, and read-only checks of the public Render service. Phase 0 and Phase 1 are implemented and previously verified on their noted build. Phases 2–5 are now implemented locally and require deployment of the new commit plus a restored Render PostgreSQL service for durable-production acceptance.

## 1. Findings and confidence

### Confirmed deployment mismatch

The public homepage at https://tripbandhu.onrender.com returned HTTP 200 and referenced `/static/script.js?v=6d27667bb5e3`. Its HTML still includes `html2pdf.bundle.min.js`. The served JavaScript calls `html2pdf().from(pdfContent).save()` with image capture and `pagebreak: { mode: ["avoid-all", "css", "legacy"] }`.

The matching local Git commit is `6d27667`, dated 24 August 2026. The current local HEAD is `480367c`; it contains the server-side PDF changes and subsequent specialist fixes. The deployed asset stamp precedes those changes. The public `/health` endpoint reports only that the API is running; it does not expose a build identity or demonstrate provider availability.

This confirms that the live frontend is serving an older implementation. It strongly suggests an old backend deployment as well, but the actual running backend SHA must be verified through Render's deployment record or a version endpoint. The reason Render remained on that version is not yet established: check its selected repository/branch, auto-deploy configuration, deploy history, failed builds, rollbacks, and domain-to-service mapping.

### Confirmed PDF defects

The supplied `F:\tripbandhu-travel-plan.pdf` has six pages and producer metadata `jsPDF 2.3.1`. Text extraction returns no text from any page. All six pages were rendered and visually inspected:

- Page 1 is blank.
- Content starts partway down page 2.
- Tables are cut across pages without useful continuation headers.
- Text overlaps in narrow location cells.
- The daily detailed itinerary stops during Day 1 on the final page.
- The export copies website styling instead of using a print layout.

This PDF did not come from the current ReportLab renderer. Correcting deployment should remove this particular legacy path, but the current generic Markdown renderer still needs a dedicated document design to meet the requested travel-agency quality.

### Confirmed trace failures

The supplied JSON describes a Delhi-to-Georgia, five-day draft; the PDF describes a seven-day Greece trip. They are separate examples and must remain separate test fixtures. The JSON has `approved: null` and an empty `final_response`; it cannot establish the content of the subsequently approved itinerary.

| Capability | Observed result | Interpretation |
| --- | --- | --- |
| Aviation routes | TIMEOUT, 16.065 seconds | No provider response available in this trace |
| Airport lookup | TIMEOUT, 15.896 seconds | Repeated failure adds latency |
| Airline lookup | TIMEOUT, 15.497 seconds | Repeated failure adds latency |
| Tavily hotel research | AVAILABLE, 7.034 seconds; empty title placeholder, URL and snippet | False positive evidence availability |
| Current weather | TIMEOUT, 12.151 seconds | No valid observation retrieved |
| Forecast | TIMEOUT, 12.103 seconds | No valid forecast retrieved |

The old deployed commit configures 15-second aviation and 10-second weather timeouts, consistent with these observations after startup/cancellation overhead. Current local defaults are 45 and 25 seconds. This trace contains no reported 429 or authentication rejection. Remaining credits cannot prevent subprocess startup, transport, timeout, entitlement, or schema failures.

The six provider attempts total about 78.7 seconds. The trace does not separate process startup, tool discovery, network time, provider execution, and parsing, so it cannot prove which stage consumes that time.

### Confirmed factual and completeness problems

The draft budget converts USD 8,930 into INR 7.3-7.6 million using 82-85 INR/USD. At the stated rate the correct result is INR 732,260-759,050, or about 7.32-7.59 lakh. Its category totals also need reconciliation. Budget and itinerary text both end mid-section. Specific hotel amenities and inclusion claims appear despite empty hotel evidence; those claims are unsupported by this run, whether or not an individual claim happens to be true.

Approval confirms the user's preference, not factual accuracy. A polished final answer requires the same evidence and arithmetic validation as a draft.

## 2. Target architecture

```text
Request + traveler/session identity
    -> normalized trip constraints and locations
    -> provider adapters + validated evidence + freshness
    -> structured draft + deterministic checks
    -> human review / revision
    -> immutable approved itinerary version
         -> web presentation
         -> professional PDF renderer -> stored artifact -> authorized download
```

Both outputs must use the same saved itinerary version. Formatting may differ; dates, days, hotel choices, prices, source references, and warnings must agree. Export must not trigger a new travel-planning LLM call.

## 3. Implementation phases and acceptance gates

### Phase 0: restore and prove deployment parity

Primary areas: Render configuration, `Dockerfile`, `app.py`, `templates/index.html`, `static/script.js`, CI.

1. Inspect the Render service's connected repository, branch, build/start command, latest successful deploy, auto-deploy rules, and failed-deploy logs. Confirm the public hostname maps to that service.
2. Deploy a deliberately selected, tested commit containing the existing PDF/provider fixes. Do not blindly replay old commits or erase unrelated local edits; `docker-compose.yml` is already modified locally.
3. Add `/version` with build SHA, application version, export renderer version, and schema version. Include build SHA and effective timeout/config versions in run metadata. Do not expose credentials.
4. Keep HTML revalidated and static assets versioned by build/content. Eliminate the legacy html2pdf dependency from the served page and verify the browser actually calls the server export endpoint.
5. Add a post-deploy smoke check that compares expected SHA, returned SHA, asset URLs, and exported PDF producer/text. A green health endpoint alone is insufficient.

Acceptance: served JS uses `/api/travel/download-pdf`; public build identity matches the release; downloaded PDF has extractable text and no blank leading page. Record one fresh complete request and approval from this build before diagnosing remaining providers.

Implementation status: complete. Build `18f9928` was verified on the public Render hostname through `/version`, commit-versioned assets, and a ReportLab PDF smoke download. A fresh public two-day Delhi-to-Jaipur request completed draft and approval. Its trace established the Phase 1 baseline: the old AviationStack MCP flow consumed approximately 110 seconds across three serial attempts while Tavily completed in approximately eight seconds.

### Phase 1: isolate and repair provider failures

Primary areas: `mcp_client.py`, `provider_utils.py`, `capability_registry.py`, `custom_weather_mcp_server.py`, `agent_config.py`, new adapter and diagnostic modules.

1. Add a restricted diagnostic command that performs bounded, minimal requests from the actual deployed runtime. Compare direct HTTPS calls with the corresponding MCP calls. Record status, provider error code, request ID, Retry-After, elapsed stage timings, and normalized record counts. Never print keys, credential-bearing URLs, raw exceptions containing keys, or private itinerary data.
2. Verify endpoint entitlement as well as key validity. A key may have credits but lack access to a specific AviationStack endpoint. Validate routes, schedules, dates, and filters against the subscribed plan; classify unsupported access separately from timeout. Do not silently retry forbidden endpoints or change providers to hide the result.
3. Recommended production transport: application-scoped async HTTP clients for the three fixed providers, behind the existing capability interface. Tavily exposes structured search JSON; weather is already a small HTTP wrapper; direct adapters avoid repeated subprocess/discovery work. Preserve optional MCP adapters for compatibility and comparison. A transport switch must not duplicate an already successful provider request.
4. If MCP remains on the critical path, manage sessions and tool discovery through application lifespan, close them cleanly, and install pinned executable dependencies at image build time. Avoid per-request `uvx` downloads. Measure startup versus request latency rather than only enlarging timeouts.
5. Add bounded timeouts and retries with jitter and an overall request budget. Retry transient transport/5xx failures; honor Retry-After for 429; do not retry invalid input or entitlement/auth failures. Use a short circuit-breaker cooldown after repeated outages.
6. Deduplicate simultaneous identical requests. Cache only validated evidence using keys that include location, route, dates, occupancy, currency and relevant filters. Initial TTLs to validate: current weather 10 minutes, forecasts 30 minutes, hotel research six hours, airport reference data days. Follow provider terms; never label cached data as freshly live. Cache failures separately with short cooldowns.
7. Replace `source_count=1 if data else 0` with the number of validated sources. Transport success and usable evidence are distinct states. Require meaningful destination-matched hotel content and a valid source URL; reject placeholder-only records. Parse MCP text blocks, structured content, error flags, HTTP bodies and schema drift explicitly.
8. Geocode a selected city into coordinates and country code before querying weather. Resolve country-only requests through the trip's explicit city choices; clarify ambiguous locations. Use airport codes only for aviation, not as unqualified weather names.
9. Preserve weather observation time, forecast interval, timezone and units. The current weather server keeps only five three-hour entries; aggregate the available forecast by destination-local day instead. Never present a short forecast as the weather for a trip outside its horizon. Use sourced seasonal guidance with a separate label when dates are unavailable or beyond the forecast window.
10. Use local airport references where suitable rather than repeatedly fetching global airport/airline catalogues for every itinerary. Validate requested routes and label schedules, observed flight status, and price estimates separately. Aviation data must not be represented as a bookable fare quotation.

Acceptance: a successful raw provider response becomes usable evidence; an empty/malformed response cannot become COMPLETED with live data. Every failure has an actionable internal cause and a concise user message. Direct/MCP comparisons from the deployed region establish the actual remaining cause. Local success alone is not the release gate.

Implementation status: direct-adapter core complete locally. AviationStack now uses one route-specific HTTPS call and local airport reference data; Tavily and OpenWeather use pooled direct HTTPS; weather geocoding is coalesced and current/forecast calls execute concurrently; forecasts are aggregated by destination-local date. Validated TTL caching, cancellation-safe identical-request coalescing, concurrency limits, a circuit breaker, typed access/rate/timeout/invalid-response errors, sanitized tracing, and a restricted provider diagnostic command are covered by 152 fast unit/contract tests and a 49/49 deterministic benchmark. The final pre-deployment live diagnostic returned 10 route-matched aviation records in 890 ms, six destination-relevant Tavily sources in 764 ms, current weather in 313 ms, and six daily forecast groups in 485 ms. The remaining gate is deployment of Phase 1 followed by the same sanitized diagnostic and one traced public request from that build. Optional MCP comparison remains available through `PROVIDER_TRANSPORT=mcp`; it is not an automatic duplicate fallback.

### Phase 2: establish a validated itinerary contract

Primary areas: `schemas.py`, `backend.py`, `llm_utils.py`, new itinerary validation and persistence modules.

1. Introduce a versioned `ItineraryDocument`: origin/destinations, coordinates/timezones, dates or explicitly undated days, traveler count, room/night assumptions, daily activities, transport legs, accommodation options, budget line items, evidence IDs, retrieval timestamps, warnings, and approval/version identifiers.
2. Use structured generation with schema validation and bounded repair. Every precise factual claim must reference evidence or carry an explicit estimate/unverified status. Do not invent hotel amenities, breakfast inclusion, confirmed availability, operating hours or flight departure times when evidence is absent. Unsupported named recommendations should be omitted or clearly presented as unverified suggestions requiring research.
3. Ask for dates, traveler count and occupancy when needed for personalized live availability/pricing. Otherwise state assumptions and deliver a planning estimate; a country and luxury preference are insufficient to produce a quotation.
4. Calculate totals, ranges, contingency, traveler/night quantities and currency conversions in Python using Decimal. Store exchange-rate value, source/date and estimate status. Use INR as primary currency for India-origin trips and local currency where useful. Do not let the LLM independently recalculate displayed totals or infer a salary.
5. Validate day count, date continuity, feasible transfers, location consistency, duplicated activities, budget reconciliation, source linkage and forecast coverage. Add a bounded evidence review for disputed facts; deterministic checks remain authoritative for arithmetic and structure.
6. Replace text concatenation as the completion guarantee. `invoke_llm_complete_text` currently returns accumulated text even if its final allowed segment still hits a length limit; final synthesis has a separate ordinary call. Require a complete validated document before declaring completion. Repair missing sections within budget or return an explicit incomplete state, preserving completed work.
7. Persist the approved document version. Cosmetic finalization must preserve approved choices; substantive corrections require a new revision and user review. Existing threads need an explicit schema compatibility/migration policy, with no retroactive claim that old drafts were approved or verified.

Acceptance: the supplied currency error and truncated draft fail validation; all requested days survive approval and export; exported version/hash equals the displayed approved version; no additional LLM is needed for repeat downloads.

Implementation status: implemented locally. `ItineraryDocument` is a versioned Pydantic contract with contiguous-day validation, approval state, content hashing, source notes, warning severity, and Decimal-backed planning-budget line items. A missing requested day becomes a visible validation warning and cannot be saved/exported. The export boundary rejects non-approved or incomplete documents rather than inventing omitted content.

### Phase 3: design and build the travel-agency PDF

Primary areas: `pdf_generator.py`, bundled fonts/assets, export endpoint and tests.

Keep ReportLab as the initial production renderer, but feed it the structured document and dedicated page components. It already produces selectable text and avoids the memory cost of launching a browser on every download. Do not extend the current line-by-line Markdown parser as the primary document model. A future renderer replacement must prove better pagination and resource usage with the same fixtures.

Design brief:

- A4 portrait with generous margins, evergreen/cream colors, restrained gold accents, clear typography and bundled licensed Unicode fonts.
- Cover with destination image, trip title, route, duration, travel dates/assumptions, traveler count and document reference.
- At-a-glance trip summary, transport and accommodation cards, followed by complete daily plans using morning/afternoon/evening blocks. Keep long prose out of dense multi-column tables.
- Budget table with correct totals and explicit per-person/per-group and per-night/per-stay units; separate estimated and sourced values.
- Weather/packing guidance, booking checklist, source notes, retrieval dates and image credits.
- Subtle TripBandhu watermark behind content; page X of Y, running destination title, generation timestamp and revision. Use “Travel proposal” and “Not a booking confirmation” where appropriate. A watermark must not imply an actual travel agency has booked or certified the trip.
- Measured row heights, weighted table columns, repeating headers, controlled page breaks, long-row splitting, and heading/paragraph keep rules. Avoid keeping an entire lengthy day/table together.

Images: begin with a small curated library of licensed, destination-verified photographs and attribution metadata. Use one cover image and occasional chapter images, with consistent crops. Cache resized images; impose byte/pixel limits. Do not hotlink arbitrary search images, assume search results grant reuse rights, or let untrusted URLs access internal services. Missing images must fall back to a tasteful typographic cover without blocking export. Synthetic imagery, if later added, should be labeled illustrative and never used as evidence of a hotel or destination feature.

Acceptance: inspect every rendered page for 3-, 5-, 7-, 14- and 30-day fixtures; zero blank pages, clipping, overlap or missing days. Text is selectable/searchable; links work; INR and multilingual names render; the document remains attractive without images. Compare extracted content to the canonical approved document, not just the PDF header or file size.

Implementation status: implemented locally with a dedicated ReportLab renderer, not a website screenshot. The renderer has a travel-proposal cover, route/duration cards, day blocks, planning-budget table, source notes, page numbering, subtle watermark, and booking-verification language. A rendered multi-day QA PDF was inspected for pagination and text extraction. Broader 5/7/14/30-day visual fixture review remains a deployment/release acceptance task.

### Phase 4: store and serve approved exports

Primary areas: `app.py`, `static/script.js`, PostgreSQL artifact records, optional `artifact_storage.py`.

1. Change export requests from browser-supplied `{text, title}` to `{plan_id, version}`. Validate session/user ownership, approval and completeness server-side. A random thread ID alone is not authorization; support a signed anonymous session if login is not required.
2. Create an export artifact keyed by approved content hash, renderer version and asset version. Repeated clicks reuse that artifact; revisions create new ones. Return a clear pending/ready/failed state with retry only for export failures.
3. Start with authenticated streaming for the repair. Add private S3 storage for persistent downloads and sharing if required: least-privilege credentials, encryption, content metadata, short-lived presigned GET links, and a retention/deletion policy. Authorization happens before issuing a link; a presigned link is a bearer credential.
4. S3 holds the completed PDF bytes. It does not render, format, fact-check or automatically repair a document. Keep artifact metadata in PostgreSQL; do not depend on Render's local filesystem for persistence.
5. If storage fails, preserve the approved itinerary and permit retry. Where appropriate, allow authorized direct streaming of an already generated PDF without a second planning run.

Acceptance: download works after refresh/restart; versions cannot be mixed; another session cannot retrieve the plan; expired links can be renewed by the owner; repeated clicks do not spend provider/LLM quota.

The Cortex project was not found among the top-level F: directories during this inspection. Its implementation has not been reviewed. If its checkout becomes available, review only its artifact storage/download pattern and adapt that to TripBandhu's authorization and versioning requirements.

Implementation status: implemented locally with authenticated direct streaming first. A signed HttpOnly browser session owns a thread and immutable approved plan/version. The export API accepts only that plan reference, authorizes ownership before rendering, and caches renderer/asset-version PDF bytes. PostgreSQL tables persist thread ownership, document JSON, and PDF bytes when `DATABASE_URL` is reachable; a clearly marked in-memory fallback is used otherwise. S3 is intentionally deferred: it is useful only for sharing/large artifact retention and does not correct generation quality.

### Phase 5: operate reliably on Render

1. Keep Render's container deployment. Separate liveness, readiness and restricted dependency diagnostics; health polling must not repeatedly consume API quota.
2. For production traffic, use an always-on web service and a durable job worker with external PostgreSQL. A PostgreSQL-backed job table with leases/recovery is sufficient initially; Redis is optional if an existing queue requires it. Do not use a background coroutine in an expiring HTTP request as a durable job system.
3. Return a run/job ID promptly; use polling or SSE for stage progress. Bound concurrency per provider and globally; add per-session quotas, idempotency keys and abuse controls. On restart, reclaim abandoned jobs safely. A human-review pause must not occupy a worker or keep an HTTP request open.
4. Keep database and application regions close, bound pools, and validate concurrent PDF memory/CPU usage. Pin and prebuild required dependencies and assets. Add readiness checks for the renderer/fonts and database migrations.
5. Instrument build version, stage latency, normalized evidence count, failure category, token usage, completion reason, export duration and artifact version. Restrict trace access and redact credentials and unnecessary personal data before export.
6. Establish a staged release: deterministic CI, Docker parity tests, limited live provider probes, deployed end-to-end approval/download, then public rollout. Roll back on verified regressions, preserving approved document versions.

Implementation status: local safeguards are in place: bounded per-instance trip concurrency, per-session request limiting, session cookies, `/health/live`, and dependency-mode `/health/ready`. The readiness endpoint can be configured to fail closed with `REQUIRE_PERSISTENT_STORAGE=true`. A durable distributed queue/rate limiter is not appropriate to simulate on a single free Render instance; it requires a restored PostgreSQL/Redis-backed worker deployment and is therefore an explicit production infrastructure follow-up rather than an untrue completion claim.

Render Free remains suitable for a portfolio demo, but its documented idle spin-down and resource restrictions prevent an always-on production guarantee. Use a constrained demo mode if staying free: explicit cold-start/loading state, low concurrency, cached evidence with timestamps, and truthful unavailable states. Paid hosting will not itself create missing provider entitlements or make unverified claims correct.

## 4. Verification and release criteria

| Test layer | Required evidence |
| --- | --- |
| Parser/provider contracts | Sanitized fixtures for success, empty payloads, nested MCP errors, 401/403, 429, 5xx, timeout, missing fields and destination mismatch |
| Grounding and arithmetic | No false AVAILABLE records; budget totals/conversions reconcile; no asserted amenity/price/schedule without supporting evidence or an estimate label |
| Completion | All requested days and mandatory sections present; token exhaustion never marked complete; finalization preserves approval |
| PDF content and visual checks | Every page rendered and inspected, selectable text, all days present, long names/URLs, wide tables, non-Latin text, missing/slow images, no blank/clipped/overlapping content |
| Download/security | Correct owner/version, refresh/restart recovery, duplicate-click idempotency, expired link renewal, rejection of arbitrary remote image URLs |
| Deployment | Running SHA and assets match intended release, legacy exporter absent, cold/warm tests, low-concurrency load tests and failure recovery from Render |

Treat the previously passing 137 fast tests and one PostgreSQL test as a regression baseline, not proof of current production reliability. They did not prove that Render served the tested SHA, and the current PDF tests emphasize text/header validity over design and page completeness. Do not publish fresh success metrics until this release is actually tested.

Suggested initial measured targets, to ratify after a deployed baseline: all export fixtures pass content and visual checks; zero unsupported precise claims in the reviewed evaluation set; warm export P95 under five seconds for a seven-day plan with cached assets; no duplicate external calls for repeated export clicks. Provider availability should be reported separately from application success. No system can promise fresh live data while an upstream provider is unavailable.

## 5. Coding model assignments

These are models for doing the development work in Codex/Antigravity, not a proposal to send every traveler request to several commercial models. Availability must be checked in the relevant account. Recommendations are engineering choices, not a guarantee of perfect code.

| Work package | Primary model and setting | Alternate or independent reviewer | Why |
| --- | --- | --- | --- |
| Deployment audit and difficult cross-layer diagnosis | Codex GPT-6 Astra, High; XHigh only for unresolved architecture decisions | Antigravity Claude Opus 4.6 (Thinking) | Trace evidence across deployment, processes, schemas and state |
| HTTP/MCP adapters, retries, caches and diagnostics | Codex GPT-5.6 Sol, High | Claude Sonnet 4.6 (Thinking) | Bounded implementation with strong contract-test requirements |
| Canonical document, approval/versioning and factual validators | Codex GPT-6 Astra, High | Claude Opus 4.6 (Thinking) | Highest impact on consistency, truthfulness and migration safety |
| PDF visual direction and review of rendered pages | Antigravity Gemini 3.1 Pro, High | Codex GPT-6 Astra, High for content/visual verification | Compare whole pages, hierarchy, imagery and spacing |
| ReportLab components, pagination and asset handling | Codex GPT-5.6 Sol, High | Claude Sonnet 4.6 (Thinking) | Precise implementation of a reviewed visual specification |
| S3 delivery, session authorization and durable jobs | Codex GPT-5.6 Sol, High | Claude Opus 4.6 (Thinking) for authorization/recovery review | Keep storage, ownership and idempotency explicit |
| End-to-end regression and release review | Codex GPT-6 Astra, High | Gemini 3.1 Pro High for PDF QA; Opus 4.6 Thinking for backend review | Independent review against acceptance gates |
| README, runbook and small follow-up edits | Codex GPT-5.6 Sol, Medium | Claude Sonnet 4.6 (Thinking) if already working in Antigravity | Lower complexity does not justify maximum reasoning |

Use one implementation owner per package and hand reviewers the diff, fixtures and acceptance criteria. Avoid simultaneous edits by multiple coding tools to shared files. Do not assume Claude's Thinking option equals a Codex High/XHigh setting; use the controls exposed by Antigravity. For a simpler workflow, GPT-5.6 Sol High can implement the entire project with Astra used only for architecture and final review.

Runtime LLM choice is a separate decision: retain the existing configurable Groq models for the initial repair, log finish reasons and evaluate structured generation against the regression corpus. Upgrade or change a runtime model only if measured factual/formatting quality, cost and latency justify it. Provider retrieval, budget arithmetic and PDF rendering should not depend on a more powerful prose model to become correct.

## 6. Delivery order

1. Restore deployment parity and record a new known-version run.
2. Establish provider contracts and approved-document invariants.
3. Implement the professional PDF against deterministic document fixtures.
4. Integrate canonical web/PDF versions and authenticated downloads.
5. Add durable hosting/jobs and optional S3 artifacts, then complete deployed acceptance checks.

Visual design can start using fixtures while provider work proceeds, but production export must wait for the approved-document contract. Each phase ends with a reviewable commit and its acceptance evidence. Do not describe a GitHub push as a verified Render deployment.

## Sources

- Public TripBandhu homepage, served JavaScript and health endpoint, inspected 15 September 2026: https://tripbandhu.onrender.com/
- Tavily structured search response and error contracts: https://docs.tavily.com/documentation/api-reference/endpoint/search
- OpenWeather five-day/three-hour forecast and geocoding guidance: https://openweathermap.org/api/forecast5
- AviationStack documentation; endpoint access must be checked against the actual account: https://docs.apilayer.com/aviationstack/docs/api-documentation
- Render free service limits and production caveat: https://render.com/docs/free
- AWS S3 presigned downloads and bearer-token behavior: https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html
- OpenAI current model guidance: https://developers.openai.com/api/docs/guides/latest-model
- GPT-5.6 Sol and reasoning settings: https://developers.openai.com/api/docs/models/gpt-5.6-sol
- Antigravity model availability and selector settings: https://antigravity.google/docs/models
