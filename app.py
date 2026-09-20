import asyncio
import os
import re
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import backend
from backend import (
    resume_travel_agent,
    run_travel_agent,
    create_travel_service,
    DATABASE_URL,
)
from project_config import PROJECT_ROOT
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import InMemorySaver
from artifact_store import ArtifactStore, SESSION_COOKIE_NAME, new_session_id, sign_session_id, verify_signed_session
from itinerary_document import ItineraryValidationError, build_itinerary_document
from pdf_generator import PdfContentError, build_professional_itinerary_pdf
from deployment_info import APP_VERSION, PDF_RENDERER_VERSION, get_asset_version, get_build_sha, get_deployment_info
from mcp_client import close_provider_clients, start_provider_clients
from agent_config import (
    MAX_CONCURRENT_TRIPS,
    REQUIRE_PERSISTENT_STORAGE,
    SESSION_REQUEST_LIMIT,
    SESSION_REQUEST_WINDOW_SECONDS,
)


BASE_DIR = PROJECT_ROOT


# =========================
# Lifespan: Application-Scoped AsyncPostgresSaver & TravelAgentService
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the AsyncPostgresSaver and bind TravelAgentService to app.state."""
    checkpointer_cm = None
    await start_provider_clients()
    artifact_store = ArtifactStore(
        database_url=DATABASE_URL,
        request_limit=SESSION_REQUEST_LIMIT,
        request_window_seconds=SESSION_REQUEST_WINDOW_SECONDS,
    )
    await artifact_store.initialize()
    app.state.artifact_store = artifact_store
    app.state.trip_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRIPS)
    try:
        checkpointer_cm = AsyncPostgresSaver.from_conn_string(DATABASE_URL)
        checkpointer = await checkpointer_cm.__aenter__()
        await checkpointer.setup()
        app.state.travel_service = create_travel_service(checkpointer)
        app.state.checkpointer_backend = "postgresql"
        print("[lifespan] TravelAgentService initialized with AsyncPostgresSaver", flush=True)
    except Exception as exc:
        print(
            f"[lifespan] AsyncPostgresSaver initialization failed ({type(exc).__name__}). "
            "Falling back to InMemorySaver for application lifecycle.",
            flush=True,
        )
        app.state.travel_service = create_travel_service(InMemorySaver())
        app.state.checkpointer_backend = "memory"

    try:
        yield
    finally:
        if checkpointer_cm is not None:
            try:
                await checkpointer_cm.__aexit__(None, None, None)
                print("[lifespan] AsyncPostgresSaver closed cleanly.", flush=True)
            except Exception as close_exc:
                print(
                    f"[lifespan] Error closing AsyncPostgresSaver ({type(close_exc).__name__}).",
                    flush=True,
                )
        await close_provider_clients()
        await artifact_store.close()


app = FastAPI(
    title="TripBandhu - AI Travel Planner",
    description="LangGraph Multi-Agent Travel Planner with FastAPI Frontend",
    version=APP_VERSION,
    lifespan=lifespan,
)
# Provides a safe memory-only store for direct unit calls that do not enter the
# ASGI lifespan. Lifespan replaces it with the initialized application store.
app.state.artifact_store = ArtifactStore(
    request_limit=SESSION_REQUEST_LIMIT,
    request_window_seconds=SESSION_REQUEST_WINDOW_SECONDS,
)
app.state.trip_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRIPS)


@app.middleware("http")
async def add_build_identity_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-TripBandhu-Build"] = get_build_sha()
    return response


app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)


templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)


class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None


class TravelResumeRequest(BaseModel):
    thread_id: str
    approved: bool
    feedback: str = ""


PUBLIC_CONTRACT_DEFAULTS = {
    "thread_id": "",
    "answer": "",
    "requires_approval": False,
    "approval_request": "",
    "flight_results": "",
    "hotel_results": "",
    "weather_results": "",
    "budget_results": "",
    "itinerary": "",
    "selected_agents": [],
    "specialist_statuses": {},
    "trip_constraints": {},
    "supervisor_reasoning": "",
    "guardrail_allowed": True,
    "guardrail_reason": "",
    "approved": None,
    "human_feedback": "",
    "review_iteration": 0,
    "review_limit_reached": False,
    "capability_trace": [],
    "llm_calls": 0,
    "llm_token_usage": {},
    "run_status": "COMPLETED",
    "plan_id": "",
    "plan_version": None,
    "plan_content_hash": "",
    "export_status": "UNAVAILABLE",
    "export_error": "",
}


def normalize_travel_response(result: dict) -> dict:
    content = {"success": True}
    for field, default in PUBLIC_CONTRACT_DEFAULTS.items():
        content[field] = result.get(field, default)
    return content


def public_error(code: str, message: str, status_code: int = 500) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error_code": code,
            "error": message,
        },
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    asset_version = get_asset_version()
    response = templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"asset_version": asset_version},
    )
    response.headers["Cache-Control"] = "no-cache"
    return response


class PdfRequest(BaseModel):
    plan_id: str = Field(min_length=1, max_length=100)
    version: int = Field(default=1, ge=1, le=100)


_SMOKE_PLAN_ID = "__tripbandhu_pdf_smoke__"
_SMOKE_ITINERARY = """# TripBandhu PDF Renderer Check

### Day 1 - Arrival
- Morning: Verify the server-side PDF renderer.
- Afternoon: Confirm selectable text and pagination.
- Evening: Review the generated document.
"""


def _request_session(request: Request) -> tuple[str, bool]:
    cookies = getattr(request, "cookies", {})
    cookie_value = cookies.get(SESSION_COOKIE_NAME) if hasattr(cookies, "get") else None
    session_id = verify_signed_session(cookie_value if isinstance(cookie_value, str) else None)
    return (session_id, False) if session_id else (new_session_id(), True)


def _set_session_cookie(response: JSONResponse | Response, session_id: str, should_set: bool) -> None:
    if should_set:
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=sign_session_id(session_id),
            httponly=True,
            samesite="lax",
            secure=bool(os.getenv("RENDER")),
            max_age=60 * 60 * 24 * 14,
        )


def _artifact_store(request: Request) -> ArtifactStore | None:
    store = getattr(getattr(request, "app", None), "state", None)
    candidate = getattr(store, "artifact_store", None) if store is not None else None
    return candidate if isinstance(candidate, ArtifactStore) else None


def _safe_download_name(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(title).casefold()).strip("-")[:70]
    return f"{slug or 'tripbandhu-travel-plan'}.pdf"


async def _acquire_trip_slot(request: Request) -> bool:
    semaphore = getattr(request.app.state, "trip_semaphore", None)
    if not isinstance(semaphore, asyncio.Semaphore):
        return True
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=0.15)
        return True
    except TimeoutError:
        return False


@app.post("/api/travel/download-pdf")
async def download_pdf(request: Request, req: PdfRequest):
    """Stream the cached PDF of an approved, ownership-bound plan version."""
    try:
        # A fixed non-user plan keeps public deployment smoke tests from accepting
        # arbitrary browser-supplied text while proving the server renderer works.
        if req.plan_id == _SMOKE_PLAN_ID:
            smoke_document = build_itinerary_document(
                thread_id="deployment-smoke",
                markdown=_SMOKE_ITINERARY,
                constraints={"origin": "Delhi", "destination": "Jaipur", "duration": "1 day"},
                approved=True,
            )
            pdf_bytes = await run_in_threadpool(build_professional_itinerary_pdf, smoke_document)
            filename = "tripbandhu-renderer-check.pdf"
        else:
            store = _artifact_store(request)
            if store is None:
                return public_error("EXPORT_STORE_UNAVAILABLE", "Approved itinerary exports are initializing. Please retry.", 503)
            session_id, _ = _request_session(request)
            record, pdf_bytes = await store.get_or_create_pdf(
                session_id=session_id,
                plan_id=req.plan_id,
                version=req.version,
                renderer_version=PDF_RENDERER_VERSION,
                asset_version=get_asset_version(),
                renderer=build_professional_itinerary_pdf,
            )
            filename = _safe_download_name(record.document.title)
    except PermissionError:
        return public_error("PLAN_NOT_FOUND", "The approved itinerary is unavailable for this session.", 404)
    except ItineraryValidationError as exc:
        return public_error("INCOMPLETE_APPROVED_PLAN", str(exc), 409)
    except PdfContentError as exc:
        return public_error("INVALID_PDF_CONTENT", str(exc), 400)
    except Exception as exc:
        print(f"PDF_GENERATION_FAILED type={type(exc).__name__}", flush=True)
        return public_error("PDF_GENERATION_FAILED", "Could not generate PDF.", 500)

    return Response(content=pdf_bytes, media_type="application/pdf", headers={
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Cache-Control": "private, no-store",
        "X-Content-Type-Options": "nosniff",
    })




@app.post("/api/travel")
async def travel_planner(request: Request, request_data: TravelRequest):
    try:
        user_message = request_data.message.strip()

        if not user_message:
            return public_error(
                "INVALID_REQUEST",
                "Message cannot be empty.",
                status_code=400,
            )

        session_id, set_session = _request_session(request)
        store = _artifact_store(request)
        if store is not None:
            allowed, retry_after = await store.allow_request(session_id)
            if not allowed:
                response = public_error(
                    "REQUEST_RATE_LIMITED",
                    "Too many trip requests from this browser session. Please try again shortly.",
                    429,
                )
                response.headers["Retry-After"] = str(retry_after)
                _set_session_cookie(response, session_id, set_session)
                return response
        if request_data.thread_id and store is not None and not await store.owns_thread(session_id, request_data.thread_id):
            return public_error("THREAD_NOT_FOUND", "This travel plan is unavailable for this session.", 404)

        service = getattr(request.app.state, "travel_service", None)
        if not await _acquire_trip_slot(request):
            return public_error(
                "TRIP_CAPACITY_REACHED",
                "Trip planning is busy. Please retry in a moment.",
                503,
            )
        try:
            result = await run_travel_agent(
                user_input=user_message,
                thread_id=request_data.thread_id,
                service=service,
            )
        finally:
            semaphore = getattr(request.app.state, "trip_semaphore", None)
            if isinstance(semaphore, asyncio.Semaphore):
                semaphore.release()

        if store is not None and result.get("thread_id"):
            await store.bind_thread(session_id, result["thread_id"])
        response = JSONResponse(content=normalize_travel_response(result))
        _set_session_cookie(response, session_id, set_session)
        return response

    except Exception as exc:
        print(f"TRAVEL_PLANNING_FAILED type={type(exc).__name__}", flush=True)

        return public_error(
            "TRAVEL_PLANNING_FAILED",
            "Travel planning failed. Please try again in a moment.",
        )


@app.post("/api/travel/resume")
async def resume_travel_planner(request: Request, request_data: TravelResumeRequest):
    try:
        thread_id = request_data.thread_id.strip()
        feedback = request_data.feedback.strip()

        if not thread_id:
            return public_error(
                "INVALID_REQUEST",
                "thread_id is required to resume a travel plan.",
                status_code=400,
            )

        # Revision requires meaningful feedback
        if not request_data.approved and not feedback:
            return public_error(
                "REVISION_REQUIRES_FEEDBACK",
                "Please provide revision feedback when requesting changes.",
                status_code=400,
            )

        session_id, set_session = _request_session(request)
        store = _artifact_store(request)
        if store is not None and not await store.owns_thread(session_id, thread_id):
            return public_error("THREAD_NOT_FOUND", "This travel plan is unavailable for this session.", 404)

        service = getattr(request.app.state, "travel_service", None)

        result = await resume_travel_agent(
            thread_id=thread_id,
            approved=request_data.approved,
            feedback=feedback,
            service=service,
        )

        if request_data.approved and result.get("approved") and store is not None:
            try:
                document = build_itinerary_document(
                    thread_id=thread_id,
                    markdown=result.get("itinerary", ""),
                    constraints=result.get("trip_constraints", {}),
                    evidence_store=result.get("_evidence_store", {}),
                    budget_markdown=result.get("budget_results", ""),
                    version=max(1, int(result.get("review_iteration", 0)) + 1),
                    approved=True,
                )
                record = await store.save_approved(session_id, document)
                result.update(record.public_reference())
            except ItineraryValidationError as exc:
                result.update({
                    "export_status": "UNAVAILABLE",
                    "export_error": str(exc),
                })

        response = JSONResponse(content=normalize_travel_response(result))
        _set_session_cookie(response, session_id, set_session)
        return response

    except ValueError:
        return public_error(
            "INVALID_REQUEST",
            "The resume request is invalid.",
            status_code=400,
        )

    except Exception as exc:
        print(f"RESUME_FAILED type={type(exc).__name__}", flush=True)

        return public_error(
            "RESUME_FAILED",
            "Travel plan resume failed. Please try again in a moment.",
        )


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "AI Travel Planner API is running",
    }


@app.get("/health/live")
async def liveness_check():
    """No provider/database calls: suitable for process liveness checks."""
    return {"status": "ok", "service": "tripbandhu"}


@app.get("/health/ready")
async def readiness_check(request: Request):
    """Expose dependency mode without leaking connection strings or credentials."""
    store = _artifact_store(request)
    artifact_status = await store.readiness() if store is not None else {"ready": False, "backend": "unavailable", "persistent": False}
    checkpointer_backend = getattr(request.app.state, "checkpointer_backend", "memory")
    persistence_ready = artifact_status["persistent"] and checkpointer_backend == "postgresql"
    ready = artifact_status["ready"] and (persistence_ready or not REQUIRE_PERSISTENT_STORAGE)
    content = {
        "status": "ready" if ready else "degraded",
        "ready": ready,
        "checkpointer_backend": checkpointer_backend,
        "artifact_store": artifact_status,
        "persistent_storage": persistence_ready,
        "provider_config_version": get_deployment_info()["provider_config_version"],
    }
    return JSONResponse(content=content, status_code=200 if ready else 503, headers={"Cache-Control": "no-store"})


@app.get("/version")
async def version_info():
    """Expose non-secret build identity for release and deployment checks."""
    return JSONResponse(
        content=get_deployment_info(),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=int(os.getenv("PORT", "8080")),
        reload=True,
    )
