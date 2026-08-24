import os
import traceback
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from io import BytesIO

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


BASE_DIR = PROJECT_ROOT


# =========================
# Lifespan: Application-Scoped AsyncPostgresSaver & TravelAgentService
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the AsyncPostgresSaver and bind TravelAgentService to app.state."""
    checkpointer_cm = None
    try:
        checkpointer_cm = AsyncPostgresSaver.from_conn_string(DATABASE_URL)
        checkpointer = await checkpointer_cm.__aenter__()
        await checkpointer.setup()
        app.state.travel_service = create_travel_service(checkpointer)
        print("[lifespan] TravelAgentService initialized with AsyncPostgresSaver", flush=True)
    except Exception as exc:
        print(
            f"[lifespan] AsyncPostgresSaver initialization failed ({type(exc).__name__}: {exc}). "
            "Falling back to InMemorySaver for application lifecycle.",
            flush=True,
        )
        app.state.travel_service = create_travel_service(InMemorySaver())

    try:
        yield
    finally:
        if checkpointer_cm is not None:
            try:
                await checkpointer_cm.__aexit__(None, None, None)
                print("[lifespan] AsyncPostgresSaver closed cleanly.", flush=True)
            except Exception as close_exc:
                print(f"[lifespan] Error closing AsyncPostgresSaver: {close_exc}", flush=True)


app = FastAPI(
    title="TripBandhu - AI Travel Planner",
    description="LangGraph Multi-Agent Travel Planner with FastAPI Frontend",
    version="1.0.2",
    lifespan=lifespan,
)


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
    asset_version = (os.getenv("RENDER_GIT_COMMIT") or "dev")[:12]
    response = templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"asset_version": asset_version},
    )
    response.headers["Cache-Control"] = "no-cache"
    return response


class PdfRequest(BaseModel):
    text: str
    title: str = "TripBandhu Travel Plan"


@app.post("/api/travel/download-pdf")
async def download_pdf(req: PdfRequest):
    """
    Server-side PDF generation — same pattern as cortex-ai's PDFKit agent.
    Receives plain text from the frontend, renders a clean A4 PDF with
    reportlab, and returns it as a binary file download.
    No html2canvas, no browser rendering, no CSS bleed.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, HRFlowable
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER

        buf = BytesIO()

        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
        )

        styles = getSampleStyleSheet()

        # Custom styles — plain, readable, no decorative elements
        title_style = ParagraphStyle(
            "TBTitle",
            parent=styles["Normal"],
            fontSize=18,
            leading=24,
            textColor=colors.HexColor("#111827"),
            spaceAfter=4,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        )
        meta_style = ParagraphStyle(
            "TBMeta",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#6B7280"),
            spaceAfter=10,
            alignment=TA_CENTER,
        )
        h1_style = ParagraphStyle(
            "TBH1",
            parent=styles["Normal"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#003f29"),
            spaceBefore=10,
            spaceAfter=4,
            fontName="Helvetica-Bold",
        )
        h2_style = ParagraphStyle(
            "TBH2",
            parent=styles["Normal"],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#00643b"),
            spaceBefore=8,
            spaceAfter=3,
            fontName="Helvetica-Bold",
        )
        body_style = ParagraphStyle(
            "TBBody",
            parent=styles["Normal"],
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#374151"),
            spaceAfter=3,
        )
        bullet_style = ParagraphStyle(
            "TBBullet",
            parent=body_style,
            leftIndent=12,
            bulletIndent=0,
            spaceAfter=2,
        )
        footer_style = ParagraphStyle(
            "TBFooter",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#9CA3AF"),
            alignment=TA_CENTER,
        )

        from datetime import datetime
        story = []

        # Cover header
        story.append(Paragraph(req.title, title_style))
        story.append(Paragraph(
            f"Generated by TripBandhu · {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
            meta_style
        ))
        story.append(HRFlowable(width="100%", thickness=1,
                                 color=colors.HexColor("#e5e7eb"), spaceAfter=8))

        # Parse lines — detect headings and bullets, rest is body text
        for raw_line in req.text.split("\n"):
            line = raw_line.strip()
            if not line:
                story.append(Spacer(1, 4))
                continue

            # Strip markdown symbols
            clean = (line
                     .replace("**", "")
                     .replace("__", "")
                     .replace("`", "")
                     .replace("&", "&amp;")
                     .replace("<", "&lt;")
                     .replace(">", "&gt;"))

            if line.startswith("# "):
                story.append(Paragraph(clean[2:].strip(), h1_style))
            elif line.startswith("## "):
                story.append(Paragraph(clean[3:].strip(), h2_style))
            elif line.startswith("### "):
                story.append(Paragraph(clean[4:].strip(), h2_style))
            elif line.startswith(("- ", "* ", "• ")):
                story.append(Paragraph(f"• {clean[2:].strip()}", bullet_style))
            elif line.startswith(("---", "===")):
                story.append(HRFlowable(width="100%", thickness=0.5,
                                         color=colors.HexColor("#d1d5db"), spaceAfter=4))
            else:
                story.append(Paragraph(clean, body_style))

        # Footer
        story.append(Spacer(1, 12))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                 color=colors.HexColor("#e5e7eb"), spaceAfter=4))
        story.append(Paragraph("Generated by TripBandhu — AI Travel Research Assistant", footer_style))

        doc.build(story)
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="application/pdf",
            headers={
                "Content-Disposition": 'attachment; filename="tripbandhu-travel-plan.pdf"'
            },
        )

    except Exception as exc:
        import traceback as _tb
        print(f"[download_pdf] ERROR: {exc}\n{_tb.format_exc()}", flush=True)
        return public_error("PDF_GENERATION_FAILED", "Could not generate PDF.", 500)




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

        service = getattr(request.app.state, "travel_service", None)

        result = await run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id,
            service=service,
        )

        return JSONResponse(content=normalize_travel_response(result))

    except Exception:
        print("TRAVEL_PLANNING_FAILED")
        traceback.print_exc()

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

        service = getattr(request.app.state, "travel_service", None)

        result = await resume_travel_agent(
            thread_id=thread_id,
            approved=request_data.approved,
            feedback=feedback,
            service=service,
        )

        return JSONResponse(content=normalize_travel_response(result))

    except ValueError:
        return public_error(
            "INVALID_REQUEST",
            "The resume request is invalid.",
            status_code=400,
        )

    except Exception:
        print("RESUME_FAILED")
        traceback.print_exc()

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
