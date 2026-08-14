import traceback
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend import resume_travel_agent, run_travel_agent
from project_config import PROJECT_ROOT

# This is to allow nested event loops for async calls in FastAPI
import nest_asyncio
nest_asyncio.apply()


BASE_DIR = PROJECT_ROOT

app = FastAPI(
    title="TripBandhu - AI Travel Planner",
    description="LangGraph Multi-Agent Travel Planner with FastAPI Frontend",
    version="1.0.0"
)


app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
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
    "llm_calls": 0,
}


def normalize_travel_response(result: dict) -> dict:
    content = {
        "success": True,
    }

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
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
    try:
        user_message = request_data.message.strip()

        if not user_message:
            return public_error(
                "INVALID_REQUEST",
                "Message cannot be empty.",
                status_code=400,
            )

        result = run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id
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
async def resume_travel_planner(request_data: TravelResumeRequest):
    try:
        thread_id = request_data.thread_id.strip()
        feedback = request_data.feedback.strip()

        if not thread_id:
            return public_error(
                "INVALID_REQUEST",
                "thread_id is required to resume a travel plan.",
                status_code=400,
            )

        result = resume_travel_agent(
            thread_id=thread_id,
            approved=request_data.approved,
            feedback=feedback,
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
        "message": "AI Travel Planner API is running"
    }


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})



if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
