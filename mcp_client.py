import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timezone

import certifi
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from project_config import PROJECT_ROOT, load_project_env
from agent_config import GROQ_CONTROL_MODEL, LLM_REASONING_EFFORT, LLM_REASONING_FORMAT, MAX_TOKENS_BY_TASK

from schemas import (
    CapabilityHealth,
    CapabilityTraceEntry,
    CapabilityEvidence,
    EvidenceKind,
    ErrorCode,
)
from capability_registry import (
    CAPABILITY_REGISTRY,
    SPECIALIST_ALLOWED_CAPABILITIES,
    validate_specialist_capability,
    normalize_tavily_results,
    normalize_weather_results,
    normalize_flight_results,
)


# =========================================================
# Environment setup
# =========================================================

BASE_DIR = PROJECT_ROOT
load_project_env()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

AVIATION_STACK_API_KEY = (
    os.getenv("AVIATION_STACK_API_KEY")
    or os.getenv("AVIATIONSTACK_API_KEY")
    or os.getenv("AVIATIONSTACK_KEY")
)

OPENWEATHER_API_KEY = (
    os.getenv("OPENWEATHER_API_KEY")
    or os.getenv("OPENWEATHER_KEY")
)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

WEATHER_SERVER_PATH = BASE_DIR / "custom_weather_mcp_server.py"
UVX_COMMAND = shutil.which("uvx") or "uvx"

# TF2: Pinned AviationStack MCP Version
PINNED_AVIATIONSTACK_VERSION = "aviationstack-mcp==1.6.0"


def _require_env(name: str, value: str | None) -> str:
    if not value:
        raise RuntimeError(
            f"{name} is missing. "
            f"Add {name}=your_key to the project .env file."
        )
    return value


def _subprocess_env(**updates: str | None) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in updates.items():
        if value:
            env[key] = value
    return env


# =========================================================
# LLM Factory
# =========================================================

def make_groq_llm(
    model_id: str,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    reasoning_format: str | None = None,
    **extra_kwargs,
) -> ChatGroq:
    """
    Factory for ChatGroq instances used across backend and mcp_client.

    All ChatGroq instances MUST be created through this function so that:
    - model ID, api_key, and max_retries are never silently inconsistent.
    - max_retries=0 ensures the Groq SDK / LangChain client does NOT perform
      hidden automatic 429 retries. Our llm_utils.invoke_llm() owns all retries.
    """
    kwargs: dict[str, Any] = dict(
        model=model_id,
        api_key=GROQ_API_KEY or "dummy_ci_key",
        max_retries=0,
        **extra_kwargs,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if reasoning_format is not None:
        kwargs["reasoning_format"] = reasoning_format
    return ChatGroq(**kwargs)


# Destination extractor LLM — control model, tiny token budget (city name only).
# Used only when supervisor destination is absent (normal trips skip this call).
llm = make_groq_llm(
    GROQ_CONTROL_MODEL,
    max_tokens=MAX_TOKENS_BY_TASK["destination_extract"],
    reasoning_effort=LLM_REASONING_EFFORT,
    reasoning_format=LLM_REASONING_FORMAT,
)


# =========================================================
# MCP Client (Reusable MultiServer Client)
# =========================================================

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": (
                "https://mcp.tavily.com/mcp/"
                f"?tavilyApiKey={TAVILY_API_KEY or ''}"
            ),
        },

        "aviationstack": {
            "transport": "stdio",
            "command": UVX_COMMAND,
            "args": [
                PINNED_AVIATIONSTACK_VERSION,
            ],
            # Pass all common env var name variants — the aviationstack-mcp package
            # uses "access_key" in its API calls and may look for any of these names.
            "env": _subprocess_env(
                AVIATION_STACK_API_KEY=AVIATION_STACK_API_KEY,
                AVIATION_STACK_ACCESS_KEY=AVIATION_STACK_API_KEY,
                AVIATIONSTACK_API_KEY=AVIATION_STACK_API_KEY,
                AVIATIONSTACK_ACCESS_KEY=AVIATION_STACK_API_KEY,
                PYTHONUNBUFFERED="1",
            ),
        },

        "weather": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [
                str(WEATHER_SERVER_PATH),
            ],
            "env": _subprocess_env(
                OPENWEATHER_API_KEY=OPENWEATHER_API_KEY,
            ),
        },
    }
)


# =========================================================
# Tool Discovery & Schema Drift Detection
# =========================================================

async def discover_mcp_tools(server_name: str) -> list[Any]:
    """Discover available tools from a connected MCP server."""
    if server_name == "tavily":
        _require_env("TAVILY_API_KEY", TAVILY_API_KEY)
    elif server_name == "aviationstack":
        _require_env("AVIATION_STACK_API_KEY", AVIATION_STACK_API_KEY)
        if shutil.which("uvx") is None:
            raise RuntimeError("uvx was not found in PATH.")
    elif server_name == "weather":
        _require_env("OPENWEATHER_API_KEY", OPENWEATHER_API_KEY)
        if not WEATHER_SERVER_PATH.is_file():
            raise FileNotFoundError(f"Weather MCP server not found: {WEATHER_SERVER_PATH}")

    return await client.get_tools(server_name=server_name)


async def validate_mcp_capability_health() -> dict[str, CapabilityHealth]:
    """Inspect all registered capabilities against runtime MCP servers to detect schema drift."""
    health_status: dict[str, CapabilityHealth] = {}

    for cap_name, cap_def in CAPABILITY_REGISTRY.items():
        try:
            tools = await discover_mcp_tools(cap_def.server)
            tool = next((t for t in tools if t.name == cap_def.tool_name), None)
            if tool is None:
                health_status[cap_name] = CapabilityHealth.CONTRACT_MISMATCH
            else:
                health_status[cap_name] = CapabilityHealth.AVAILABLE
        except Exception:
            health_status[cap_name] = CapabilityHealth.UNAVAILABLE

    return health_status


async def _get_server_tool(server_name: str, tool_name: str):
    """Load a specific tool from an MCP server, raising on missing tools."""
    tools = await discover_mcp_tools(server_name)

    tool = next((item for item in tools if item.name == tool_name), None)

    if tool is None:
        available_tools = ", ".join(sorted(item.name for item in tools)) or "none"
        raise RuntimeError(
            f"MCP tool '{tool_name}' was not found on server '{server_name}'. "
            f"Available tools: {available_tools}"
        )

    return tool


# =========================================================
# Low-Level Tool Invocation Adapters
# =========================================================

async def tavily_mcp_search(query: str):
    search_tool = await _get_server_tool("tavily", "tavily_search")
    return await search_tool.ainvoke({"query": query})


async def aviation_mcp_call(tool_name: str, tool_args: dict[str, Any] | None = None):
    aviation_tool = await _get_server_tool("aviationstack", tool_name)
    return await aviation_tool.ainvoke(tool_args or {})


async def weather_mcp_search(city: str):
    weather_tool = await _get_server_tool("weather", "get_current_weather")
    return await weather_tool.ainvoke({"city": city})


async def forecast_mcp_search(city: str):
    forecast_tool = await _get_server_tool("weather", "get_forecast")
    return await forecast_tool.ainvoke({"city": city})


# =========================================================
# Destination Extractor
# =========================================================

def extract_destination(query: str) -> str:
    prompt = (
        "Extract only the destination city or country from the travel request.\n\n"
        f"Travel request:\n{query}\n\n"
        "Return only the destination name.\nDo not add any explanation."
    )
    response = llm.invoke(prompt)
    destination = str(response.content).strip()
    if not destination:
        raise ValueError("The destination could not be extracted.")
    return destination


async def extract_destination_async(query: str) -> str:
    prompt = (
        "Extract only the destination city or country from the travel request.\n\n"
        f"Travel request:\n{query}\n\n"
        "Return only the destination name.\nDo not add any explanation."
    )
    response = await llm.ainvoke(prompt)
    destination = str(response.content).strip()
    if not destination:
        raise ValueError("The destination could not be extracted.")
    return destination
