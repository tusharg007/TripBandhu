import os
from pathlib import Path
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP
from location_utils import normalize_weather_location
from project_config import PROJECT_ROOT, load_project_env


BASE_DIR = PROJECT_ROOT
load_project_env()

mcp = FastMCP("Weather MCP Server")

OPENWEATHER_API_KEY = (
    os.getenv("OPENWEATHER_API_KEY")
    or os.getenv("OPENWEATHER_KEY")
)

REQUEST_TIMEOUT_SECONDS = 20


def _get_api_key() -> str:
    if not OPENWEATHER_API_KEY:
        raise RuntimeError(
            "OPENWEATHER_API_KEY is missing "
            "from the project .env file."
        )

    return OPENWEATHER_API_KEY


def _request_json(
    url: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        return response.json()

    except requests.Timeout as exc:
        raise RuntimeError("The weather provider timed out. Please try again shortly.") from exc
    except requests.RequestException as exc:
        failed_response = getattr(exc, "response", None)
        status_code = getattr(failed_response, "status_code", None)
        provider_message = ""
        if failed_response is not None:
            try:
                provider_message = str(failed_response.json().get("message") or "").strip()
            except (ValueError, AttributeError):
                provider_message = ""

        if status_code == 404:
            message = "The weather provider could not find that location."
        elif status_code in {401, 403}:
            message = "Weather access is not configured correctly."
        elif status_code == 429:
            message = "The weather provider is temporarily rate-limited."
        elif status_code and status_code >= 500:
            message = "The weather provider is temporarily unavailable."
        else:
            message = "The weather provider request failed."

        # Never include the requests exception string: it contains the full request
        # URL and therefore the OpenWeather API key from the query string.
        if provider_message and status_code not in {401, 403}:
            message = f"{message} Provider message: {provider_message}."
        raise RuntimeError(message) from exc


@mcp.tool()
def get_current_weather(
    city: str,
) -> dict[str, Any]:
    """Return the current weather for a city."""

    city = normalize_weather_location(city)

    if not city:
        raise ValueError(
            "city cannot be empty"
        )

    data = _request_json(
        "https://api.openweathermap.org/data/2.5/weather",
        {
            "q": city,
            "appid": _get_api_key(),
            "units": "metric",
        },
    )

    return {
        "city": data["name"],
        "temperature_c": data["main"]["temp"],
        "feels_like_c": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"],
    }


@mcp.tool()
def get_forecast(
    city: str,
) -> dict[str, Any]:
    """
    Return the first five three-hour
    forecast entries for a city.
    """

    city = normalize_weather_location(city)

    if not city:
        raise ValueError(
            "city cannot be empty"
        )

    data = _request_json(
        "https://api.openweathermap.org/data/2.5/forecast",
        {
            "q": city,
            "appid": _get_api_key(),
            "units": "metric",
        },
    )

    forecast = [
        {
            "datetime": item["dt_txt"],
            "temperature_c": item["main"]["temp"],
            "condition": item["weather"][0]["description"],
        }
        for item in data.get("list", [])[:5]
    ]

    return {
        "city": data.get(
            "city",
            {},
        ).get(
            "name",
            city,
        ),
        "forecast": forecast,
    }


if __name__ == "__main__":
    # mcp_client.py launches this as a stdio subprocess.
    mcp.run(
        transport="stdio",
    )
