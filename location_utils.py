"""Small, deterministic location normalizers shared by travel providers."""

from __future__ import annotations

import re


# Weather providers generally expect city names, not airport or metropolitan codes.
# Keep this map intentionally explicit so an arbitrary three-letter city name is not
# accidentally rewritten as an airport code.
_WEATHER_LOCATION_ALIASES = {
    "nyc": "New York City",
    "new york ny": "New York City",
    "new york city ny": "New York City",
    "del": "Delhi",
    "new delhi": "Delhi",
    "bom": "Mumbai",
    "blr": "Bengaluru",
    "bangalore": "Bengaluru",
    "maa": "Chennai",
    "ccu": "Kolkata",
    "hyd": "Hyderabad",
    "lax": "Los Angeles",
    "sfo": "San Francisco",
    "ord": "Chicago",
    "lon": "London",
    "par": "Paris",
    "tyo": "Tokyo",
    "dxb": "Dubai",
    "sin": "Singapore",
}


def normalize_weather_location(value: str | None) -> str:
    """Return a provider-friendly city name without guessing unknown codes."""
    location = re.sub(r"\s+", " ", str(value or "")).strip(" ,")
    if not location:
        return ""

    lookup_key = re.sub(r"[^a-z0-9]+", " ", location.casefold()).strip()
    return _WEATHER_LOCATION_ALIASES.get(lookup_key, location)
