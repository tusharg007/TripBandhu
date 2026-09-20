"""Public, non-secret build metadata used for deployment verification."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from project_config import PROJECT_ROOT


APP_VERSION = "1.1.0"
PUBLIC_SCHEMA_VERSION = "1"
PDF_RENDERER_VERSION = "reportlab-v1"
PROVIDER_CONFIG_VERSION = "direct-provider-adapters-v1"

_BUILD_ENV_NAMES = (
    "RENDER_GIT_COMMIT",
    "GITHUB_SHA",
    "SOURCE_VERSION",
    "GIT_COMMIT",
    "COMMIT_SHA",
)
_SAFE_BUILD_VALUE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def get_build_sha() -> str:
    """Return a safe deployment revision without exposing environment details."""
    for name in _BUILD_ENV_NAMES:
        value = str(os.getenv(name) or "").strip()
        if _SAFE_BUILD_VALUE.fullmatch(value):
            return value
    return "dev"


def _static_asset_digest() -> str:
    digest = hashlib.sha256()
    for relative_path in ("static/style.css", "static/script.js"):
        path = Path(PROJECT_ROOT, relative_path)
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def get_asset_version() -> str:
    build_sha = get_build_sha()
    return build_sha[:12] if build_sha != "dev" else _static_asset_digest()


def get_deployment_info() -> dict[str, str]:
    """Return the public build contract consumed by smoke checks."""
    return {
        "app_version": APP_VERSION,
        "build_sha": get_build_sha(),
        "asset_version": get_asset_version(),
        "pdf_renderer": PDF_RENDERER_VERSION,
        "provider_config_version": PROVIDER_CONFIG_VERSION,
        "public_schema_version": PUBLIC_SCHEMA_VERSION,
        "environment": "render" if os.getenv("RENDER") else "development",
    }
