"""Deployment identity and legacy-export regression tests."""

from __future__ import annotations

from io import BytesIO
import asyncio
import os
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from pypdf import PdfReader

import app
import backend
from agent_config import PROVIDER_TIMEOUT_SECONDS


def test_version_exposes_safe_build_contract():
    commit = "a" * 40
    with patch.dict(os.environ, {"RENDER": "true", "RENDER_GIT_COMMIT": commit}):
        response = TestClient(app.app).get("/version")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-tripbandhu-build"] == commit
    assert response.json() == {
        "app_version": "1.0.3",
        "build_sha": commit,
        "asset_version": commit[:12],
        "pdf_renderer": "reportlab-v1",
        "provider_config_version": "provider-timeouts-v2",
        "public_schema_version": "1",
        "environment": "render",
    }


def test_home_uses_build_version_and_has_no_browser_pdf_exporter():
    commit = "b" * 40
    with patch.dict(os.environ, {"RENDER_GIT_COMMIT": commit}):
        response = TestClient(app.app).get("/")

    assert response.status_code == 200
    assert response.headers["x-tripbandhu-build"] == commit
    assert f'/static/style.css?v={commit[:12]}' in response.text
    assert f'/static/script.js?v={commit[:12]}' in response.text
    assert "html2pdf" not in response.text.casefold()
    assert "jspdf" not in response.text.casefold()


def test_download_identifies_reportlab_and_contains_selectable_text():
    commit = "c" * 40
    with patch.dict(os.environ, {"RENDER_GIT_COMMIT": commit}):
        response = TestClient(app.app).post(
            "/api/travel/download-pdf",
            json={
                "title": "Deployment PDF Check",
                "text": "# Deployment PDF Check\n\n## Day 1\n- Arrive in Delhi.",
            },
        )

    reader = PdfReader(BytesIO(response.content))
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert response.status_code == 200
    assert response.headers["x-tripbandhu-build"] == commit
    assert "ReportLab" in str(reader.metadata.producer)
    assert "Arrive in Delhi" in extracted
    assert all((page.extract_text() or "").strip() for page in reader.pages)


def test_graph_run_metadata_identifies_build_and_provider_configuration():
    commit = "d" * 40
    graph = AsyncMock()
    graph.ainvoke.return_value = {}
    service = backend.TravelAgentService(graph)

    with patch.dict(os.environ, {"RENDER_GIT_COMMIT": commit}):
        asyncio.run(service.run("Plan a weekend in Jaipur", thread_id="phase0-check"))

    config = graph.ainvoke.await_args.kwargs["config"]
    assert config["metadata"] == {
        "app_version": "1.0.3",
        "build_sha": commit,
        "provider_config_version": "provider-timeouts-v2",
        "provider_timeouts_seconds": PROVIDER_TIMEOUT_SECONDS,
    }
