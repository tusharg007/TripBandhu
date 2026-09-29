import asyncio
import os
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import app
from artifact_store import ArtifactStore


def test_liveness_does_not_require_database():
    response = TestClient(app.app).get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "tripbandhu"}


def test_required_persistence_reports_not_ready_for_memory_backends():
    old_store = app.app.state.artifact_store
    old_backend = getattr(app.app.state, "checkpointer_backend", None)
    app.app.state.artifact_store = ArtifactStore()
    app.app.state.checkpointer_backend = "memory"
    try:
        with patch.object(app, "REQUIRE_PERSISTENT_STORAGE", True), patch.dict(os.environ, {"RENDER": "false"}):
            response = TestClient(app.app).get("/health/ready")
        assert response.status_code == 503
        payload = response.json()
        assert payload["ready"] is False
        assert payload["persistent_storage"] is False
        assert "password" not in response.text.casefold()
        assert "postgresql://" not in response.text.casefold()
    finally:
        app.app.state.artifact_store = old_store
        if old_backend is None:
            try:
                delattr(app.app.state, "checkpointer_backend")
            except AttributeError:
                pass
        else:
            app.app.state.checkpointer_backend = old_backend


def test_persistent_backends_report_ready_when_external_session_secret_is_configured():
    old_store = app.app.state.artifact_store
    old_backend = getattr(app.app.state, "checkpointer_backend", None)
    store = ArtifactStore()
    store.backend = "postgresql"
    store.persistent = True
    app.app.state.artifact_store = store
    app.app.state.checkpointer_backend = "postgresql"
    try:
        with patch.object(app, "REQUIRE_PERSISTENT_STORAGE", True), \
             patch.object(app, "database_pool_ready", AsyncMock(return_value=True)), \
             patch("artifact_store.database_pool_ready", AsyncMock(return_value=True)), \
             patch.object(app, "session_security_status", return_value={"configured": True, "source": "external"}), \
             patch.dict(os.environ, {"RENDER": "true"}):
            response = TestClient(app.app).get("/health/ready")
        assert response.status_code == 200
        payload = response.json()
        assert payload["ready"] is True
        assert payload["persistent_storage"] is True
        assert payload["session_security"] == {"configured": True, "source": "external"}
    finally:
        app.app.state.artifact_store = old_store
        if old_backend is None:
            try:
                delattr(app.app.state, "checkpointer_backend")
            except AttributeError:
                pass
        else:
            app.app.state.checkpointer_backend = old_backend


def test_dead_checkpoint_database_is_not_ready_even_with_optional_persistence():
    with patch.object(app.app.state, "checkpointer_backend", "postgresql", create=True), \
         patch.object(app, "database_pool_ready", AsyncMock(return_value=False)), \
         patch.object(app, "REQUIRE_PERSISTENT_STORAGE", False), \
         patch.object(app, "session_security_status", return_value={"configured": True, "source": "external"}):
        response = TestClient(app.app).get("/health/ready")
        assert response.status_code == 503
        assert response.json()["checkpointer_ready"] is False
        assert TestClient(app.app).get("/health/live").status_code == 200


def test_dead_artifact_database_is_not_ready():
    store = ArtifactStore()
    store.backend = "postgresql"
    store.persistent = True
    with patch("artifact_store.database_pool_ready", AsyncMock(return_value=False)):
        assert asyncio.run(store.readiness())["ready"] is False
