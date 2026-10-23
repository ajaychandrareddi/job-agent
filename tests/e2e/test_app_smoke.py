from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from job_agent.config import get_settings
from job_agent.main import create_app


def test_health_and_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    get_settings.cache_clear()
    try:
        client = TestClient(create_app())
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").json() == {"status": "ready"}
    finally:
        get_settings.cache_clear()
