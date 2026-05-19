"""Tests for the health and readiness check endpoints (api/routers/health.py)."""

from contextlib import nullcontext
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from api import create_app


@pytest.fixture()
def client():
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    def test_liveness_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestReadyEndpoint:
    """Tests target the async /ready endpoint that uses get_async_db()."""

    @staticmethod
    def _async_cm(conn):
        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def _ctx():
            yield conn
        return _ctx

    @staticmethod
    def _make_async_conn(execute_side_effect=None):
        class _Cursor:
            async def fetchone(self): return (1,)
            async def close(self): pass
        class _Conn:
            async def execute(self, *a, **kw):
                if execute_side_effect is not None:
                    raise execute_side_effect
                return _Cursor()
        return _Conn()

    def test_ready_when_database_accessible(self, client):
        conn = self._make_async_conn()
        with mock.patch("api.routers.health.get_async_db", self._async_cm(conn)):
            resp = client.get("/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["checks"]["database"] == "ok"

    def test_not_ready_when_database_unavailable(self, client):
        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def _broken():
            raise Exception("connection refused")
            yield  # unreachable
        with mock.patch("api.routers.health.get_async_db", _broken):
            resp = client.get("/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"
        assert body["checks"]["database"] == "unavailable"

    def test_not_ready_when_query_fails(self, client):
        conn = self._make_async_conn(execute_side_effect=Exception("disk I/O error"))
        with mock.patch("api.routers.health.get_async_db", self._async_cm(conn)):
            resp = client.get("/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"
        assert body["checks"]["database"] == "unavailable"
