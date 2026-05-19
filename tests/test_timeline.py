"""Tests for the timeline endpoint (api/routers/timeline.py)."""

from contextlib import asynccontextmanager, contextmanager
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from api import create_app


def _cm(conn):
    @contextmanager
    def _ctx():
        yield conn
    return _ctx()


def _async_cm(conn):
    """Async context manager wrapping a mock connection — for get_async_db patches."""
    @asynccontextmanager
    async def _ctx():
        yield conn
    return _ctx


def _make_async_conn(fetchall_side_effects):
    """Build a mock that mimics aiosqlite.Connection.

    Each call to ``await conn.execute(...)`` returns a cursor whose
    ``await cursor.fetchall()`` returns the next item from ``fetchall_side_effects``.
    """
    class _Cursor:
        def __init__(self, rows):
            self._rows = rows

        async def fetchall(self):
            return self._rows

        async def fetchone(self):
            return self._rows[0] if self._rows else None

        async def close(self):
            pass

    rows_iter = iter(fetchall_side_effects)

    class _Conn:
        async def execute(self, *args, **kwargs):
            try:
                return _Cursor(next(rows_iter))
            except StopIteration:
                return _Cursor([])

    return _Conn()


@pytest.fixture()
def client():
    app = create_app()
    return TestClient(app)


class TestTimelineEndpoint:
    """Tests for GET /api/timeline."""

    def test_returns_date_groups(self, client):
        """Returns photos grouped by date."""
        date_rows = [
            {"photo_date": "2025-03-10", "cnt": 5},
            {"photo_date": "2025-03-09", "cnt": 3},
        ]
        # The async endpoint runs two queries: dates, then one big photo query
        # using ROW_NUMBER() (not one per date as the old sync code suggested).
        photo_rows = [
            {"path": "/a.jpg", "date_taken": "2025:03:10 14:00:00", "aggregate": 8.5, "tags": "landscape", "filename": "a.jpg", "_photo_date": "2025-03-10", "_rn": 1},
            {"path": "/b.jpg", "date_taken": "2025:03:09 10:00:00", "aggregate": 7.0, "tags": "portrait", "filename": "b.jpg", "_photo_date": "2025-03-09", "_rn": 1},
        ]

        async_conn = _make_async_conn([date_rows, photo_rows])

        async def _no_op_attach(*args, **kwargs):
            return None

        with (
            mock.patch("api.routers.timeline.get_async_db", _async_cm(async_conn)),
            mock.patch("api.routers.timeline.get_visibility_clause", return_value=("1=1", [])),
            mock.patch("api.routers.timeline.get_photos_from_clause", return_value=("photos", [])),
            mock.patch("api.routers.timeline.build_photo_select_columns", return_value=["path", "date_taken", "aggregate", "tags"]),
            mock.patch("api.routers.timeline.VIEWER_CONFIG", {"display": {"tags_per_photo": 10}}),
            mock.patch("api.routers.timeline.split_photo_tags", side_effect=lambda rows, limit: [dict(r) for r in rows]),
            mock.patch("api.routers.timeline.attach_person_data_async", _no_op_attach),
            mock.patch("api.routers.timeline.sanitize_float_values"),
            mock.patch("api.routers.timeline.format_date", return_value="10/03/2025"),
        ):
            resp = client.get("/api/timeline", params={"limit": 50})

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["groups"]) == 2
        assert body["groups"][0]["date"] == "2025-03-10"
        assert body["groups"][0]["count"] == 5
        assert body["has_more"] is False


    def test_cursor_pagination(self, client):
        """Cursor parameter filters dates before/after the cursor."""
        async_conn = _make_async_conn([[]])  # no date_rows after cursor

        with (
            mock.patch("api.routers.timeline.get_async_db", _async_cm(async_conn)),
            mock.patch("api.routers.timeline.get_visibility_clause", return_value=("1=1", [])),
            mock.patch("api.routers.timeline.get_photos_from_clause", return_value=("photos", [])),
            mock.patch("api.routers.timeline.build_photo_select_columns", return_value=["path"]),
            mock.patch("api.routers.timeline.VIEWER_CONFIG", {"display": {"tags_per_photo": 10}}),
        ):
            resp = client.get("/api/timeline", params={
                "cursor": "2025-03-10",
                "direction": "older",
                "limit": 10,
            })

        assert resp.status_code == 200
        body = resp.json()
        assert body["groups"] == []
        assert body["has_more"] is False

    def test_has_more_when_extra_dates(self, client):
        """has_more is True when more dates exist beyond the limit."""
        # Return limit+1 rows to trigger has_more
        date_rows = [{"photo_date": f"2025-03-{10-i:02d}", "cnt": 1} for i in range(4)]
        photo_rows = [
            {"path": f"/{i}.jpg", "date_taken": f"2025:03:{10-i:02d} 10:00:00",
             "aggregate": 5.0, "tags": "", "filename": f"{i}.jpg",
             "_photo_date": f"2025-03-{10-i:02d}", "_rn": 1}
            for i in range(3)
        ]

        async_conn = _make_async_conn([date_rows, photo_rows])

        async def _no_op_attach(*args, **kwargs):
            return None

        with (
            mock.patch("api.routers.timeline.get_async_db", _async_cm(async_conn)),
            mock.patch("api.routers.timeline.get_visibility_clause", return_value=("1=1", [])),
            mock.patch("api.routers.timeline.get_photos_from_clause", return_value=("photos", [])),
            mock.patch("api.routers.timeline.build_photo_select_columns", return_value=["path", "date_taken", "aggregate", "tags"]),
            mock.patch("api.routers.timeline.VIEWER_CONFIG", {"display": {"tags_per_photo": 10}}),
            mock.patch("api.routers.timeline.split_photo_tags", side_effect=lambda rows, limit: [dict(r) for r in rows]),
            mock.patch("api.routers.timeline.attach_person_data_async", _no_op_attach),
            mock.patch("api.routers.timeline.sanitize_float_values"),
            mock.patch("api.routers.timeline.format_date", return_value=""),
        ):
            resp = client.get("/api/timeline", params={"limit": 3})

        assert resp.status_code == 200
        body = resp.json()
        assert body["has_more"] is True
        assert body["next_cursor"] is not None
        assert len(body["groups"]) == 3

    def test_date_from_and_date_to_filtering(self, client):
        """date_from and date_to parameters filter the results."""
        async_conn = _make_async_conn([
            [{"photo_date": "2025-03-12", "cnt": 2}],
            [{"path": "/x.jpg", "date_taken": "2025:03:12 10:00:00", "aggregate": 6.0, "tags": "", "filename": "x.jpg", "_photo_date": "2025-03-12", "_rn": 1}],
        ])

        async def _no_op_attach(*args, **kwargs):
            return None

        with (
            mock.patch("api.routers.timeline.get_async_db", _async_cm(async_conn)),
            mock.patch("api.routers.timeline.get_visibility_clause", return_value=("1=1", [])),
            mock.patch("api.routers.timeline.get_photos_from_clause", return_value=("photos", [])),
            mock.patch("api.routers.timeline.build_photo_select_columns", return_value=["path", "date_taken", "aggregate", "tags"]),
            mock.patch("api.routers.timeline.VIEWER_CONFIG", {"display": {"tags_per_photo": 10}}),
            mock.patch("api.routers.timeline.split_photo_tags", side_effect=lambda rows, limit: [dict(r) for r in rows]),
            mock.patch("api.routers.timeline.attach_person_data_async", _no_op_attach),
            mock.patch("api.routers.timeline.sanitize_float_values"),
            mock.patch("api.routers.timeline.format_date", return_value="12/03/2025"),
        ):
            resp = client.get("/api/timeline", params={
                "date_from": "2025-03-10",
                "date_to": "2025-03-15",
            })

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["groups"]) == 1
        assert body["groups"][0]["date"] == "2025-03-12"

    def test_newer_direction(self, client):
        """direction=newer fetches dates after the cursor."""
        async_conn = _make_async_conn([
            [{"photo_date": "2025-03-15", "cnt": 1}],
            [{"path": "/n.jpg", "date_taken": "2025:03:15 10:00:00", "aggregate": 6.0, "tags": "", "filename": "n.jpg", "_photo_date": "2025-03-15", "_rn": 1}],
        ])

        async def _no_op_attach(*args, **kwargs):
            return None

        with (
            mock.patch("api.routers.timeline.get_async_db", _async_cm(async_conn)),
            mock.patch("api.routers.timeline.get_visibility_clause", return_value=("1=1", [])),
            mock.patch("api.routers.timeline.get_photos_from_clause", return_value=("photos", [])),
            mock.patch("api.routers.timeline.build_photo_select_columns", return_value=["path", "date_taken", "aggregate", "tags"]),
            mock.patch("api.routers.timeline.VIEWER_CONFIG", {"display": {"tags_per_photo": 10}}),
            mock.patch("api.routers.timeline.split_photo_tags", side_effect=lambda rows, limit: [dict(r) for r in rows]),
            mock.patch("api.routers.timeline.attach_person_data_async", _no_op_attach),
            mock.patch("api.routers.timeline.sanitize_float_values"),
            mock.patch("api.routers.timeline.format_date", return_value="15/03/2025"),
        ):
            resp = client.get("/api/timeline", params={
                "cursor": "2025-03-10",
                "direction": "newer",
            })

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["groups"]) == 1

    def test_db_error_returns_empty(self):
        """On database exception, returns empty result instead of 500."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        class _BrokenConn:
            async def execute(self, *a, **kw):
                raise Exception("DB error")

        with (
            mock.patch("api.routers.timeline.get_async_db", _async_cm(_BrokenConn())),
            mock.patch("api.routers.timeline.get_visibility_clause", return_value=("1=1", [])),
            mock.patch("api.routers.timeline.get_photos_from_clause", return_value=("photos", [])),
        ):
            resp = client.get("/api/timeline")

        assert resp.status_code == 200
        body = resp.json()
        assert body["groups"] == []
        assert body["has_more"] is False


class TestTimelineDates:
    """Tests for GET /api/timeline/dates."""

    def test_returns_date_counts(self, client):
        async_conn = _make_async_conn([[
            {"group_key": "2025-03-10", "cnt": 15, "hero_photo_path": "/photos/a.jpg"},
            {"group_key": "2025-03-11", "cnt": 8, "hero_photo_path": "/photos/b.jpg"},
        ]])

        with (
            mock.patch("api.routers.timeline.get_async_db", _async_cm(async_conn)),
            mock.patch("api.routers.timeline.get_visibility_clause", return_value=("1=1", [])),
            mock.patch("api.routers.timeline.get_photos_from_clause", return_value=("photos", [])),
        ):
            resp = client.get("/api/timeline/dates", params={"year": 2025})

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["dates"]) == 2
        assert body["dates"][0]["date"] == "2025-03-10"
        assert body["dates"][0]["count"] == 15


    def test_year_and_month_filter(self, client):
        async_conn = _make_async_conn([[
            {"group_key": "2025-06-15", "cnt": 3, "hero_photo_path": "/photos/c.jpg"},
        ]])

        with (
            mock.patch("api.routers.timeline.get_async_db", _async_cm(async_conn)),
            mock.patch("api.routers.timeline.get_visibility_clause", return_value=("1=1", [])),
            mock.patch("api.routers.timeline.get_photos_from_clause", return_value=("photos", [])),
        ):
            resp = client.get("/api/timeline/dates", params={"year": 2025, "month": 6})

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["dates"]) == 1

    def test_db_error_returns_empty_dates(self):
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        class _BrokenConn:
            async def execute(self, *a, **kw):
                raise Exception("DB error")

        with (
            mock.patch("api.routers.timeline.get_async_db", _async_cm(_BrokenConn())),
            mock.patch("api.routers.timeline.get_visibility_clause", return_value=("1=1", [])),
            mock.patch("api.routers.timeline.get_photos_from_clause", return_value=("photos", [])),
        ):
            resp = client.get("/api/timeline/dates", params={"year": 2025})

        assert resp.status_code == 200
        assert resp.json()["dates"] == []

    def test_missing_year_returns_422(self, client):
        resp = client.get("/api/timeline/dates")
        assert resp.status_code == 422
