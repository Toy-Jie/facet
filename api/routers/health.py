"""
Health and readiness check endpoints.

Provides /health (liveness) and /ready (readiness) for orchestrators
and load balancers.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse

from api.database import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    """Liveness check — confirms the process is running."""
    return {"status": "ok"}


@router.get("/ready")
def ready():
    """Readiness check — verifies the database is accessible."""
    checks = {}
    try:
        with get_db() as conn:
            conn.execute("SELECT 1")
            checks["database"] = "ok"
    except Exception:
        checks["database"] = "unavailable"
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "checks": checks},
        )

    return {"status": "ready", "checks": checks}


@router.get("/metrics")
def metrics():
    """Prometheus-style metrics endpoint.

    Returns text in Prometheus exposition format. Includes:
    - facet_photos_total
    - facet_photos_with_embedding
    - facet_photos_with_topiq
    - facet_persons_total
    - facet_faces_total
    - facet_db_size_bytes
    - facet_process_memory_bytes (if psutil is installed)

    Intentionally lightweight (no histograms / counters that require state) —
    sufficient for monitoring scan progress and library size over time.
    """
    lines: list[str] = []

    def gauge(name: str, value: float | int, help_text: str) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value}")

    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT "
                "COUNT(*) AS photos, "
                "SUM(CASE WHEN clip_embedding IS NOT NULL THEN 1 ELSE 0 END) AS with_emb, "
                "SUM(CASE WHEN topiq_score IS NOT NULL THEN 1 ELSE 0 END) AS with_topiq "
                "FROM photos"
            ).fetchone()
            gauge("facet_photos_total", row["photos"] or 0, "Total photos in DB")
            gauge("facet_photos_with_embedding", row["with_emb"] or 0, "Photos with cached CLIP/SigLIP embedding")
            gauge("facet_photos_with_topiq", row["with_topiq"] or 0, "Photos with TOPIQ score populated")

            persons_row = conn.execute("SELECT COUNT(*) AS n FROM persons").fetchone()
            gauge("facet_persons_total", persons_row["n"] or 0, "Total person clusters")

            faces_row = conn.execute("SELECT COUNT(*) AS n FROM faces").fetchone()
            gauge("facet_faces_total", faces_row["n"] or 0, "Total faces")
    except Exception:
        # If the DB is unreachable, still serve metrics that don't depend on it.
        pass

    # DB file size on disk (sum of main file + WAL + SHM)
    try:
        from pathlib import Path
        from db import DEFAULT_DB_PATH
        db_path = Path(DEFAULT_DB_PATH)
        total_bytes = 0
        for suffix in ("", "-wal", "-shm"):
            p = db_path.with_name(db_path.name + suffix) if suffix else db_path
            if p.exists():
                total_bytes += p.stat().st_size
        gauge("facet_db_size_bytes", total_bytes, "DB file size on disk including WAL and SHM")
    except Exception:
        pass

    # Process memory (best-effort, requires psutil)
    try:
        import psutil
        import os as _os
        rss = psutil.Process(_os.getpid()).memory_info().rss
        gauge("facet_process_memory_bytes", rss, "Resident set size of the API process")
    except Exception:
        pass

    body = "\n".join(lines) + "\n"
    return PlainTextResponse(body, media_type="text/plain; version=0.0.4")
