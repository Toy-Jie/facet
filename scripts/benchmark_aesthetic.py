"""AVA aesthetic-score benchmark.

Compares the score columns stored in a Facet SQLite database against the AVA
mean-opinion-score (MOS) ground truth from ``AVA.txt``. Prints SRCC and PLCC for
every populated score column on the photos that overlap between the two sources.

Usage::

    python scripts/benchmark_aesthetic.py \\
        --db D:/photo-llm/ava_test.db \\
        --ava D:/photo-llm/AVA.txt \\
        --photo-dir D:/photo-llm/ava_test

    # Specific columns only:
    python scripts/benchmark_aesthetic.py --columns aesthetic aesthetic_iaa liqe_score

A column is included automatically if it exists in the photos table and has at
least one non-NULL value. Override with ``--columns`` to force a subset.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import pearsonr, spearmanr


DEFAULT_CANDIDATE_COLUMNS = [
    "aesthetic",
    "aesthetic_iaa",
    "topiq_score",
    "liqe_score",
    "quality_score",
    "aggregate",
    "aesthetic_clip",
]


def load_ava_ground_truth(ava_path: Path) -> dict[int, float]:
    """Return a dict mapping AVA image_id -> mean opinion score (1..10)."""
    mos: dict[int, float] = {}
    with ava_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 12:
                continue
            try:
                image_id = int(parts[1])
                votes = [int(x) for x in parts[2:12]]
            except ValueError:
                continue
            total = sum(votes)
            if total == 0:
                continue
            weighted = sum(v * (i + 1) for i, v in enumerate(votes))
            mos[image_id] = weighted / total
    return mos


def detect_columns(conn: sqlite3.Connection, requested: Iterable[str] | None) -> list[str]:
    """Return columns present in photos table; filter to ``requested`` if given."""
    rows = conn.execute("PRAGMA table_info(photos);").fetchall()
    available = {row[1] for row in rows}
    if requested:
        missing = [c for c in requested if c not in available]
        if missing:
            print(f"warning: columns not in DB: {missing}", file=sys.stderr)
        return [c for c in requested if c in available]
    return [c for c in DEFAULT_CANDIDATE_COLUMNS if c in available]


def load_scores(conn: sqlite3.Connection, photo_dir: Path, columns: list[str]) -> dict[int, dict[str, float]]:
    """Return ``{image_id: {column: value}}`` for photos whose path matches ``photo_dir``."""
    conn.row_factory = sqlite3.Row
    col_sql = ", ".join(columns)
    rows = conn.execute(
        f"SELECT path, {col_sql} FROM photos WHERE path LIKE ?",
        (f"%{photo_dir.name}%",),
    ).fetchall()

    scores: dict[int, dict[str, float]] = {}
    for row in rows:
        # Filename without extension = AVA image_id
        try:
            image_id = int(Path(row["path"]).stem)
        except ValueError:
            continue
        scores[image_id] = {c: row[c] for c in columns}
    return scores


def compute_metrics(
    scores: dict[int, dict[str, float]],
    mos: dict[int, float],
    columns: list[str],
) -> dict[str, dict[str, float | int]]:
    """Compute SRCC and PLCC per column on photos appearing in both sources."""
    results: dict[str, dict[str, float | int]] = {}
    for col in columns:
        x: list[float] = []
        y: list[float] = []
        for image_id, col_values in scores.items():
            val = col_values.get(col)
            if val is None:
                continue
            ground = mos.get(image_id)
            if ground is None:
                continue
            x.append(float(val))
            y.append(float(ground))
        if len(x) < 10:
            results[col] = {"n": len(x), "srcc": float("nan"), "plcc": float("nan")}
            continue
        srcc, _ = spearmanr(x, y)
        plcc, _ = pearsonr(x, y)
        results[col] = {"n": len(x), "srcc": float(srcc), "plcc": float(plcc)}
    return results


def print_report(results: dict[str, dict[str, float | int]], baseline: str) -> None:
    print()
    print("=" * 64)
    print(f"AVA benchmark - N photos / SRCC / PLCC (baseline: {baseline})")
    print("=" * 64)
    print(f"{'Column':<24}{'N':>8}{'SRCC':>12}{'PLCC':>12}{'d SRCC':>10}")
    print("-" * 64)
    base = results.get(baseline, {})
    base_srcc = base.get("srcc")
    base_srcc_float = float(base_srcc) if isinstance(base_srcc, (int, float)) and not np.isnan(base_srcc) else None
    for col, m in results.items():
        srcc = m["srcc"]
        plcc = m["plcc"]
        if isinstance(srcc, float) and np.isnan(srcc):
            srcc_str = "n/a"
        else:
            srcc_str = f"{srcc:+.4f}"
        if isinstance(plcc, float) and np.isnan(plcc):
            plcc_str = "n/a"
        else:
            plcc_str = f"{plcc:+.4f}"
        if base_srcc_float is not None and isinstance(srcc, (int, float)) and not np.isnan(srcc) and col != baseline:
            delta_str = f"{(srcc - base_srcc_float) * 100:+.2f}%"
        else:
            delta_str = "-"
        print(f"{col:<24}{m['n']:>8}{srcc_str:>12}{plcc_str:>12}{delta_str:>10}")
    print("=" * 64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=Path("ava_test.db"))
    p.add_argument("--ava", type=Path, default=Path("AVA.txt"))
    p.add_argument("--photo-dir", type=Path, default=Path("ava_test"))
    p.add_argument("--columns", nargs="*", help="Score columns to benchmark (default: auto-detect)")
    p.add_argument("--baseline", default="aesthetic", help="Column to compare deltas against")
    args = p.parse_args()

    if not args.db.exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 1
    if not args.ava.exists():
        print(f"AVA ground truth not found: {args.ava}", file=sys.stderr)
        return 1

    print(f"Loading AVA ground truth from {args.ava} ...")
    mos = load_ava_ground_truth(args.ava)
    print(f"  -> {len(mos):,} images with MOS")

    print(f"Loading scores from {args.db} ...")
    with sqlite3.connect(os.fspath(args.db)) as conn:
        columns = detect_columns(conn, args.columns)
        if not columns:
            print("No score columns found.", file=sys.stderr)
            return 1
        print(f"  -> benchmarking columns: {columns}")
        scores = load_scores(conn, args.photo_dir, columns)
    print(f"  -> {len(scores):,} photos in DB matching {args.photo_dir}")

    overlap = sum(1 for k in scores if k in mos)
    print(f"  -> {overlap:,} overlap with AVA MOS")

    results = compute_metrics(scores, mos, columns)
    print_report(results, baseline=args.baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
