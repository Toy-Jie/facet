"""Populate the ``aesthetic_clip`` column from cached embeddings + text projection.

Loads the same CLIP/SigLIP model used at scan time, builds the aesthetic axis
from prompts in ``analyzers/aesthetic_clip.py``, then scores every photo with a
cached ``clip_embedding`` BLOB. The column is created if missing.

Usage::

    python scripts/compute_aesthetic_clip.py --db D:/photo-llm/ava_test.db

By default reads the active CLIP/SigLIP config from ``scoring_config.json``;
override with ``--model`` and ``--backend`` for ad-hoc benchmarks.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

# Make project root importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzers.aesthetic_clip import (
    NEGATIVE_PROMPTS,
    POSITIVE_PROMPTS,
    build_aesthetic_axis,
    score_embeddings,
)


def load_clip_config() -> dict:
    from config import ScoringConfig
    return ScoringConfig(validate=False).get_clip_config()


def make_text_encoder(model_name: str, backend: str):
    """Return ``(encode_fn, embed_dim)`` where ``encode_fn`` maps list[str] -> (N, D) float32."""
    import torch

    if backend == "transformers":
        from transformers import AutoModel, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model = model.to(device).eval()
        if device == "cuda":
            model = model.half()
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        def encode(texts):
            inputs = processor(text=list(texts), padding="max_length", return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.get_text_features(**inputs)
            # NaFlex variant may return a BaseModelOutputWithPooling wrapper
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                feats = out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                feats = out.last_hidden_state[:, 0]  # CLS-equivalent
            else:
                feats = out  # already a tensor
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.float().cpu().numpy().astype(np.float32)

        # Peek at one prompt to learn the embedding dim
        dim = encode(["probe"]).shape[1]
        return encode, dim

    # open_clip path
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained = "laion2b_s32b_b82k"  # 8gb-profile default; override if needed
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()
    tokenizer = open_clip.get_tokenizer(model_name)

    def encode(texts):
        tokens = tokenizer(list(texts)).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy().astype(np.float32)

    dim = encode(["probe"]).shape[1]
    return encode, dim


def ensure_column(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(photos);").fetchall()}
    if "aesthetic_clip" not in cols:
        conn.execute("ALTER TABLE photos ADD COLUMN aesthetic_clip REAL;")
        print("Added column photos.aesthetic_clip")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--model", default=None, help="Override CLIP model name (default: from config)")
    p.add_argument("--backend", default=None, choices=("open_clip", "transformers"))
    p.add_argument("--photo-dir", default=None, help="Only score photos whose path contains this substring")
    p.add_argument("--dry-run", action="store_true", help="Compute scores but don't write to DB")
    args = p.parse_args()

    if not args.db.exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 1

    cfg = load_clip_config()
    model_name = args.model or cfg.get("model_name", "ViT-L-14")
    backend = args.backend or cfg.get("backend", "open_clip")
    print(f"Loading {backend} model: {model_name}")
    encode, dim = make_text_encoder(model_name, backend)
    print(f"  text embedding dim: {dim}")

    print(f"Building aesthetic axis from {len(POSITIVE_PROMPTS)}+{len(NEGATIVE_PROMPTS)} prompts ...")
    axis = build_aesthetic_axis(encode)
    print(f"  axis shape: {axis.shape}")

    with sqlite3.connect(os.fspath(args.db)) as conn:
        conn.row_factory = sqlite3.Row
        ensure_column(conn)

        where_clauses = ["clip_embedding IS NOT NULL"]
        params: list = []
        if args.photo_dir:
            where_clauses.append("path LIKE ?")
            params.append(f"%{args.photo_dir}%")
        where_sql = " AND ".join(where_clauses)
        rows = conn.execute(
            f"SELECT path, clip_embedding FROM photos WHERE {where_sql}", params,
        ).fetchall()
        print(f"Found {len(rows):,} photos with cached embeddings")

        t0 = time.time()
        BATCH = 1024
        n_written = 0
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            embs = np.stack([
                np.frombuffer(r["clip_embedding"], dtype=np.float32) for r in batch
            ])
            if embs.shape[1] != dim:
                print(
                    f"Embedding/axis dim mismatch: cached emb is {embs.shape[1]}-dim, "
                    f"text axis is {dim}-dim. The model you loaded ({model_name}) "
                    "does not match the model that produced these embeddings.",
                    file=sys.stderr,
                )
                return 1
            scores = score_embeddings(embs, axis)
            if not args.dry_run:
                conn.executemany(
                    "UPDATE photos SET aesthetic_clip = ? WHERE path = ?",
                    [(float(s), r["path"]) for s, r in zip(scores, batch)],
                )
            n_written += len(batch)
        if not args.dry_run:
            conn.commit()
        elapsed = time.time() - t0
        suffix = " (dry-run)" if args.dry_run else ""
        print(f"Scored {n_written:,} photos in {elapsed:.1f}s{suffix}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
