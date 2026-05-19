"""Supplementary aesthetic score derived from cached CLIP/SigLIP embeddings.

The idea: pick a vector in CLIP/SigLIP space that points in the "aesthetic"
direction, then score each photo by the cosine of its already-computed image
embedding with that axis.

Why text-projection rather than a learned head:

- No new model weights to host or version.
- Reuses the existing embedding cached in ``photos.clip_embedding`` (zero extra
  image inference at scan time).
- Deterministic and trivially auditable: anyone can read the prompt list.

The catch: text-projection is a coarse signal. Expect ~0.35-0.55 SRCC against
AVA mean-opinion scores — usable as a supplementary input or a fast filter, not
a TOPIQ-IAA replacement (which sits around 0.94 SRCC). Benchmark with
``scripts/benchmark_aesthetic.py`` before promoting to a default.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


# Prompt sets crafted to anchor the axis without leaking subject-matter bias
# (no "portrait" vs "landscape" terms, no color words). Edit only if you can
# show a >=3% SRCC improvement on AVA — otherwise you're tuning to noise.
POSITIVE_PROMPTS: tuple[str, ...] = (
    "a professional, high-quality photograph",
    "an aesthetically beautiful image",
    "a masterful, award-winning photograph",
    "a sharp, well-composed photograph",
    "a stunning, visually striking image",
)
NEGATIVE_PROMPTS: tuple[str, ...] = (
    "a low-quality, amateur photograph",
    "a blurry, poorly composed photograph",
    "an unattractive, mundane snapshot",
    "a noisy, badly lit photograph",
    "a boring, forgettable image",
)


def build_aesthetic_axis(
    text_encode: callable,
    positive_prompts: Iterable[str] = POSITIVE_PROMPTS,
    negative_prompts: Iterable[str] = NEGATIVE_PROMPTS,
) -> np.ndarray:
    """Return a unit-norm float32 vector representing the aesthetic axis.

    ``text_encode`` is a caller-supplied function that maps an iterable of
    strings to a ``(N, D)`` float32 ndarray of L2-normalized text embeddings.
    The axis is ``mean(positive) - mean(negative)`` renormalized.
    """
    pos = text_encode(list(positive_prompts))
    neg = text_encode(list(negative_prompts))
    pos_mean = pos.mean(axis=0)
    neg_mean = neg.mean(axis=0)
    axis = pos_mean - neg_mean
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        raise ValueError("Aesthetic axis is degenerate (positive and negative means collapsed).")
    return (axis / norm).astype(np.float32)


def score_embedding(embedding: np.ndarray, axis: np.ndarray) -> float:
    """Score a single image embedding against the axis. Returns a value in [0, 10].

    The raw projection is ``<embedding, axis>`` in roughly [-1, 1]. We rescale
    so 0.0 maps to ~5.0 and ±1.0 maps to ~10 / ~0 — matching the rest of
    Facet's 0-10 score convention.
    """
    if embedding.shape != axis.shape:
        raise ValueError(
            f"Embedding shape {embedding.shape} != axis shape {axis.shape}"
        )
    cos = float(np.dot(embedding, axis))
    # Clip and rescale [-1, 1] -> [0, 10].
    cos = max(-1.0, min(1.0, cos))
    return (cos + 1.0) * 5.0


def score_embeddings(embeddings: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Score a batch of image embeddings. Shape ``(N, D)`` -> ``(N,)`` floats in [0, 10]."""
    if embeddings.ndim != 2 or embeddings.shape[1] != axis.shape[0]:
        raise ValueError(
            f"Embeddings shape {embeddings.shape} incompatible with axis {axis.shape}"
        )
    cos = embeddings @ axis
    cos = np.clip(cos, -1.0, 1.0)
    return ((cos + 1.0) * 5.0).astype(np.float32)
