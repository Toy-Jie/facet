"""Central torch device selection."""

from __future__ import annotations


def get_device() -> str:
    """Return the torch device string Facet should run on.

    Currently CUDA→CPU only. MPS (Apple Silicon) is detected separately via
    `mps_available()` for diagnostics, but Facet does not route torch models
    to MPS yet — see issue #7. Returning 'mps' here would silently break
    InsightFace (ONNX CUDAExecutionProvider) and several other torch paths
    that have not been validated on Metal.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def mps_available() -> bool:
    """True iff PyTorch reports Apple Silicon MPS is available.

    Used by `--doctor` to report MPS presence. Does NOT influence runtime
    device selection (see `get_device()`).
    """
    try:
        import torch
    except ImportError:
        return False
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    is_available = getattr(mps, "is_available", None) if mps is not None else None
    return bool(is_available()) if callable(is_available) else False
