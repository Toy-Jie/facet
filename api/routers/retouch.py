"""Local, non-destructive portrait retouch endpoints."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image, ImageEnhance, ImageOps
from pydantic import BaseModel, Field

from api.auth import CurrentUser, require_edition
from api.config import is_multi_user_enabled
from api.database import get_db
from api.db_helpers import get_visibility_clause
from api.path_validation import resolve_photo_disk_path

logger = logging.getLogger(__name__)
router = APIRouter(tags=["retouch"])


class CropRect(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    unit: str = Field(default="normalized", pattern="^(normalized|pixel)$")


class RetouchAdjustments(BaseModel):
    crop: Optional[CropRect] = None
    rotate: int = Field(default=0, ge=-360, le=360)
    flip_horizontal: bool = False
    flip_vertical: bool = False
    brightness: float = Field(default=0, ge=-100, le=100)
    contrast: float = Field(default=0, ge=-100, le=100)
    saturation: float = Field(default=0, ge=-100, le=100)
    temperature: float = Field(default=0, ge=-100, le=100)
    smooth_skin: float = Field(default=0, ge=0, le=100)
    whiten_skin: float = Field(default=0, ge=0, le=100)
    face_blemish: float = Field(default=0, ge=0, le=100)
    face_wrinkle: float = Field(default=0, ge=0, le=100)
    wrinkle_nasolabial_fold: float = Field(default=0, ge=0, le=100)
    wrinkle_under_eye: float = Field(default=0, ge=0, le=100)
    wrinkle_forehead: float = Field(default=0, ge=0, le=100)
    wrinkle_glabella: float = Field(default=0, ge=0, le=100)
    wrinkle_mouth_corner: float = Field(default=0, ge=0, le=100)
    wrinkle_smooth_radius: float = Field(default=100, ge=25, le=300)
    wrinkle_blend: float = Field(default=100, ge=0, le=200)
    wrinkle_detail_protection: float = Field(default=100, ge=0, le=200)
    body_blemish: float = Field(default=0, ge=0, le=100)
    skin_texture: float = Field(default=0, ge=0, le=100)
    skin_tone: float = Field(default=0, ge=-100, le=100)
    face_fullness: float = Field(default=0, ge=0, le=100)
    face_shape: float = Field(default=0, ge=-100, le=100)
    eyebrow: float = Field(default=0, ge=-100, le=100)
    nose: float = Field(default=0, ge=-100, le=100)
    eyes: float = Field(default=0, ge=-100, le=100)
    mouth: float = Field(default=0, ge=-100, le=100)
    close_mouth: float = Field(default=0, ge=0, le=100)
    teeth: float = Field(default=0, ge=0, le=100)
    eye_enhance: float = Field(default=0, ge=0, le=100)
    beauty_skin_mask_strength: float = Field(default=100, ge=0, le=200)
    beauty_skin_mask_feather: float = Field(default=100, ge=0, le=200)
    beauty_detail_protection: float = Field(default=100, ge=0, le=200)
    beauty_smooth_color: float = Field(default=100, ge=0, le=200)
    beauty_smooth_radius: float = Field(default=100, ge=0, le=200)
    beauty_smooth_blend: float = Field(default=100, ge=0, le=200)
    beauty_whiten_brightness: float = Field(default=100, ge=0, le=200)
    beauty_whiten_saturation: float = Field(default=100, ge=0, le=200)
    beauty_whiten_blend: float = Field(default=100, ge=0, le=200)
    beauty_skin_tone_temperature: float = Field(default=100, ge=0, le=200)
    beauty_texture_amount: float = Field(default=100, ge=0, le=200)
    beauty_texture_radius: float = Field(default=100, ge=25, le=300)
    beauty_feature_detail: float = Field(default=100, ge=0, le=200)
    beauty_feature_radius: float = Field(default=100, ge=25, le=300)
    beauty_teeth_brightness: float = Field(default=100, ge=0, le=200)
    beauty_teeth_saturation: float = Field(default=100, ge=0, le=200)
    beauty_teeth_threshold: float = Field(default=100, ge=0, le=200)
    beauty_inpaint_radius: float = Field(default=100, ge=25, le=300)
    hair_recolor: float = Field(default=0, ge=0, le=100)
    hair_color: str = Field(default="#3b2418", pattern=r"^#[0-9a-fA-F]{6}$")
    hair_part_fill: float = Field(default=0, ge=0, le=100)
    hair_smooth: float = Field(default=0, ge=0, le=100)
    hair_mask_feather: float = Field(default=100, ge=0, le=200)
    hair_texture_preserve: float = Field(default=100, ge=0, le=200)
    background_blur: float = Field(default=0, ge=0, le=100)
    background_subject_protection: float = Field(default=100, ge=0, le=150)
    background_subject_expand: float = Field(default=100, ge=0, le=200)
    background_edge_feather: float = Field(default=100, ge=0, le=200)
    background_depth_strength: float = Field(default=100, ge=0, le=200)
    background_model_depth_weight: float = Field(default=72, ge=0, le=100)
    background_foreground_protection: float = Field(default=100, ge=0, le=200)
    background_near_blur: float = Field(default=100, ge=0, le=200)
    background_mid_blur: float = Field(default=100, ge=0, le=200)
    background_far_blur: float = Field(default=100, ge=0, le=200)
    selected_face_ids: Optional[list[int]] = Field(default=None, max_length=200)
    inpaint_mask_base64: Optional[str] = None


class RetouchPreviewBody(BaseModel):
    photo_id: Optional[str] = None
    image_path: Optional[str] = None
    params: RetouchAdjustments = Field(default_factory=RetouchAdjustments)
    max_size: int = Field(default=0, ge=0, le=12000)
    compare: bool = False


class RetouchApplyBody(BaseModel):
    photo_id: Optional[str] = None
    image_path: Optional[str] = None
    params: RetouchAdjustments = Field(default_factory=RetouchAdjustments)
    output_suffix: str = Field(default=".retouch")


class RetouchDownloadBody(BaseModel):
    photo_id: Optional[str] = None
    image_path: Optional[str] = None
    params: RetouchAdjustments = Field(default_factory=RetouchAdjustments)


class RetouchMaskBody(BaseModel):
    photo_id: Optional[str] = None
    image_path: Optional[str] = None
    max_size: int = Field(default=768, ge=128, le=1536)


class RetouchInpaintBody(BaseModel):
    photo_id: Optional[str] = None
    image_path: Optional[str] = None
    mask_base64: str
    params: RetouchAdjustments = Field(default_factory=RetouchAdjustments)
    save_copy: bool = False


def _photo_path(body: Any) -> str:
    path = body.photo_id or body.image_path
    if not path:
        raise HTTPException(status_code=422, detail="photo_id or image_path is required")
    return path


def _visible_photo_row(conn: sqlite3.Connection, db_path: str, user: CurrentUser):
    user_id = user.user_id if user else None
    vis_sql, vis_params = get_visibility_clause(user_id)
    row = conn.execute(
        f"SELECT * FROM photos WHERE path = ? AND {vis_sql}",
        [db_path] + vis_params,
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Photo not found")
    return row


def _resolve_visible_path(conn: sqlite3.Connection, db_path: str, user: CurrentUser) -> tuple[sqlite3.Row, str]:
    row = _visible_photo_row(conn, db_path, user)
    return row, resolve_photo_disk_path(db_path)


def _selected_face_id_set(params: RetouchAdjustments) -> Optional[set[int]]:
    if params.selected_face_ids is None:
        return None
    return {int(face_id) for face_id in params.selected_face_ids if int(face_id) > 0}


def _face_boxes_for_photo(
    conn: sqlite3.Connection,
    db_path: str,
    selected_face_ids: Optional[set[int]] = None,
) -> list[tuple[int, int, int, int]]:
    if selected_face_ids is not None and not selected_face_ids:
        return []
    try:
        rows = conn.execute(
            """SELECT id, bbox_x1, bbox_y1, bbox_x2, bbox_y2
               FROM faces
               WHERE photo_path = ?
                 AND bbox_x1 IS NOT NULL AND bbox_y1 IS NOT NULL
                 AND bbox_x2 IS NOT NULL AND bbox_y2 IS NOT NULL
               ORDER BY face_index, id""",
            [db_path],
        ).fetchall()
    except sqlite3.Error:
        return []
    boxes: list[tuple[int, int, int, int]] = []
    for row in rows:
        if selected_face_ids is not None and int(row[0]) not in selected_face_ids:
            continue
        x1, y1, x2, y2 = int(row[1]), int(row[2]), int(row[3]), int(row[4])
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def _face_landmarks_for_photo(
    conn: sqlite3.Connection,
    db_path: str,
    selected_face_ids: Optional[set[int]] = None,
) -> list[Optional[np.ndarray]]:
    if selected_face_ids is not None and not selected_face_ids:
        return []
    try:
        rows = conn.execute(
            """SELECT id, landmark_2d_106
               FROM faces
               WHERE photo_path = ?
               ORDER BY face_index, id""",
            [db_path],
        ).fetchall()
    except sqlite3.Error:
        return []
    landmarks: list[Optional[np.ndarray]] = []
    for row in rows:
        if selected_face_ids is not None and int(row[0]) not in selected_face_ids:
            continue
        if row[1] is None:
            landmarks.append(None)
            continue
        try:
            lm = np.frombuffer(row[1], dtype=np.float32).reshape(106, 2).copy()
        except Exception:
            landmarks.append(None)
            continue
        if np.isfinite(lm).all() and lm.shape == (106, 2):
            landmarks.append(lm)
        else:
            landmarks.append(None)
    return landmarks


def _load_image(path: str) -> Image.Image:
    try:
        with Image.open(path) as img:
            return ImageOps.exif_transpose(img).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=415, detail="Unsupported image file") from exc


def _downsample(img: Image.Image, max_size: int) -> Image.Image:
    if max_size <= 0:
        return img.copy()
    if max(img.size) <= max_size:
        return img.copy()
    preview = img.copy()
    preview.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return preview


def _factor(percent: float, scale: float = 1.0) -> float:
    return max(0.05, 1.0 + (percent / 100.0) * scale)


def _apply_crop(img: Image.Image, crop: CropRect) -> Image.Image:
    w, h = img.size
    if crop.unit == "normalized":
        left = int(round(crop.x * w))
        top = int(round(crop.y * h))
        right = int(round((crop.x + crop.width) * w))
        bottom = int(round((crop.y + crop.height) * h))
    else:
        left = int(round(crop.x))
        top = int(round(crop.y))
        right = int(round(crop.x + crop.width))
        bottom = int(round(crop.y + crop.height))
    left = max(0, min(w - 1, left))
    top = max(0, min(h - 1, top))
    right = max(left + 1, min(w, right))
    bottom = max(top + 1, min(h, bottom))
    return img.crop((left, top, right, bottom))


def _apply_temperature(arr: np.ndarray, value: float) -> np.ndarray:
    if abs(value) < 0.01:
        return arr
    out = arr.astype(np.float32)
    amount = value / 100.0
    out[:, :, 0] *= 1.0 + 0.10 * amount
    out[:, :, 1] *= 1.0 + 0.025 * abs(amount)
    out[:, :, 2] *= 1.0 - 0.10 * amount
    return np.clip(out, 0, 255).astype(np.uint8)


def _skin_mask(rgb: np.ndarray, strength: float = 100, feather: float = 100) -> np.ndarray:
    """Conservative skin mask using YCrCb + HSV, feathered for blending."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    y, cr, cb = cv2.split(ycrcb)
    h, s, v = cv2.split(hsv)

    skin = (
        (cr > 132) & (cr < 178) &
        (cb > 76) & (cb < 136) &
        (s > 18) & (s < 190) &
        (v > 45) &
        (y > 35)
    )
    mask = (skin.astype(np.uint8) * 255)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    sigma = max(0.1, 5 * (feather / 100.0))
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip((mask.astype(np.float32) / 255.0) * (strength / 100.0), 0, 1)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _fallback_hair_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, w = rgb.shape[:2]
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    upper_prior = np.repeat(np.clip(1.1 - y * 1.45, 0, 1), w, axis=1)
    dark_or_saturated = ((hsv[:, :, 2] < 120) | ((hsv[:, :, 1] > 55) & (hsv[:, :, 2] < 185))).astype(np.float32)
    skin = _skin_mask(rgb)
    mask = dark_or_saturated * upper_prior * (1.0 - np.clip(skin * 1.4, 0, 1))
    mask = cv2.morphologyEx((mask > 0.20).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 4)
    return np.clip(mask, 0, 1)


def _hair_mask(rgb: np.ndarray, feather: float = 100) -> np.ndarray:
    global _OPTIONAL_HAIR_PARSER, _OPTIONAL_HAIR_FAILED
    cache_key = _image_cache_key(f"hair:{int(feather)}", rgb)
    if cache_key in _HAIR_MASK_CACHE:
        return _HAIR_MASK_CACHE[cache_key].copy()
    mask: Optional[np.ndarray] = None
    if not _OPTIONAL_HAIR_FAILED:
        try:
            if _OPTIONAL_HAIR_PARSER is None:
                from uniface import BiSeNet
                from uniface.constants import ParsingWeights

                _OPTIONAL_HAIR_PARSER = BiSeNet(model_name=ParsingWeights.RESNET18)
            parsed = _OPTIONAL_HAIR_PARSER.parse(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            # CelebAMask-HQ face parsing label 17 is hair.
            mask = (parsed == 17).astype(np.float32)
        except Exception as exc:
            _OPTIONAL_HAIR_FAILED = True
            logger.warning("UniFace hair parsing unavailable; falling back to OpenCV hair estimate: %s", exc)
    if mask is None or mask.max() <= 0.01:
        mask = _fallback_hair_mask(rgb)
    sigma = max(0.1, 4.0 * (feather / 100.0))
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    mask = np.clip(mask, 0, 1)
    _cache_store(_HAIR_MASK_CACHE, cache_key, mask)
    return mask


def _apply_hair_effects(
    rgb: np.ndarray,
    params: RetouchAdjustments,
    skin: np.ndarray,
    effect_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if max(params.hair_recolor, params.hair_part_fill, params.hair_smooth) <= 0:
        return rgb
    hair = _hair_mask(rgb, params.hair_mask_feather)
    if effect_mask is not None:
        hair = hair * np.clip(effect_mask, 0, 1)
    if hair.max() <= 0.01:
        return rgb
    result = rgb

    if params.hair_part_fill > 0:
        hair_dilated = cv2.dilate((hair > 0.18).astype(np.uint8), np.ones((13, 13), np.uint8), iterations=1)
        part = ((skin > 0.18) & (hair_dilated > 0)).astype(np.uint8) * 255
        if part.max() > 0:
            bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            repaired = cv2.inpaint(bgr, part, 4, cv2.INPAINT_TELEA)
            repaired = cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)
            alpha = cv2.GaussianBlur(part.astype(np.float32) / 255.0, (0, 0), 2)[:, :, None] * (params.hair_part_fill / 100.0)
            result = np.clip(result.astype(np.float32) * (1 - alpha) + repaired.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    if params.hair_smooth > 0:
        try:
            smooth = cv2.edgePreservingFilter(result, flags=1, sigma_s=38, sigma_r=0.18)
        except Exception:
            smooth = cv2.bilateralFilter(result, d=0, sigmaColor=45, sigmaSpace=22)
        detail = _detail_protection_mask(result)
        preserve = 1.0 - (1.0 - detail) * (params.hair_texture_preserve / 100.0)
        alpha = (hair * preserve * 0.45 * (params.hair_smooth / 100.0))[:, :, None]
        result = np.clip(result.astype(np.float32) * (1 - alpha) + smooth.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    if params.hair_recolor > 0:
        target_rgb = np.uint8([[_hex_to_rgb(params.hair_color)]])
        target_hsv = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)[0, 0]
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
        recolored = hsv.copy()
        recolored[:, :, 0] = target_hsv[0]
        recolored[:, :, 1] = np.clip(0.55 * hsv[:, :, 1] + 0.45 * target_hsv[1], 0, 255)
        recolored[:, :, 2] = np.clip(hsv[:, :, 2] * 0.92 + target_hsv[2] * 0.08, 0, 255)
        recolored_rgb = cv2.cvtColor(np.clip(recolored, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        alpha = (hair * (params.hair_recolor / 100.0))[:, :, None]
        result = np.clip(result.astype(np.float32) * (1 - alpha) + recolored_rgb.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    return result


def _detail_protection_mask(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_32F)
    edges = np.abs(edges)
    edges = cv2.GaussianBlur(edges, (0, 0), 1.5)
    protected = np.clip(edges / 26.0, 0, 1)
    return 1.0 - protected


def _portrait_like_mask(rgb: np.ndarray, skin: np.ndarray) -> np.ndarray:
    h, w = skin.shape
    if skin.max() <= 0.05:
        return np.zeros_like(skin)
    binary = (skin > 0.15).astype(np.uint8)
    kernel = np.ones((35, 35), np.uint8)
    expanded = cv2.dilate(binary, kernel, iterations=2)
    expanded = cv2.GaussianBlur(expanded.astype(np.float32), (0, 0), max(9, min(w, h) / 45))
    return np.clip(expanded, 0, 1)


_OPTIONAL_SALIENCY_SCORER = None
_OPTIONAL_SALIENCY_FAILED = False
_OPTIONAL_DEPTH_ESTIMATOR = None
_OPTIONAL_DEPTH_FAILED = False
_OPTIONAL_HAIR_PARSER = None
_OPTIONAL_HAIR_FAILED = False
_SALIENCY_MASK_CACHE: dict[str, np.ndarray] = {}
_DEPTH_MAP_CACHE: dict[str, np.ndarray] = {}
_HAIR_MASK_CACHE: dict[str, np.ndarray] = {}


def _retouch_model_mode() -> str:
    return os.getenv("FACET_RETOUCH_MODEL_MODE", "auto").strip().lower()


def _retouch_models_enabled() -> bool:
    return _retouch_model_mode() not in {"0", "false", "off", "opencv"}


def _image_cache_key(prefix: str, rgb: np.ndarray, max_side: int = 96) -> str:
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / float(max(1, max(h, w))))
    if scale < 1.0:
        sample = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    else:
        sample = rgb
    digest = hashlib.sha1(np.ascontiguousarray(sample).tobytes()).hexdigest()
    return f"{prefix}:{w}x{h}:{digest}"


def _cache_store(cache: dict[str, np.ndarray], key: str, value: np.ndarray, max_items: int = 12):
    if len(cache) >= max_items:
        cache.pop(next(iter(cache)))
    cache[key] = value.copy()


def _small_mask(mask: np.ndarray, max_side: int = 900) -> tuple[np.ndarray, float]:
    h, w = mask.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return mask, 1.0
    scale = max_side / float(longest)
    resized = cv2.resize(mask, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def _remove_tiny_components(mask: np.ndarray, min_area_ratio: float = 0.003) -> np.ndarray:
    binary = (mask > 0.2).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return mask
    h, w = mask.shape
    min_area = max(24, int(h * w * min_area_ratio))
    keep = np.zeros_like(binary)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == label] = 1
    if keep.max() == 0:
        return mask
    return mask * keep.astype(np.float32)


def _optional_birefnet_subject_mask(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Use BiRefNet in auto/high-quality mode, with an in-process preview cache."""
    global _OPTIONAL_SALIENCY_SCORER, _OPTIONAL_SALIENCY_FAILED
    if not _retouch_models_enabled() or _OPTIONAL_SALIENCY_FAILED:
        return None
    cache_key = _image_cache_key("birefnet", rgb)
    if cache_key in _SALIENCY_MASK_CACHE:
        return _SALIENCY_MASK_CACHE[cache_key].copy()
    try:
        if _OPTIONAL_SALIENCY_SCORER is None:
            from models.saliency_scorer import SaliencyScorer

            _OPTIONAL_SALIENCY_SCORER = SaliencyScorer(resolution=768, mask_threshold=0.35)
        pil = Image.fromarray(rgb, mode="RGB")
        mask = _OPTIONAL_SALIENCY_SCORER.get_saliency_mask(pil).astype(np.float32) / 255.0
        if mask.shape != rgb.shape[:2]:
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        if 0.01 < float(mask.mean()) < 0.85:
            mask = cv2.GaussianBlur(mask, (0, 0), max(2, min(rgb.shape[:2]) / 180))
            mask = np.clip(mask, 0, 1)
            _cache_store(_SALIENCY_MASK_CACHE, cache_key, mask)
            return mask
    except Exception as exc:
        _OPTIONAL_SALIENCY_FAILED = True
        logger.warning("Optional BiRefNet subject mask unavailable; falling back to OpenCV depth blur: %s", exc)
    return None


def _optional_depth_map(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Estimate relative scene depth when transformers depth models are available."""
    global _OPTIONAL_DEPTH_ESTIMATOR, _OPTIONAL_DEPTH_FAILED
    if not _retouch_models_enabled() or _OPTIONAL_DEPTH_FAILED:
        return None
    cache_key = _image_cache_key("depth", rgb)
    if cache_key in _DEPTH_MAP_CACHE:
        return _DEPTH_MAP_CACHE[cache_key].copy()
    try:
        if _OPTIONAL_DEPTH_ESTIMATOR is None:
            from transformers import pipeline

            model_name = os.getenv("FACET_RETOUCH_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
            device = -1
            try:
                import torch

                if torch.cuda.is_available():
                    device = 0
            except Exception:
                device = -1
            _OPTIONAL_DEPTH_ESTIMATOR = pipeline("depth-estimation", model=model_name, device=device)
        pil = Image.fromarray(rgb, mode="RGB")
        output = _OPTIONAL_DEPTH_ESTIMATOR(pil)
        depth_img = output.get("depth")
        if depth_img is None:
            return None
        depth = np.array(depth_img.convert("L").resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32)
        p2, p98 = np.percentile(depth, [2, 98])
        if p98 - p2 < 1e-3:
            return None
        depth = np.clip((depth - p2) / (p98 - p2), 0, 1)
        # HF depth pipelines differ in polarity. Invert when the lower foreground is brighter.
        top_mean = float(depth[: max(1, rgb.shape[0] // 4), :].mean())
        bottom_mean = float(depth[-max(1, rgb.shape[0] // 4):, :].mean())
        if bottom_mean > top_mean:
            depth = 1.0 - depth
        depth = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), max(2, min(rgb.shape[:2]) / 220))
        depth = np.clip(depth, 0, 1)
        _cache_store(_DEPTH_MAP_CACHE, cache_key, depth)
        return depth
    except Exception as exc:
        _OPTIONAL_DEPTH_FAILED = True
        logger.warning("Optional depth-estimation model unavailable; falling back to perspective heuristic: %s", exc)
    return None


def _central_subject_prior(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (x / max(1, w - 1) - 0.5) / 0.34
    ny = (y / max(1, h - 1) - 0.52) / 0.48
    ellipse = np.exp(-(nx * nx + ny * ny) * 1.8)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    edges = cv2.GaussianBlur(edges, (0, 0), 2.0)
    edge_norm = edges / (np.percentile(edges, 96) + 1e-6)
    texture = np.clip(edge_norm, 0, 1)
    return np.clip(ellipse * (0.35 + 0.65 * texture), 0, 1).astype(np.float32)


def _face_subject_prior(shape: tuple[int, int], face_boxes: Optional[list[tuple[int, int, int, int]]]) -> np.ndarray:
    h, w = shape
    prior = np.zeros((h, w), dtype=np.float32)
    if not face_boxes:
        return prior
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    for x1, y1, x2, y2 in face_boxes:
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(x1 + 1, min(w, int(x2)))
        y2 = max(y1 + 1, min(h, int(y2)))
        face_w = max(1, x2 - x1)
        face_h = max(1, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        head = np.exp(-(((x - cx) / (face_w * 0.92)) ** 2 + ((y - cy) / (face_h * 1.05)) ** 2) * 1.6)
        torso_cy = min(h * 0.98, cy + face_h * 2.15)
        torso = np.exp(-(((x - cx) / (face_w * 2.4)) ** 2 + ((y - torso_cy) / (face_h * 3.2)) ** 2) * 1.35)
        shoulders_cy = min(h * 0.95, cy + face_h * 1.35)
        shoulders = np.exp(-(((x - cx) / (face_w * 3.0)) ** 2 + ((y - shoulders_cy) / (face_h * 1.5)) ** 2) * 1.5)
        prior = np.maximum(prior, np.maximum(head, np.maximum(torso * 0.72, shoulders * 0.52)))

    return np.clip(prior, 0, 1)


def _selected_face_effect_mask(
    shape: tuple[int, int],
    face_boxes: Optional[list[tuple[int, int, int, int]]],
) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)
    if not face_boxes:
        return mask
    for x1, y1, x2, y2 in face_boxes:
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(x1 + 1, min(w, int(x2)))
        y2 = max(y1 + 1, min(h, int(y2)))
        face_w = max(1, x2 - x1)
        face_h = max(1, y2 - y1)
        center = (int(round((x1 + x2) / 2)), int(round(y1 + face_h * 0.44)))
        axes = (max(3, int(round(face_w * 0.90))), max(3, int(round(face_h * 1.28))))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1, lineType=cv2.LINE_AA)
        hair_center = (center[0], int(round(y1 + face_h * 0.12)))
        hair_axes = (max(3, int(round(face_w * 1.25))), max(3, int(round(face_h * 0.78))))
        cv2.ellipse(mask, hair_center, hair_axes, 0, 0, 360, 1.0, -1, lineType=cv2.LINE_AA)
    sigma = max(1.2, min(h, w) / 180)
    return np.clip(cv2.GaussianBlur(mask, (0, 0), sigma), 0, 1)


def _add_soft_ellipse(mask: np.ndarray, center: tuple[float, float], axes: tuple[float, float], value: float, angle: float = 0):
    if value <= 0:
        return
    h, w = mask.shape
    cx = int(round(center[0]))
    cy = int(round(center[1]))
    ax = max(1, int(round(axes[0])))
    ay = max(1, int(round(axes[1])))
    if cx < -ax or cx > w + ax or cy < -ay or cy > h + ay:
        return
    layer = np.zeros_like(mask, dtype=np.float32)
    cv2.ellipse(layer, (cx, cy), (ax, ay), angle, 0, 360, float(value), -1, lineType=cv2.LINE_AA)
    np.maximum(mask, layer, out=mask)


def _add_soft_line(mask: np.ndarray, start: tuple[float, float], end: tuple[float, float], width: float, value: float):
    if value <= 0:
        return
    layer = np.zeros_like(mask, dtype=np.float32)
    cv2.line(
        layer,
        (int(round(start[0])), int(round(start[1]))),
        (int(round(end[0])), int(round(end[1]))),
        float(value),
        max(1, int(round(width))),
        lineType=cv2.LINE_AA,
    )
    np.maximum(mask, layer, out=mask)


def _wrinkle_zone_mask(
    shape: tuple[int, int],
    params: RetouchAdjustments,
    face_boxes: Optional[list[tuple[int, int, int, int]]] = None,
    face_landmarks: Optional[list[Optional[np.ndarray]]] = None,
) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)
    strengths = {
        "nasolabial": params.wrinkle_nasolabial_fold / 100.0,
        "under_eye": params.wrinkle_under_eye / 100.0,
        "forehead": params.wrinkle_forehead / 100.0,
        "glabella": params.wrinkle_glabella / 100.0,
        "mouth_corner": params.wrinkle_mouth_corner / 100.0,
    }
    if max(strengths.values()) <= 0:
        return mask

    boxes = face_boxes or []
    landmarks = face_landmarks or []
    count = max(len(boxes), len(landmarks))
    for idx in range(count):
        lm = landmarks[idx] if idx < len(landmarks) else None
        if idx < len(boxes):
            x1, y1, x2, y2 = boxes[idx]
        elif lm is not None:
            valid = lm[np.any(lm > 0, axis=1)]
            if valid.size == 0:
                continue
            x1, y1 = valid.min(axis=0)
            x2, y2 = valid.max(axis=0)
        else:
            continue
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        fw = max(1.0, x2 - x1)
        fh = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0

        if lm is not None and lm.shape[0] >= 96:
            left_eye = lm[[35, 39, 37, 38, 41, 40]]
            right_eye = lm[[89, 93, 91, 92, 95, 94]]
            mouth = lm[52:72] if lm.shape[0] >= 72 else np.empty((0, 2), dtype=np.float32)
            left_eye_c = left_eye.mean(axis=0)
            right_eye_c = right_eye.mean(axis=0)
            eye_c = (left_eye_c + right_eye_c) / 2.0
            mouth_c = mouth.mean(axis=0) if mouth.size else np.array([cx, y1 + fh * 0.72], dtype=np.float32)
            mouth_left = mouth[np.argmin(mouth[:, 0])] if mouth.size else np.array([cx - fw * 0.16, y1 + fh * 0.72], dtype=np.float32)
            mouth_right = mouth[np.argmax(mouth[:, 0])] if mouth.size else np.array([cx + fw * 0.16, y1 + fh * 0.72], dtype=np.float32)
            eye_span = max(1.0, float(np.linalg.norm(right_eye_c - left_eye_c)))
        else:
            left_eye_c = np.array([x1 + fw * 0.34, y1 + fh * 0.38], dtype=np.float32)
            right_eye_c = np.array([x1 + fw * 0.66, y1 + fh * 0.38], dtype=np.float32)
            eye_c = (left_eye_c + right_eye_c) / 2.0
            mouth_c = np.array([cx, y1 + fh * 0.74], dtype=np.float32)
            mouth_left = np.array([x1 + fw * 0.36, y1 + fh * 0.74], dtype=np.float32)
            mouth_right = np.array([x1 + fw * 0.64, y1 + fh * 0.74], dtype=np.float32)
            eye_span = fw * 0.32

        nose_left = np.array([cx - fw * 0.10, eye_c[1] + (mouth_c[1] - eye_c[1]) * 0.45], dtype=np.float32)
        nose_right = np.array([cx + fw * 0.10, eye_c[1] + (mouth_c[1] - eye_c[1]) * 0.45], dtype=np.float32)
        _add_soft_line(mask, nose_left, mouth_left, fw * 0.045, strengths["nasolabial"])
        _add_soft_line(mask, nose_right, mouth_right, fw * 0.045, strengths["nasolabial"])

        _add_soft_ellipse(mask, (left_eye_c[0], left_eye_c[1] + fh * 0.10), (eye_span * 0.30, fh * 0.055), strengths["under_eye"], -8)
        _add_soft_ellipse(mask, (right_eye_c[0], right_eye_c[1] + fh * 0.10), (eye_span * 0.30, fh * 0.055), strengths["under_eye"], 8)
        _add_soft_ellipse(mask, (cx, y1 + fh * 0.18), (fw * 0.34, fh * 0.085), strengths["forehead"])
        _add_soft_ellipse(mask, (cx, eye_c[1] - fh * 0.055), (fw * 0.075, fh * 0.085), strengths["glabella"])
        _add_soft_ellipse(mask, (mouth_left[0], mouth_left[1]), (fw * 0.10, fh * 0.075), strengths["mouth_corner"], -18)
        _add_soft_ellipse(mask, (mouth_right[0], mouth_right[1]), (fw * 0.10, fh * 0.075), strengths["mouth_corner"], 18)

    feather = max(0.1, 3.0 * (params.beauty_skin_mask_feather / 100.0))
    mask = cv2.GaussianBlur(mask, (0, 0), feather)
    return np.clip(mask, 0, 1)


def _apply_targeted_wrinkle_smoothing(
    rgb: np.ndarray,
    wrinkle_mask: np.ndarray,
    params: RetouchAdjustments,
) -> np.ndarray:
    if wrinkle_mask.max() <= 0.01:
        return rgb
    radius = max(1.0, 32.0 * (params.wrinkle_smooth_radius / 100.0))
    try:
        smoothed = cv2.edgePreservingFilter(rgb, flags=1, sigma_s=radius, sigma_r=0.24)
    except Exception:
        smoothed = cv2.bilateralFilter(rgb, d=0, sigmaColor=48, sigmaSpace=radius)
    protect = _detail_protection_mask(rgb)
    protect = 1.0 - (1.0 - protect) * (params.wrinkle_detail_protection / 100.0)
    alpha = (wrinkle_mask * protect * 0.55 * (params.wrinkle_blend / 100.0))[:, :, None]
    return np.clip(rgb.astype(np.float32) * (1 - alpha) + smoothed.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def _grabcut_subject_mask(rgb: np.ndarray, seed: np.ndarray) -> np.ndarray:
    small_rgb, scale = _small_mask(rgb, 820)
    if scale != 1.0:
        small_seed = cv2.resize(seed, (small_rgb.shape[1], small_rgb.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        small_seed = seed.copy()

    h, w = small_seed.shape
    if h < 16 or w < 16:
        return seed

    grab_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    grab_mask[small_seed > 0.18] = cv2.GC_PR_FGD
    grab_mask[small_seed > 0.62] = cv2.GC_FGD

    border = max(2, int(min(h, w) * 0.035))
    weak = small_seed < 0.28
    grab_mask[:border, :][weak[:border, :]] = cv2.GC_BGD
    grab_mask[-border:, :][weak[-border:, :]] = cv2.GC_BGD
    grab_mask[:, :border][weak[:, :border]] = cv2.GC_BGD
    grab_mask[:, -border:][weak[:, -border:]] = cv2.GC_BGD

    if not np.any(grab_mask == cv2.GC_FGD):
        yy0, yy1 = int(h * 0.20), int(h * 0.84)
        xx0, xx1 = int(w * 0.22), int(w * 0.78)
        grab_mask[yy0:yy1, xx0:xx1] = np.maximum(grab_mask[yy0:yy1, xx0:xx1], cv2.GC_PR_FGD)
        strong_y0, strong_y1 = int(h * 0.36), int(h * 0.68)
        strong_x0, strong_x1 = int(w * 0.36), int(w * 0.64)
        grab_mask[strong_y0:strong_y1, strong_x0:strong_x1] = cv2.GC_FGD

    if np.count_nonzero(grab_mask == cv2.GC_FGD) < 8 or np.count_nonzero(grab_mask == cv2.GC_BGD) < 8:
        return seed

    try:
        bgr = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2BGR)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, grab_mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        refined = np.where((grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        if scale != 1.0:
            refined = cv2.resize(refined, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        area = float(refined.mean())
        if 0.01 < area < 0.88:
            return np.clip(refined, 0, 1)
    except Exception as exc:
        logger.debug("OpenCV GrabCut subject refinement failed: %s", exc)
    return seed


def _subject_mask_for_depth_blur(
    rgb: np.ndarray,
    skin: np.ndarray,
    params: RetouchAdjustments,
    face_boxes: Optional[list[tuple[int, int, int, int]]] = None,
) -> np.ndarray:
    face_prior = _face_subject_prior(skin.shape, face_boxes)
    protect_scale = params.background_subject_protection / 100.0
    saliency = _optional_birefnet_subject_mask(rgb)
    if saliency is not None:
        base = np.maximum(saliency * protect_scale, face_prior * 0.85 * protect_scale)
    else:
        portrait = _portrait_like_mask(rgb, skin)
        prior = _central_subject_prior(rgb)
        prior = np.maximum(prior * 0.78, face_prior)
        if portrait.max() > 0.05:
            base = np.maximum(portrait * protect_scale, prior * 0.25 * protect_scale)
        else:
            base = prior * 0.72 * protect_scale
        base = _grabcut_subject_mask(rgb, np.clip(base, 0, 1))

    base = _remove_tiny_components(base)
    binary = (base > 0.18).astype(np.uint8)
    h, w = binary.shape
    expand = max(1, int(min(h, w) * 0.018 * (params.background_subject_expand / 100.0)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand | 1, expand | 1))
    protected = cv2.dilate(binary, kernel, iterations=1).astype(np.float32)
    sigma = max(0.1, max(2.5, min(h, w) / 160) * (params.background_edge_feather / 100.0))
    protected = cv2.GaussianBlur(protected, (0, 0), sigmaX=sigma, sigmaY=sigma)
    protected = np.maximum(protected, base)
    protected = np.maximum(protected, face_prior * 0.92 * protect_scale)
    return np.clip(protected, 0, 1)


def _perspective_depth_map(subject: np.ndarray) -> np.ndarray:
    h, w = subject.shape
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    vertical_far = np.repeat(np.clip(1.12 - y * 0.88, 0.18, 1.0), w, axis=1)

    background = (subject < 0.32).astype(np.uint8)
    distance = cv2.distanceTransform(background, cv2.DIST_L2, 3).astype(np.float32)
    distance = np.clip(distance / max(12.0, min(h, w) * 0.33), 0, 1)

    depth = 0.58 * vertical_far + 0.42 * distance
    coords = np.argwhere(subject > 0.45)
    if coords.size:
        top, left = coords.min(axis=0)
        bottom, right = coords.max(axis=0)
        above = np.clip((top - np.arange(h, dtype=np.float32))[:, None] / max(1.0, h * 0.35), 0, 1)
        side_gap = np.zeros((h, w), dtype=np.float32)
        if left > 0:
            side_gap[:, :left] = np.linspace(1, 0, left, dtype=np.float32)[None, :]
        if right + 1 < w:
            side_gap[:, right + 1:] = np.linspace(0, 1, w - right - 1, dtype=np.float32)[None, :]
        depth = np.maximum(depth, above * 0.95)
        depth = np.maximum(depth, side_gap * 0.55)
        lower_foreground = np.clip((np.arange(h, dtype=np.float32)[:, None] - bottom) / max(1.0, h * 0.30), 0, 1)
        depth *= 1.0 - lower_foreground * 0.36

    return np.clip(cv2.GaussianBlur(depth, (0, 0), max(2, min(h, w) / 220)), 0, 1)


def _blurred_layer(rgb: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.1:
        return rgb
    kernel = max(3, int(round(sigma * 6)) | 1)
    return cv2.GaussianBlur(rgb, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)


def _apply_depth_background_blur(
    rgb: np.ndarray,
    subject: np.ndarray,
    params: RetouchAdjustments,
    strength: float,
) -> np.ndarray:
    if subject.max() <= 0.05:
        return rgb
    depth = _optional_depth_map(rgb)
    heuristic = _perspective_depth_map(subject)
    if depth is None:
        depth = heuristic
    else:
        model_weight = params.background_model_depth_weight / 100.0
        depth = np.clip(depth * model_weight + heuristic * (1.0 - model_weight), 0, 1)
    depth = np.clip(depth * (params.background_depth_strength / 100.0), 0, 1)
    matte = np.clip((1.0 - subject) * (0.24 + 0.76 * depth), 0, 1)
    extra_foreground_protection = max(0.0, (params.background_foreground_protection - 100.0) / 100.0)
    if extra_foreground_protection > 0:
        h = matte.shape[0]
        lower = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        protection = np.clip(lower * extra_foreground_protection * 0.36, 0, 0.72)
        matte = matte * (1.0 - protection)
    blur_amount = np.clip(matte * (0.35 + 0.65 * strength), 0, 1)

    near_scale = params.background_near_blur / 100.0
    mid_scale = params.background_mid_blur / 100.0
    far_scale = params.background_far_blur / 100.0
    near = _blurred_layer(rgb, (1.6 + strength * 6.5) * near_scale)
    mid = _blurred_layer(rgb, (3.2 + strength * 15.0) * mid_scale)
    far = _blurred_layer(rgb, (6.0 + strength * 30.0) * far_scale)

    out = rgb.astype(np.float32)
    alpha_near = np.clip(blur_amount / 0.32, 0, 1)[:, :, None] * 0.70
    alpha_mid = np.clip((blur_amount - 0.24) / 0.42, 0, 1)[:, :, None] * 0.82
    alpha_far = np.clip((blur_amount - 0.56) / 0.36, 0, 1)[:, :, None] * 0.95
    out = out * (1 - alpha_near) + near.astype(np.float32) * alpha_near
    out = out * (1 - alpha_mid) + mid.astype(np.float32) * alpha_mid
    out = out * (1 - alpha_far) + far.astype(np.float32) * alpha_far
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_portrait_effects(
    rgb: np.ndarray,
    params: RetouchAdjustments,
    face_boxes: Optional[list[tuple[int, int, int, int]]] = None,
    face_landmarks: Optional[list[Optional[np.ndarray]]] = None,
) -> np.ndarray:
    result = rgb
    skin = _skin_mask(result, params.beauty_skin_mask_strength, params.beauty_skin_mask_feather)
    effect_mask: Optional[np.ndarray] = None
    if params.selected_face_ids is not None:
        effect_mask = _selected_face_effect_mask(result.shape[:2], face_boxes)
        skin = skin * effect_mask
    blemish_strength = max(params.face_blemish, params.body_blemish) / 100.0
    wrinkle_strength = params.face_wrinkle / 100.0
    texture_strength = params.skin_texture / 100.0

    smooth_amount = max(params.smooth_skin / 100.0, blemish_strength * 0.65, wrinkle_strength * 0.45)
    if smooth_amount > 0 and skin.max() > 0.05:
        strength = smooth_amount
        smooth = cv2.bilateralFilter(
            result,
            d=0,
            sigmaColor=(28 + 52 * strength) * (params.beauty_smooth_color / 100.0),
            sigmaSpace=(12 + 24 * strength) * (params.beauty_smooth_radius / 100.0),
        )
        protect = _detail_protection_mask(result)
        protect = 1.0 - (1.0 - protect) * (params.beauty_detail_protection / 100.0)
        alpha = (skin * protect * (0.12 + 0.42 * strength) * (params.beauty_smooth_blend / 100.0))[:, :, None]
        result = np.clip(result.astype(np.float32) * (1 - alpha) + smooth.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    wrinkle_mask = _wrinkle_zone_mask(result.shape[:2], params, face_boxes, face_landmarks)
    result = _apply_targeted_wrinkle_smoothing(result, wrinkle_mask, params)

    tone_amount = params.skin_tone / 100.0
    if (params.whiten_skin > 0 or abs(tone_amount) > 0.01) and skin.max() > 0.05:
        strength = params.whiten_skin / 100.0
        source = result.astype(np.float32)
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
        alpha = (skin * (0.18 + 0.50 * max(strength, abs(tone_amount))) * (params.beauty_whiten_blend / 100.0))[:, :, None]
        hsv[:, :, 1:2] *= 1 - 0.30 * max(strength, abs(tone_amount)) * (params.beauty_whiten_saturation / 100.0)
        hsv[:, :, 2:3] *= 1 + ((0.18 * strength * (params.beauty_whiten_brightness / 100.0)) + 0.10 * tone_amount)
        toned = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        result = np.clip(source * (1 - alpha) + toned * alpha, 0, 255).astype(np.uint8)
        if abs(tone_amount) > 0.01:
            adjusted = _apply_temperature(result, tone_amount * 45 * (params.beauty_skin_tone_temperature / 100.0))
            alpha_rgb = (skin * min(0.45, abs(tone_amount) * 0.45))[:, :, None]
            result = np.clip(result.astype(np.float32) * (1 - alpha_rgb) + adjusted.astype(np.float32) * alpha_rgb, 0, 255).astype(np.uint8)

    if texture_strength > 0 and skin.max() > 0.05:
        radius = max(0.1, 1.2 * (params.beauty_texture_radius / 100.0))
        amount = 0.35 * (params.beauty_texture_amount / 100.0)
        detail = cv2.addWeighted(result, 1.0 + amount, cv2.GaussianBlur(result, (0, 0), radius), -amount, 0)
        alpha = (skin * 0.22 * texture_strength * (params.beauty_texture_amount / 100.0))[:, :, None]
        result = np.clip(result.astype(np.float32) * (1 - alpha) + detail.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    enhance_strength = max(abs(params.eyes), params.eye_enhance, params.teeth, abs(params.eyebrow), abs(params.nose), abs(params.mouth), abs(params.face_shape), params.close_mouth, params.face_fullness) / 100.0
    if enhance_strength > 0:
        non_skin_detail = (1.0 - np.clip(skin * 1.4, 0, 1))
        if effect_mask is not None:
            non_skin_detail = non_skin_detail * effect_mask
        non_skin_detail = non_skin_detail[:, :, None]
        feature_amount = 0.22 * (params.beauty_feature_detail / 100.0)
        feature_radius = max(0.1, 1.0 * (params.beauty_feature_radius / 100.0))
        detail = cv2.addWeighted(result, 1.0 + feature_amount, cv2.GaussianBlur(result, (0, 0), feature_radius), -feature_amount, 0)
        alpha = non_skin_detail * min(0.20, enhance_strength * 0.20 * (params.beauty_feature_detail / 100.0))
        result = np.clip(result.astype(np.float32) * (1 - alpha) + detail.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    if params.teeth > 0:
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
        value_threshold = np.clip(145 * (params.beauty_teeth_threshold / 100.0), 70, 230)
        sat_threshold = np.clip(95 * (2.0 - params.beauty_teeth_threshold / 100.0), 35, 180)
        bright_low_sat = ((hsv[:, :, 2] > value_threshold) & (hsv[:, :, 1] < sat_threshold)).astype(np.float32)
        if effect_mask is not None:
            bright_low_sat *= effect_mask
        alpha = cv2.GaussianBlur(bright_low_sat, (0, 0), 2)[:, :, None] * (params.teeth / 100.0) * 0.18
        hsv[:, :, 1:2] *= (1 - alpha * 0.35 * (params.beauty_teeth_saturation / 100.0))
        hsv[:, :, 2:3] *= (1 + alpha * (params.beauty_teeth_brightness / 100.0))
        result = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

    result = _apply_hair_effects(result, params, skin, effect_mask)

    if params.background_blur > 0:
        subject = _subject_mask_for_depth_blur(result, skin, params, face_boxes)
        result = _apply_depth_background_blur(result, subject, params, params.background_blur / 100.0)

    return result


def _decode_mask(mask_base64: str, size: tuple[int, int]) -> np.ndarray:
    try:
        payload = mask_base64.split(",", 1)[-1]
        raw = base64.b64decode(payload)
        mask = Image.open(BytesIO(raw)).convert("L").resize(size, Image.Resampling.BILINEAR)
        arr = np.array(mask)
        return np.where(arr > 8, 255, 0).astype(np.uint8)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid inpaint mask") from exc


def _apply_inpaint(rgb: np.ndarray, mask_base64: Optional[str], radius_percent: float = 100) -> np.ndarray:
    if not mask_base64:
        return rgb
    mask = _decode_mask(mask_base64, (rgb.shape[1], rgb.shape[0]))
    if mask.max() == 0:
        return rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    repaired = cv2.inpaint(bgr, mask, max(1, 3 * (radius_percent / 100.0)), cv2.INPAINT_TELEA)
    return cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)


def _process_geometry(img: Image.Image, params: RetouchAdjustments) -> Image.Image:
    out = img.copy()
    rotate = params.rotate % 360
    if rotate:
        out = out.rotate(-rotate, expand=True, resample=Image.Resampling.BICUBIC)
    if params.flip_horizontal:
        out = ImageOps.mirror(out)
    if params.flip_vertical:
        out = ImageOps.flip(out)
    return out


def _scale_face_boxes(
    face_boxes: Optional[list[tuple[int, int, int, int]]],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    if not face_boxes:
        return []
    source_w, source_h = source_size
    target_w, target_h = target_size
    if source_w <= 0 or source_h <= 0:
        return list(face_boxes)
    sx = target_w / float(source_w)
    sy = target_h / float(source_h)
    return [
        (int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy)))
        for x1, y1, x2, y2 in face_boxes
    ]


def _scale_face_landmarks(
    face_landmarks: Optional[list[Optional[np.ndarray]]],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[Optional[np.ndarray]]:
    if not face_landmarks:
        return []
    source_w, source_h = source_size
    target_w, target_h = target_size
    if source_w <= 0 or source_h <= 0:
        return [lm.copy() if lm is not None else None for lm in face_landmarks]
    scale = np.array([target_w / float(source_w), target_h / float(source_h)], dtype=np.float32)
    return [(lm.astype(np.float32) * scale).copy() if lm is not None else None for lm in face_landmarks]


def _transform_face_boxes(
    face_boxes: Optional[list[tuple[int, int, int, int]]],
    source_size: tuple[int, int],
    output_size: tuple[int, int],
    params: RetouchAdjustments,
) -> list[tuple[int, int, int, int]]:
    if not face_boxes:
        return []
    source_w, source_h = source_size
    rotate = params.rotate % 360
    boxes = []
    for x1, y1, x2, y2 in face_boxes:
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        if rotate == 90:
            corners = np.column_stack([source_h - corners[:, 1], corners[:, 0]])
        elif rotate == 180:
            corners = np.column_stack([source_w - corners[:, 0], source_h - corners[:, 1]])
        elif rotate == 270:
            corners = np.column_stack([corners[:, 1], source_w - corners[:, 0]])
        elif rotate:
            return []

        out_w, out_h = output_size
        if params.flip_horizontal:
            corners[:, 0] = out_w - corners[:, 0]
        if params.flip_vertical:
            corners[:, 1] = out_h - corners[:, 1]
        min_xy = np.floor(corners.min(axis=0)).astype(int)
        max_xy = np.ceil(corners.max(axis=0)).astype(int)
        boxes.append((
            max(0, min(out_w - 1, int(min_xy[0]))),
            max(0, min(out_h - 1, int(min_xy[1]))),
            max(1, min(out_w, int(max_xy[0]))),
            max(1, min(out_h, int(max_xy[1]))),
        ))
    return [box for box in boxes if box[2] > box[0] and box[3] > box[1]]


def _transform_face_landmarks(
    face_landmarks: Optional[list[Optional[np.ndarray]]],
    source_size: tuple[int, int],
    output_size: tuple[int, int],
    params: RetouchAdjustments,
) -> list[Optional[np.ndarray]]:
    if not face_landmarks:
        return []
    source_w, source_h = source_size
    out_w, out_h = output_size
    rotate = params.rotate % 360
    transformed = []
    for lm in face_landmarks:
        if lm is None:
            transformed.append(None)
            continue
        pts = lm.astype(np.float32).copy()
        if rotate == 90:
            pts = np.column_stack([source_h - pts[:, 1], pts[:, 0]]).astype(np.float32)
        elif rotate == 180:
            pts = np.column_stack([source_w - pts[:, 0], source_h - pts[:, 1]]).astype(np.float32)
        elif rotate == 270:
            pts = np.column_stack([pts[:, 1], source_w - pts[:, 0]]).astype(np.float32)
        elif rotate:
            return []
        if params.flip_horizontal:
            pts[:, 0] = out_w - pts[:, 0]
        if params.flip_vertical:
            pts[:, 1] = out_h - pts[:, 1]
        transformed.append(pts)
    return transformed


def _process_image(
    img: Image.Image,
    params: RetouchAdjustments,
    face_boxes: Optional[list[tuple[int, int, int, int]]] = None,
    face_landmarks: Optional[list[Optional[np.ndarray]]] = None,
    face_source_size: Optional[tuple[int, int]] = None,
) -> Image.Image:
    original_size = img.size
    scaled_faces = _scale_face_boxes(face_boxes, face_source_size or original_size, original_size)
    scaled_landmarks = _scale_face_landmarks(face_landmarks, face_source_size or original_size, original_size)
    out = _process_geometry(img, params)
    transformed_faces = _transform_face_boxes(scaled_faces, original_size, out.size, params)
    transformed_landmarks = _transform_face_landmarks(scaled_landmarks, original_size, out.size, params)
    if params.brightness:
        out = ImageEnhance.Brightness(out).enhance(_factor(params.brightness, 0.75))
    if params.contrast:
        out = ImageEnhance.Contrast(out).enhance(_factor(params.contrast, 0.65))
    if params.saturation:
        out = ImageEnhance.Color(out).enhance(_factor(params.saturation, 0.85))

    rgb = np.array(out.convert("RGB"))
    rgb = _apply_temperature(rgb, params.temperature)
    rgb = _apply_portrait_effects(rgb, params, transformed_faces, transformed_landmarks)
    rgb = _apply_inpaint(rgb, params.inpaint_mask_base64, params.beauty_inpaint_radius)
    out = Image.fromarray(rgb, mode="RGB")
    if params.crop:
        out = _apply_crop(out, params.crop)
    return out


def _jpeg_base64(img: Image.Image, quality: int = 98) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True, subsampling=0)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _safe_output_suffix(suffix: str) -> str:
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    if not safe_suffix.replace(".", "").replace("-", "").replace("_", "").isalnum():
        safe_suffix = ".retouch"
    return safe_suffix


def _next_output_paths(source_disk_path: str, source_db_path: str, suffix: str) -> tuple[str, str]:
    disk_src = Path(source_disk_path)
    db_src = Path(source_db_path)
    safe_suffix = _safe_output_suffix(suffix)
    base_name = f"{disk_src.stem}{safe_suffix}.jpg"
    disk_base = disk_src.with_name(base_name)
    if not disk_base.exists():
        return str(disk_base), str(db_src.with_name(base_name))
    for idx in range(2, 10000):
        name = f"{disk_src.stem}{safe_suffix}-{idx}.jpg"
        disk_candidate = disk_src.with_name(name)
        if not disk_candidate.exists():
            return str(disk_candidate), str(db_src.with_name(name))
    raise HTTPException(status_code=409, detail="Could not choose output filename")


def _thumbnail_bytes(img: Image.Image) -> bytes:
    thumb = img.copy()
    thumb.thumbnail((640, 640), Image.Resampling.LANCZOS)
    buf = BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _clone_photo_row(conn: sqlite3.Connection, source_row: sqlite3.Row, output_path: str, output_img: Image.Image):
    cols = [r[1] for r in conn.execute("PRAGMA table_info(photos)").fetchall()]
    source = dict(source_row)
    values = {c: source.get(c) for c in cols}
    values.update({
        "path": output_path,
        "filename": os.path.basename(output_path),
        "image_width": output_img.width,
        "image_height": output_img.height,
        "thumbnail": _thumbnail_bytes(output_img),
        "scanned_at": datetime.now().isoformat(timespec="seconds"),
        "phash": None,
        "clip_embedding": None,
        "is_burst_lead": 1,
        "burst_group_id": None,
        "duplicate_group_id": None,
        "is_duplicate_lead": 1,
        "is_favorite": 0,
        "is_rejected": 0,
        "star_rating": 0,
    })
    insert_cols = [c for c in cols if c in values]
    placeholders = ", ".join(f":{c}" for c in insert_cols)
    conn.execute(
        f"INSERT OR REPLACE INTO photos ({', '.join(insert_cols)}) VALUES ({placeholders})",
        {c: values[c] for c in insert_cols},
    )


def _record_edit(conn: sqlite3.Connection, original_path: str, output_path: str, params: RetouchAdjustments):
    conn.execute(
        """INSERT INTO retouch_edits (original_path, output_path, params_json, exif_strategy)
           VALUES (?, ?, ?, ?)""",
        [
            original_path,
            output_path,
            json.dumps(params.model_dump(), ensure_ascii=False),
            "orientation_applied_metadata_not_preserved",
        ],
    )


def _ensure_user_preference(conn: sqlite3.Connection, output_path: str, user: CurrentUser):
    if not user.user_id or not is_multi_user_enabled():
        return
    conn.execute(
        """INSERT OR IGNORE INTO user_preferences
           (user_id, photo_path, star_rating, is_favorite, is_rejected)
           VALUES (?, ?, 0, 0, 0)""",
        [user.user_id, output_path],
    )


@router.post("/api/retouch/preview")
def api_retouch_preview(
    body: RetouchPreviewBody,
    user: CurrentUser = Depends(require_edition),
):
    db_path = _photo_path(body)
    with get_db() as conn:
        row, disk_path = _resolve_visible_path(conn, db_path, user)
        selected_face_ids = _selected_face_id_set(body.params)
        face_boxes = _face_boxes_for_photo(conn, db_path, selected_face_ids)
        face_landmarks = _face_landmarks_for_photo(conn, db_path, selected_face_ids)
    source_size = (row["image_width"], row["image_height"]) if row["image_width"] and row["image_height"] else None
    img = _downsample(_load_image(disk_path), body.max_size)
    if body.compare:
        result = _process_geometry(img, body.params)
    else:
        result = _process_image(img, body.params, face_boxes, face_landmarks, source_size)
    if body.compare and body.params.crop:
        result = _apply_crop(result, body.params.crop)
    return {
        "image_base64": _jpeg_base64(result),
        "width": result.width,
        "height": result.height,
        "background_blur_available": True,
        "mask_provider": "opencv_depth_perspective_birefnet_optional",
        "exif_strategy": "preview_applies_orientation",
    }


@router.post("/api/retouch/apply")
def api_retouch_apply(
    body: RetouchApplyBody,
    user: CurrentUser = Depends(require_edition),
):
    db_path = _photo_path(body)
    with get_db() as conn:
        source_row, disk_path = _resolve_visible_path(conn, db_path, user)
        selected_face_ids = _selected_face_id_set(body.params)
        face_boxes = _face_boxes_for_photo(conn, db_path, selected_face_ids)
        face_landmarks = _face_landmarks_for_photo(conn, db_path, selected_face_ids)
        source_size = (source_row["image_width"], source_row["image_height"]) if source_row["image_width"] and source_row["image_height"] else None
        img = _load_image(disk_path)
        result = _process_image(img, body.params, face_boxes, face_landmarks, source_size)
        output_disk, output_db_path = _next_output_paths(disk_path, db_path, body.output_suffix)
        try:
            result.save(output_disk, format="JPEG", quality=95, subsampling=0)
        except Exception as exc:
            logger.exception("Failed to save retouch copy")
            raise HTTPException(status_code=500, detail="Failed to save retouch copy") from exc

        _clone_photo_row(conn, source_row, output_db_path, result)
        _record_edit(conn, db_path, output_db_path, body.params)
        _ensure_user_preference(conn, output_db_path, user)
        conn.commit()

    return {
        "original_path": db_path,
        "output_path": output_db_path,
        "thumbnail_url": f"/thumbnail?path={output_db_path}&size=640",
        "exif_strategy": "orientation_applied_metadata_not_preserved",
    }


@router.post("/api/retouch/download")
def api_retouch_download(
    body: RetouchDownloadBody,
    user: CurrentUser = Depends(require_edition),
):
    db_path = _photo_path(body)
    with get_db() as conn:
        row, disk_path = _resolve_visible_path(conn, db_path, user)
        selected_face_ids = _selected_face_id_set(body.params)
        face_boxes = _face_boxes_for_photo(conn, db_path, selected_face_ids)
        face_landmarks = _face_landmarks_for_photo(conn, db_path, selected_face_ids)
        source_size = (row["image_width"], row["image_height"]) if row["image_width"] and row["image_height"] else None

    img = _load_image(disk_path)
    result = _process_image(img, body.params, face_boxes, face_landmarks, source_size)
    buf = BytesIO()
    result.save(buf, format="JPEG", quality=95, subsampling=0)
    buf.seek(0)
    stem = Path(db_path).stem
    download_name = f"{stem}.retouch.jpg"
    return StreamingResponse(
        buf,
        media_type="image/jpeg",
        headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
    )


@router.post("/api/retouch/mask/face")
def api_retouch_mask_face(
    body: RetouchMaskBody,
    user: CurrentUser = Depends(require_edition),
):
    db_path = _photo_path(body)
    with get_db() as conn:
        _, disk_path = _resolve_visible_path(conn, db_path, user)
    img = _downsample(_load_image(disk_path), body.max_size)
    rgb = np.array(img)
    skin = (_skin_mask(rgb) * 255).astype(np.uint8)
    portrait = (_portrait_like_mask(rgb, skin.astype(np.float32) / 255.0) * 255).astype(np.uint8)
    return {
        "available": True,
        "provider": "opencv_skin_fallback",
        "skin_mask_base64": _jpeg_base64(Image.fromarray(skin, mode="L"), quality=80),
        "portrait_mask_base64": _jpeg_base64(Image.fromarray(portrait, mode="L"), quality=80),
        "notes": "Lightweight local fallback; no face-parsing or MODNet model is required.",
    }


@router.post("/api/retouch/inpaint")
def api_retouch_inpaint(
    body: RetouchInpaintBody,
    user: CurrentUser = Depends(require_edition),
):
    params = body.params.model_copy(update={"inpaint_mask_base64": body.mask_base64})
    if body.save_copy:
        apply_body = RetouchApplyBody(photo_id=body.photo_id, image_path=body.image_path, params=params)
        return api_retouch_apply(apply_body, user)
    preview_body = RetouchPreviewBody(photo_id=body.photo_id, image_path=body.image_path, params=params)
    return api_retouch_preview(preview_body, user)
