"""Local, non-destructive portrait retouch endpoints."""

from __future__ import annotations

import base64
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
    background_blur: float = Field(default=0, ge=0, le=100)
    inpaint_mask_base64: Optional[str] = None


class RetouchPreviewBody(BaseModel):
    photo_id: Optional[str] = None
    image_path: Optional[str] = None
    params: RetouchAdjustments = Field(default_factory=RetouchAdjustments)
    max_size: int = Field(default=0, ge=0, le=12000)


class RetouchApplyBody(BaseModel):
    photo_id: Optional[str] = None
    image_path: Optional[str] = None
    params: RetouchAdjustments = Field(default_factory=RetouchAdjustments)
    output_suffix: str = Field(default=".retouch")


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


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
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
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=5, sigmaY=5)
    return mask.astype(np.float32) / 255.0


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


def _apply_portrait_effects(rgb: np.ndarray, params: RetouchAdjustments) -> np.ndarray:
    result = rgb
    skin = _skin_mask(result)
    blemish_strength = max(params.face_blemish, params.body_blemish) / 100.0
    wrinkle_strength = params.face_wrinkle / 100.0
    texture_strength = params.skin_texture / 100.0

    smooth_amount = max(params.smooth_skin / 100.0, blemish_strength * 0.65, wrinkle_strength * 0.45)
    if smooth_amount > 0 and skin.max() > 0.05:
        strength = smooth_amount
        smooth = cv2.bilateralFilter(result, d=0, sigmaColor=28 + 52 * strength, sigmaSpace=12 + 24 * strength)
        protect = _detail_protection_mask(result)
        alpha = (skin * protect * (0.12 + 0.42 * strength))[:, :, None]
        result = np.clip(result.astype(np.float32) * (1 - alpha) + smooth.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    tone_amount = params.skin_tone / 100.0
    if (params.whiten_skin > 0 or abs(tone_amount) > 0.01) and skin.max() > 0.05:
        strength = params.whiten_skin / 100.0
        source = result.astype(np.float32)
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
        alpha = (skin * (0.18 + 0.50 * max(strength, abs(tone_amount))))[:, :, None]
        hsv[:, :, 1:2] *= 1 - 0.30 * max(strength, abs(tone_amount))
        hsv[:, :, 2:3] *= 1 + (0.18 * strength + 0.10 * tone_amount)
        toned = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        result = np.clip(source * (1 - alpha) + toned * alpha, 0, 255).astype(np.uint8)
        if abs(tone_amount) > 0.01:
            adjusted = _apply_temperature(result, tone_amount * 45)
            alpha_rgb = (skin * min(0.45, abs(tone_amount) * 0.45))[:, :, None]
            result = np.clip(result.astype(np.float32) * (1 - alpha_rgb) + adjusted.astype(np.float32) * alpha_rgb, 0, 255).astype(np.uint8)

    if texture_strength > 0 and skin.max() > 0.05:
        detail = cv2.addWeighted(result, 1.35, cv2.GaussianBlur(result, (0, 0), 1.2), -0.35, 0)
        alpha = (skin * 0.22 * texture_strength)[:, :, None]
        result = np.clip(result.astype(np.float32) * (1 - alpha) + detail.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    enhance_strength = max(abs(params.eyes), params.eye_enhance, params.teeth, abs(params.eyebrow), abs(params.nose), abs(params.mouth), abs(params.face_shape), params.close_mouth, params.face_fullness) / 100.0
    if enhance_strength > 0:
        non_skin_detail = (1.0 - np.clip(skin * 1.4, 0, 1))[:, :, None]
        detail = cv2.addWeighted(result, 1.22, cv2.GaussianBlur(result, (0, 0), 1.0), -0.22, 0)
        alpha = non_skin_detail * min(0.20, enhance_strength * 0.20)
        result = np.clip(result.astype(np.float32) * (1 - alpha) + detail.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    if params.teeth > 0:
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
        bright_low_sat = ((hsv[:, :, 2] > 145) & (hsv[:, :, 1] < 95)).astype(np.float32)
        alpha = cv2.GaussianBlur(bright_low_sat, (0, 0), 2)[:, :, None] * (params.teeth / 100.0) * 0.18
        hsv[:, :, 1:2] *= (1 - alpha * 0.35)
        hsv[:, :, 2:3] *= (1 + alpha)
        result = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

    if params.background_blur > 0:
        portrait = _portrait_like_mask(result, skin)
        if portrait.max() > 0.05:
            strength = params.background_blur / 100.0
            k = int(9 + strength * 42)
            if k % 2 == 0:
                k += 1
            blurred = cv2.GaussianBlur(result, (k, k), 0)
            bg_alpha = (1.0 - portrait)[:, :, None] * (0.35 + 0.65 * strength)
            result = np.clip(result.astype(np.float32) * (1 - bg_alpha) + blurred.astype(np.float32) * bg_alpha, 0, 255).astype(np.uint8)

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


def _apply_inpaint(rgb: np.ndarray, mask_base64: Optional[str]) -> np.ndarray:
    if not mask_base64:
        return rgb
    mask = _decode_mask(mask_base64, (rgb.shape[1], rgb.shape[0]))
    if mask.max() == 0:
        return rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    repaired = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)


def _process_image(img: Image.Image, params: RetouchAdjustments) -> Image.Image:
    out = img.copy()
    rotate = params.rotate % 360
    if rotate:
        out = out.rotate(-rotate, expand=True, resample=Image.Resampling.BICUBIC)
    if params.flip_horizontal:
        out = ImageOps.mirror(out)
    if params.flip_vertical:
        out = ImageOps.flip(out)
    if params.brightness:
        out = ImageEnhance.Brightness(out).enhance(_factor(params.brightness, 0.75))
    if params.contrast:
        out = ImageEnhance.Contrast(out).enhance(_factor(params.contrast, 0.65))
    if params.saturation:
        out = ImageEnhance.Color(out).enhance(_factor(params.saturation, 0.85))

    rgb = np.array(out.convert("RGB"))
    rgb = _apply_temperature(rgb, params.temperature)
    rgb = _apply_portrait_effects(rgb, params)
    rgb = _apply_inpaint(rgb, params.inpaint_mask_base64)
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
        _, disk_path = _resolve_visible_path(conn, db_path, user)
    img = _downsample(_load_image(disk_path), body.max_size)
    result = _process_image(img, body.params)
    return {
        "image_base64": _jpeg_base64(result),
        "width": result.width,
        "height": result.height,
        "background_blur_available": True,
        "mask_provider": "opencv_skin_fallback",
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
        img = _load_image(disk_path)
        result = _process_image(img, body.params)
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
