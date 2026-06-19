import base64
import os
import sqlite3
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import create_app
from db.schema import init_database


@pytest.fixture()
def retouch_client(tmp_path, monkeypatch):
    db_path = tmp_path / "retouch.db"
    init_database(str(db_path))
    monkeypatch.setattr("api.database.DEFAULT_DB_PATH", str(db_path))
    monkeypatch.setattr("api.routers.retouch.get_visibility_clause", lambda user_id: ("1=1", []))
    monkeypatch.setattr("api.routers.retouch.resolve_photo_disk_path", lambda path: path)
    monkeypatch.setenv("FACET_RETOUCH_MODEL_MODE", "opencv")

    app = create_app()
    return TestClient(app), db_path


def _make_photo(tmp_path, db_path, size=(96, 72), filename="portrait.jpg"):
    img_path = tmp_path / filename
    img = Image.new("RGB", size, (180, 130, 105))
    img.save(img_path, format="JPEG", quality=95)
    original_bytes = img_path.read_bytes()

    thumb = Image.new("RGB", (32, 24), (180, 130, 105))
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO photos
           (path, filename, image_width, image_height, thumbnail, aggregate, aesthetic, is_burst_lead, is_duplicate_lead)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [str(img_path), img_path.name, size[0], size[1], b"thumb", 7.0, 7.0, 1, 1],
    )
    conn.commit()
    conn.close()
    return img_path, original_bytes


def _make_depth_photo(tmp_path, db_path, size=(160, 120), filename="depth_portrait.jpg"):
    img_path = tmp_path / filename
    img = Image.new("RGB", size, (70, 110, 155))
    pixels = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            stripe = 42 if (x // 5 + y // 7) % 2 else -28
            pixels[x, y] = (
                max(0, min(255, 72 + stripe + y // 6)),
                max(0, min(255, 118 + stripe // 2)),
                max(0, min(255, 168 - stripe // 2)),
            )
    cx, cy = size[0] // 2, int(size[1] * 0.48)
    for y in range(size[1]):
        for x in range(size[0]):
            if ((x - cx) / 32) ** 2 + ((y - cy) / 42) ** 2 <= 1:
                pixels[x, y] = (184, 132, 104)
    img.save(img_path, format="JPEG", quality=95)
    original_bytes = img_path.read_bytes()

    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO photos
           (path, filename, image_width, image_height, thumbnail, aggregate, aesthetic, is_burst_lead, is_duplicate_lead)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [str(img_path), img_path.name, size[0], size[1], b"thumb", 7.0, 7.0, 1, 1],
    )
    conn.commit()
    conn.close()
    return img_path, original_bytes


def _insert_face_box(db_path, photo_path, box):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO faces
           (photo_path, face_index, embedding, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [str(photo_path), 0, b"0" * (512 * 4), *box],
    )
    conn.commit()
    conn.close()


def test_retouch_preview_returns_base64(retouch_client, tmp_path):
    client, db_path = retouch_client
    img_path, _ = _make_photo(tmp_path, db_path)

    resp = client.post("/api/retouch/preview", json={
        "image_path": str(img_path),
        "params": {"brightness": 10, "contrast": 5, "smooth_skin": 20},
        "max_size": 640,
    })

    assert resp.status_code == 200
    body = resp.json()
    assert body["image_base64"].startswith("data:image/jpeg;base64,")
    assert body["width"] == 96
    assert body["height"] == 72


def test_retouch_preview_zero_max_size_keeps_original_dimensions(retouch_client, tmp_path):
    client, db_path = retouch_client
    img_path, _ = _make_photo(tmp_path, db_path, size=(2048, 1024), filename="large_portrait.jpg")

    resp = client.post("/api/retouch/preview", json={
        "image_path": str(img_path),
        "params": {"skin_tone": 20},
        "max_size": 0,
    })

    assert resp.status_code == 200
    body = resp.json()
    assert body["width"] == 2048
    assert body["height"] == 1024


def test_retouch_compare_preview_keeps_crop_but_skips_color_adjustments(retouch_client, tmp_path):
    client, db_path = retouch_client
    img_path, _ = _make_photo(tmp_path, db_path, size=(100, 80), filename="compare_portrait.jpg")

    resp = client.post("/api/retouch/preview", json={
        "image_path": str(img_path),
        "params": {
            "brightness": 80,
            "saturation": -80,
            "crop": {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5, "unit": "normalized"},
        },
        "max_size": 2048,
        "compare": True,
    })

    assert resp.status_code == 200
    body = resp.json()
    assert body["width"] == 50
    assert body["height"] == 40
    payload = body["image_base64"].split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(payload))) as img:
        pixel = img.convert("RGB").getpixel((img.width // 2, img.height // 2))
    assert all(abs(a - b) <= 3 for a, b in zip(pixel, (180, 130, 105)))


def test_retouch_apply_saves_copy_and_records_history(retouch_client, tmp_path):
    client, db_path = retouch_client
    img_path, original_bytes = _make_photo(tmp_path, db_path)

    resp = client.post("/api/retouch/apply", json={
        "image_path": str(img_path),
        "params": {"brightness": 12, "saturation": -5, "whiten_skin": 15},
    })

    assert resp.status_code == 200
    output_path = resp.json()["output_path"]
    assert output_path.endswith(".retouch.jpg")
    assert os.path.exists(output_path)
    assert img_path.read_bytes() == original_bytes

    conn = sqlite3.connect(db_path)
    photo = conn.execute("SELECT path, filename FROM photos WHERE path = ?", [output_path]).fetchone()
    edit = conn.execute(
        "SELECT original_path, output_path, params_json FROM retouch_edits WHERE output_path = ?",
        [output_path],
    ).fetchone()
    conn.close()

    assert photo is not None
    assert photo[1] == "portrait.retouch.jpg"
    assert edit is not None
    assert edit[0] == str(img_path)
    assert edit[1] == output_path
    assert "whiten_skin" in edit[2]


def test_retouch_download_returns_processed_jpeg_without_saving_photo(retouch_client, tmp_path):
    client, db_path = retouch_client
    img_path, original_bytes = _make_photo(tmp_path, db_path, size=(96, 72), filename="download_portrait.jpg")

    resp = client.post("/api/retouch/download", json={
        "image_path": str(img_path),
        "params": {"brightness": 20, "saturation": 10},
    })

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/jpeg")
    assert 'filename="download_portrait.retouch.jpg"' in resp.headers["content-disposition"]
    assert resp.content.startswith(b"\xff\xd8")
    assert img_path.read_bytes() == original_bytes

    conn = sqlite3.connect(db_path)
    retouch_rows = conn.execute("SELECT COUNT(*) FROM photos WHERE filename LIKE ?", ["%.retouch.jpg"]).fetchone()[0]
    edit_rows = conn.execute("SELECT COUNT(*) FROM retouch_edits").fetchone()[0]
    conn.close()
    assert retouch_rows == 0
    assert edit_rows == 0


def test_retouch_apply_rotates_flips_and_crops_copy(retouch_client, tmp_path):
    client, db_path = retouch_client
    img_path, original_bytes = _make_photo(tmp_path, db_path)

    resp = client.post("/api/retouch/apply", json={
        "image_path": str(img_path),
        "params": {
            "rotate": 90,
            "flip_horizontal": True,
            "flip_vertical": True,
            "crop": {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5, "unit": "normalized"},
        },
    })

    assert resp.status_code == 200
    output_path = resp.json()["output_path"]
    assert img_path.read_bytes() == original_bytes

    with Image.open(output_path) as out:
        assert out.size == (36, 48)

    conn = sqlite3.connect(db_path)
    edit = conn.execute(
        "SELECT params_json FROM retouch_edits WHERE output_path = ?",
        [output_path],
    ).fetchone()
    conn.close()

    assert edit is not None
    assert '"flip_horizontal": true' in edit[0]
    assert '"flip_vertical": true' in edit[0]


def test_retouch_accepts_extended_portrait_controls(retouch_client, tmp_path):
    client, db_path = retouch_client
    img_path, original_bytes = _make_photo(tmp_path, db_path)

    resp = client.post("/api/retouch/apply", json={
        "image_path": str(img_path),
        "params": {
            "face_blemish": 30,
            "face_wrinkle": 20,
            "body_blemish": 10,
            "skin_texture": 15,
            "skin_tone": 12,
            "face_fullness": 8,
            "face_shape": -5,
            "eyebrow": 6,
            "nose": -4,
            "eyes": 25,
            "mouth": 5,
            "close_mouth": 7,
            "teeth": 35,
            "eye_enhance": 30,
        },
    })

    assert resp.status_code == 200
    output_path = resp.json()["output_path"]
    assert os.path.exists(output_path)
    assert img_path.read_bytes() == original_bytes

    conn = sqlite3.connect(db_path)
    edit = conn.execute(
        "SELECT params_json FROM retouch_edits WHERE output_path = ?",
        [output_path],
    ).fetchone()
    conn.close()

    assert edit is not None
    assert '"face_blemish": 30.0' in edit[0]
    assert '"eye_enhance": 30.0' in edit[0]


def test_retouch_depth_background_blur_saves_copy(retouch_client, tmp_path):
    client, db_path = retouch_client
    img_path, original_bytes = _make_depth_photo(tmp_path, db_path)

    resp = client.post("/api/retouch/apply", json={
        "image_path": str(img_path),
        "params": {"background_blur": 78},
    })

    assert resp.status_code == 200
    output_path = resp.json()["output_path"]
    assert os.path.exists(output_path)
    assert img_path.read_bytes() == original_bytes

    with Image.open(img_path) as original, Image.open(output_path) as edited:
        assert edited.size == original.size
        original_bg = np.array(original.convert("RGB").crop((0, 0, 40, 40)), dtype=np.int16)
        edited_bg = np.array(edited.convert("RGB").crop((0, 0, 40, 40)), dtype=np.int16)
        diff = int(np.abs(original_bg - edited_bg).sum())
    assert diff > 1500


def test_retouch_background_blur_uses_face_box_as_subject_anchor(retouch_client, tmp_path):
    client, db_path = retouch_client
    img_path, _ = _make_depth_photo(tmp_path, db_path, size=(180, 120), filename="off_center_portrait.jpg")
    _insert_face_box(db_path, img_path, (118, 35, 142, 63))

    resp = client.post("/api/retouch/preview", json={
        "image_path": str(img_path),
        "params": {"background_blur": 85},
        "max_size": 180,
    })

    assert resp.status_code == 200
    body = resp.json()
    payload = body["image_base64"].split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(payload))) as edited, Image.open(img_path) as original:
        original_arr = np.array(original.convert("RGB"), dtype=np.int16)
        edited_arr = np.array(edited.convert("RGB"), dtype=np.int16)
        subject_diff = np.abs(original_arr[40:76, 110:152] - edited_arr[40:76, 110:152]).mean()
        background_diff = np.abs(original_arr[0:42, 0:58] - edited_arr[0:42, 0:58]).mean()

    assert subject_diff < background_diff


def test_retouch_accepts_depth_blur_advanced_controls(retouch_client, tmp_path):
    client, db_path = retouch_client
    img_path, _ = _make_depth_photo(tmp_path, db_path, filename="depth_controls.jpg")

    resp = client.post("/api/retouch/apply", json={
        "image_path": str(img_path),
        "params": {
            "background_blur": 65,
            "background_subject_protection": 120,
            "background_subject_expand": 130,
            "background_edge_feather": 80,
            "background_depth_strength": 140,
            "background_model_depth_weight": 35,
            "background_foreground_protection": 160,
            "background_near_blur": 45,
            "background_mid_blur": 110,
            "background_far_blur": 175,
        },
    })

    assert resp.status_code == 200
    output_path = resp.json()["output_path"]
    conn = sqlite3.connect(db_path)
    edit = conn.execute(
        "SELECT params_json FROM retouch_edits WHERE output_path = ?",
        [output_path],
    ).fetchone()
    conn.close()

    assert edit is not None
    assert '"background_depth_strength": 140.0' in edit[0]
    assert '"background_far_blur": 175.0' in edit[0]


def test_retouch_accepts_beauty_advanced_controls(retouch_client, tmp_path):
    client, db_path = retouch_client
    img_path, original_bytes = _make_photo(tmp_path, db_path, filename="beauty_controls.jpg")

    resp = client.post("/api/retouch/apply", json={
        "image_path": str(img_path),
        "params": {
            "smooth_skin": 35,
            "whiten_skin": 25,
            "skin_texture": 20,
            "teeth": 30,
            "eye_enhance": 25,
            "beauty_skin_mask_strength": 135,
            "beauty_skin_mask_feather": 85,
            "beauty_detail_protection": 140,
            "beauty_smooth_color": 125,
            "beauty_smooth_radius": 115,
            "beauty_smooth_blend": 90,
            "beauty_whiten_brightness": 130,
            "beauty_whiten_saturation": 75,
            "beauty_whiten_blend": 95,
            "beauty_skin_tone_temperature": 120,
            "beauty_texture_amount": 80,
            "beauty_texture_radius": 140,
            "beauty_feature_detail": 150,
            "beauty_feature_radius": 125,
            "beauty_teeth_brightness": 140,
            "beauty_teeth_saturation": 80,
            "beauty_teeth_threshold": 110,
            "beauty_inpaint_radius": 175,
        },
    })

    assert resp.status_code == 200
    output_path = resp.json()["output_path"]
    assert os.path.exists(output_path)
    assert img_path.read_bytes() == original_bytes

    conn = sqlite3.connect(db_path)
    edit = conn.execute(
        "SELECT params_json FROM retouch_edits WHERE output_path = ?",
        [output_path],
    ).fetchone()
    conn.close()

    assert edit is not None
    assert '"beauty_smooth_color": 125.0' in edit[0]
    assert '"beauty_inpaint_radius": 175.0' in edit[0]
