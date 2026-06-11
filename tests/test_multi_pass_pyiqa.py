"""Tests for the PyIQA pass -> DB column mapping in the multi-pass scorer.

Locks in two behaviours:

1. When TOPIQ is the *primary* aesthetic model (16gb/24gb profiles), the pass
   fills the dedicated ``topiq_score`` column in addition to
   ``aesthetic``/``quality_score``. The viewer's TOPIQ range filter
   (``min_topiq``/``max_topiq``) reads ``topiq_score``; without this it is
   always empty after a normal scan (only ``--score-topiq`` would fill it).
2. Supplementary PyIQA models map to their own dedicated columns and never
   touch ``topiq_score``.
"""

from processing.multi_pass import ChunkedMultiPassProcessor


class _StubScorer:
    def __init__(self, scores):
        self._scores = scores

    def score_batch(self, pil_imgs):
        return self._scores[:len(pil_imgs)]


def _run_pass(model_name, scores):
    # _pass_pyiqa only needs the class-level PYIQA_COLUMN_MAP, so bypass __init__.
    proc = ChunkedMultiPassProcessor.__new__(ChunkedMultiPassProcessor)
    images = {f"p{i}.jpg": {"pil": object()} for i in range(len(scores))}
    results = {path: {} for path in images}
    proc._pass_pyiqa(_StubScorer(scores), model_name, images, results)
    return results


class TestPyiqaColumnMapping:
    def test_topiq_primary_fills_topiq_score(self):
        r = _run_pass("topiq", [6.25, 4.5])["p0.jpg"]
        assert r["aesthetic"] == 6.25
        assert r["quality_score"] == 6.25
        assert r["scoring_model"] == "topiq"
        assert r["topiq_score"] == 6.25  # the fix: dedicated column is filled

    def test_supplementary_models_map_to_dedicated_columns(self):
        assert _run_pass("topiq_iaa", [7.0])["p0.jpg"] == {"aesthetic_iaa": 7.0}
        assert _run_pass("topiq_nr_face", [8.0])["p0.jpg"] == {"face_quality_iqa": 8.0}
        assert _run_pass("liqe", [3.0])["p0.jpg"] == {"liqe_score": 3.0}

    def test_supplementary_does_not_set_topiq_score(self):
        assert "topiq_score" not in _run_pass("liqe", [3.0])["p0.jpg"]

    def test_non_topiq_primary_does_not_set_topiq_score(self):
        r = _run_pass("clip_aesthetic", [5.5])["p0.jpg"]
        assert r["aesthetic"] == 5.5
        assert "topiq_score" not in r
