"""Tests for ``scripts/audit_i18n.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def i18n_root(tmp_path, monkeypatch):
    """Build a fake i18n/translations tree and point the auditor at it."""
    root = tmp_path / 'translations'
    root.mkdir()
    monkeypatch.setattr('scripts.audit_i18n.TRANSLATIONS_DIR', root)
    return root


def _write(path: Path, data: dict) -> None:
    with path.open('w') as f:
        json.dump(data, f)


def test_walk_yields_dotted_paths():
    from scripts.audit_i18n import _walk
    paths = list(_walk('', {'a': {'b': 'x', 'c': {'d': 'y'}}, 'e': 'z'}))
    assert set(paths) == {'a.b', 'a.c.d', 'e'}


def test_audit_reports_no_gaps_when_complete(i18n_root):
    _write(i18n_root / 'en.json', {'hello': 'Hello', 'nav': {'home': 'Home'}})
    _write(i18n_root / 'fr.json', {'hello': 'Bonjour', 'nav': {'home': 'Accueil'}})
    from scripts.audit_i18n import audit
    assert audit() == {}


def test_audit_reports_missing_top_level_keys(i18n_root):
    _write(i18n_root / 'en.json', {'hello': 'Hello', 'goodbye': 'Bye'})
    _write(i18n_root / 'fr.json', {'hello': 'Bonjour'})
    from scripts.audit_i18n import audit
    assert audit() == {'fr': ['goodbye']}


def test_audit_reports_missing_nested_keys(i18n_root):
    _write(i18n_root / 'en.json', {'nav': {'home': 'Home', 'about': 'About'}})
    _write(i18n_root / 'fr.json', {'nav': {'home': 'Accueil'}})
    from scripts.audit_i18n import audit
    assert audit() == {'fr': ['nav.about']}


def test_audit_skips_reference_language(i18n_root):
    _write(i18n_root / 'en.json', {'a': 1, 'b': 2})
    from scripts.audit_i18n import audit
    # No non-en files → empty result.
    assert audit() == {}


def test_audit_handles_multiple_languages(i18n_root):
    _write(i18n_root / 'en.json', {'a': 1, 'b': 2, 'c': 3})
    _write(i18n_root / 'fr.json', {'a': 1, 'b': 2})
    _write(i18n_root / 'de.json', {'a': 1})
    from scripts.audit_i18n import audit
    result = audit()
    assert result == {'fr': ['c'], 'de': ['b', 'c']}


def test_fix_inserts_empty_strings(i18n_root):
    _write(i18n_root / 'en.json', {'hello': 'Hello', 'nav': {'home': 'Home'}})
    _write(i18n_root / 'fr.json', {'hello': 'Bonjour'})
    from scripts.audit_i18n import audit, fix
    inserts = fix(audit())
    assert inserts == 1
    with (i18n_root / 'fr.json').open() as f:
        fr = json.load(f)
    assert fr['nav']['home'] == ''


def test_main_exits_zero_when_clean(i18n_root, capsys, monkeypatch):
    _write(i18n_root / 'en.json', {'hello': 'Hello'})
    _write(i18n_root / 'fr.json', {'hello': 'Bonjour'})
    monkeypatch.setattr(sys, 'argv', ['audit_i18n'])
    from scripts.audit_i18n import main
    assert main() == 0


def test_main_exits_nonzero_when_gaps(i18n_root, capsys, monkeypatch):
    _write(i18n_root / 'en.json', {'a': 1, 'b': 2})
    _write(i18n_root / 'fr.json', {'a': 1})
    monkeypatch.setattr(sys, 'argv', ['audit_i18n'])
    from scripts.audit_i18n import main
    assert main() == 1
