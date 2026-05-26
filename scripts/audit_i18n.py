"""Diff Facet translation files and report missing keys.

The reference language is English (``i18n/translations/en.json``). Every
other language is compared key-by-key and any missing dotted-path is
listed. Exits non-zero when gaps exist so the script doubles as a CI
check.

Usage::

    venv/bin/python scripts/audit_i18n.py
    venv/bin/python scripts/audit_i18n.py --json     # machine-readable
    venv/bin/python scripts/audit_i18n.py --fix      # add empty strings
                                                     # for missing keys
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSLATIONS_DIR = REPO_ROOT / "i18n" / "translations"
REFERENCE = "en"


def _walk(prefix: str, node) -> Iterable[str]:
    """Yield dotted paths for every scalar leaf in a nested JSON dict."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(f"{prefix}.{k}" if prefix else k, v)
    else:
        yield prefix


def _set_path(node: dict, path: str, value) -> None:
    """Set ``node[a][b][c] = value`` given dotted path ``a.b.c``."""
    parts = path.split('.')
    cur = node
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
        if not isinstance(cur, dict):
            return
    cur[parts[-1]] = value


def _load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _save(path: Path, data: dict) -> None:
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write('\n')


def audit() -> dict[str, list[str]]:
    """Return ``{language: [missing dotted paths]}`` for every non-en file."""
    ref_path = TRANSLATIONS_DIR / f"{REFERENCE}.json"
    ref_data = _load(ref_path)
    ref_keys = set(_walk('', ref_data))

    missing: dict[str, list[str]] = {}
    for path in sorted(TRANSLATIONS_DIR.glob('*.json')):
        lang = path.stem
        if lang == REFERENCE:
            continue
        data = _load(path)
        keys = set(_walk('', data))
        gaps = sorted(ref_keys - keys)
        if gaps:
            missing[lang] = gaps
    return missing


def fix(missing: dict[str, list[str]]) -> int:
    """Insert empty strings for every missing path. Returns count of inserts."""
    inserts = 0
    for lang, gaps in missing.items():
        path = TRANSLATIONS_DIR / f"{lang}.json"
        data = _load(path)
        for gap in gaps:
            _set_path(data, gap, "")
            inserts += 1
        _save(path, data)
    return inserts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--json', action='store_true', help='Emit machine-readable JSON'
    )
    parser.add_argument(
        '--fix', action='store_true',
        help='Add empty strings for missing keys (preserves order)'
    )
    args = parser.parse_args()

    missing = audit()

    if args.json:
        print(json.dumps(missing, indent=2, ensure_ascii=False))
        return 0 if not missing else 1

    if not missing:
        print(f"All translations match {REFERENCE}.json.")
        return 0

    for lang, gaps in missing.items():
        print(f"\n[{lang}] missing {len(gaps)} keys:")
        for g in gaps:
            print(f"  - {g}")

    if args.fix:
        n = fix(missing)
        print(f"\nInserted {n} empty placeholders. Re-run without --fix to verify.")
        return 0

    print(f"\nTotal gaps: {sum(len(v) for v in missing.values())} across {len(missing)} languages.")
    print("Run with --fix to add empty placeholders.")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
