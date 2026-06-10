"""
Structured scan progress protocol.

The scan CLI emits machine-readable progress lines on stdout prefixed with
@FACET_PROGRESS; the viewer's scan reader thread parses them into the SSE
status payload while keeping the human-readable log untouched.
"""

import json
import sys
import time

PROGRESS_PREFIX = '@FACET_PROGRESS '

_last_emit = 0.0


def emit_progress(phase, current=None, total=None, current_file=None,
                  eta_seconds=None, extra=None, force=False):
    """Print a structured progress event, throttled to one per second.

    Args:
        phase: Pipeline phase name (gather|scoring|bursts|tagging|vec|done)
        current: Items completed in this phase
        total: Total items in this phase
        current_file: Optional file currently being processed
        eta_seconds: Optional ETA estimate
        extra: Optional dict merged into the event
        force: Emit even if the throttle window hasn't elapsed (phase changes)
    """
    global _last_emit
    now = time.monotonic()
    if not force and now - _last_emit < 1.0:
        return
    _last_emit = now
    event = {'phase': phase}
    if current is not None:
        event['current'] = current
    if total is not None:
        event['total'] = total
    if current_file is not None:
        event['current_file'] = str(current_file)
    if eta_seconds is not None:
        event['eta_seconds'] = round(eta_seconds)
    if extra:
        event.update(extra)
    print(PROGRESS_PREFIX + json.dumps(event), flush=True, file=sys.stdout)


def parse_progress_line(line):
    """Return the parsed event dict if the line carries a progress event.

    The prefix is searched anywhere in the line, not just at the start:
    tqdm progress bars end with a carriage return instead of a newline, so a
    progress event can land on the same text line as a tqdm fragment.
    """
    idx = line.find(PROGRESS_PREFIX)
    if idx == -1:
        return None
    try:
        return json.loads(line[idx + len(PROGRESS_PREFIX):])
    except (json.JSONDecodeError, ValueError):
        return None
