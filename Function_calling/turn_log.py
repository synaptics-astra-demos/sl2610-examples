"""Persistent JSONL diagnostics logger.

One JSON record per line. Default path: ``Function_calling/logs/turns.jsonl``.
Override via the ``CORAL_TURN_LOG`` env var.

Events:
    - turn_done     completed turn (source, user, raw_text, steps, latency_ms)
    - turn_failed   model raised before producing a turn
    - mic_toggle    user toggled the mic button
    - voice_text    ASR-transcribed utterance handed to _on_send
    - alarm_fired   hardware alarm callback
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "logs" / "turns.jsonl"


class TurnLogger:
    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            override = os.environ.get("CORAL_TURN_LOG")
            path = Path(override) if override else DEFAULT_LOG_PATH
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        logger.info("turn log: %s", self._path)

    @property
    def path(self) -> Path:
        return self._path

    def log(self, event: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
            **fields,
        }
        try:
            line = json.dumps(record, default=_json_default) + "\n"
        except (TypeError, ValueError):
            logger.exception("failed to serialize turn-log record %r", event)
            return
        try:
            with self._lock, self._path.open("a", encoding="utf-8") as f:
                f.write(line)
        except OSError:
            logger.exception("failed to append to turn log %s", self._path)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return repr(obj)
