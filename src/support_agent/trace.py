"""Per-conversation JSONL traces, redacted before they touch the disk."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .redact import Redactor


class Tracer(Protocol):
    def emit(self, event: str, **data: Any) -> None: ...


class JsonlTracer:
    """Appends one JSON object per event to ``<trace_dir>/<conversation_id>.jsonl``."""

    def __init__(self, trace_dir: str | Path, conversation_id: str, redactor: Redactor) -> None:
        self.path = Path(trace_dir) / f"{conversation_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conversation_id = conversation_id
        self.redactor = redactor
        self._seq = 0
        self._lock = threading.Lock()

    def emit(self, event: str, **data: Any) -> None:
        with self._lock:
            self._seq += 1
            record = {
                "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "conversation_id": self.conversation_id,
                "seq": self._seq,
                "event": event,
                **self.redactor.value(data),
            }
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class NullTracer:
    def emit(self, event: str, **data: Any) -> None:
        return None


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text("utf-8").splitlines() if line]
