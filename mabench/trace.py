"""Append-only JSONL trace, flushed per event so the demo can follow it live."""
import json
import time
from pathlib import Path


class Trace:
    def __init__(self, path=None):
        self.events = []
        self._file = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "w", encoding="utf-8")

    def emit(self, kind: str, role: str, **data) -> None:
        event = {"t": round(time.time(), 3), "kind": kind, "role": role, **data}
        self.events.append(event)
        if self._file:
            self._file.write(json.dumps(event) + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
