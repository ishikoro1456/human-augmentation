from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class TraceWriter:
    path: Path

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, event: Dict[str, object], *, ts: Optional[float] = None) -> None:
        payload = dict(event)
        payload.setdefault("ts", round(time.time() if ts is None else float(ts), 3))
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def log(self, message: str, *, source: str = "app", ts: Optional[float] = None) -> None:
        self.write({"type": "log", "source": source, "message": message}, ts=ts)

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:
                pass
