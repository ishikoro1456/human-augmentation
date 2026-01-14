from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class HumanSignalSnapshot:
    updated_at: Optional[float]
    latest: Dict[str, object] = field(default_factory=dict)
    last_present_at: Optional[float] = None
    last_present_signal: Dict[str, object] = field(default_factory=dict)

    def is_recent(self, *, now: Optional[float] = None, hold_sec: float = 1.0) -> bool:
        if self.last_present_at is None:
            return False
        now_ts = time.time() if now is None else float(now)
        return (now_ts - float(self.last_present_at)) <= float(hold_sec)


class HumanSignalStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._updated_at: Optional[float] = None
        self._latest: Dict[str, object] = {}
        self._last_present_at: Optional[float] = None
        self._last_present_signal: Dict[str, object] = {}

    def update(self, *, ts: float, signal: Dict[str, object]) -> None:
        with self._lock:
            self._updated_at = float(ts)
            self._latest = dict(signal) if signal else {}
            if bool(signal.get("present", False)):
                self._last_present_at = float(ts)
                self._last_present_signal = dict(signal)

    def snapshot(self) -> HumanSignalSnapshot:
        with self._lock:
            return HumanSignalSnapshot(
                updated_at=self._updated_at,
                latest=dict(self._latest),
                last_present_at=self._last_present_at,
                last_present_signal=dict(self._last_present_signal),
            )

