from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class LiveTranscriptLine:
    received_ts: float
    text: str
    speaker_ts_ms: Optional[int] = None


class LiveTranscriptBuffer:
    def __init__(self, *, max_lines: int = 200) -> None:
        self._max_lines = max(1, int(max_lines))
        self._lines: List[LiveTranscriptLine] = []

    def add(self, *, text: str, speaker_ts_ms: Optional[int] = None, received_ts: Optional[float] = None) -> None:
        t = time.time() if received_ts is None else float(received_ts)
        line = LiveTranscriptLine(received_ts=t, text=str(text).strip(), speaker_ts_ms=speaker_ts_ms)
        if not line.text:
            return
        self._lines.append(line)
        if len(self._lines) > self._max_lines:
            self._lines = self._lines[-self._max_lines :]

    def latest_text(self) -> str:
        return self._lines[-1].text if self._lines else ""

    def latest_received_ts(self) -> Optional[float]:
        return self._lines[-1].received_ts if self._lines else None

    def context(self, *, max_lines: int = 10) -> str:
        if not self._lines:
            return "文字起こしはまだありません"
        n = max(1, int(max_lines))
        tail = self._lines[-n:]
        return "\n".join([l.text for l in tail])

