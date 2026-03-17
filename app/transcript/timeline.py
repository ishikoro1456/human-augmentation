import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


_TIMESTAMP = re.compile(r"^\[(\d+):(\d{2})\]\s*(.*)$")


@dataclass(frozen=True)
class TranscriptSegment:
    t_sec: int
    text: str


class TranscriptTimeline:
    def __init__(self, segments: List[TranscriptSegment]) -> None:
        self._segments = sorted(segments, key=lambda s: s.t_sec)

    @classmethod
    def from_file(cls, path: Path) -> "TranscriptTimeline":
        segments: List[TranscriptSegment] = []
        raw_lines = path.read_text(encoding="utf-8").splitlines()
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            m = _TIMESTAMP.match(line)
            if not m:
                continue
            mm, ss, text = m.groups()
            t_sec = int(mm) * 60 + int(ss)
            segments.append(TranscriptSegment(t_sec=t_sec, text=text.strip()))
        if segments:
            return cls(segments)

        # タイムスタンプが無い場合は、平文の1行ずつを「短いチャンク」として扱う
        # 実際のSTTのように、短い単位で文章が流れてくる想定
        cps = 12.0  # chars per second (雑な目安)
        t_sec = 0
        for raw in raw_lines:
            text = raw.strip()
            if not text:
                continue
            segments.append(TranscriptSegment(t_sec=t_sec, text=text))
            n_chars = len(re.sub(r"\s+", "", text))
            est = int(round(max(1.0, n_chars / cps)))
            t_sec += max(1, est)
        return cls(segments)

    def window(self, current_sec: int, window_sec: int) -> List[TranscriptSegment]:
        start = max(0, current_sec - window_sec)
        return [s for s in self._segments if start <= s.t_sec <= current_sec]

    def latest_segment(self, current_sec: int) -> TranscriptSegment | None:
        candidates = [s for s in self._segments if s.t_sec <= current_sec]
        if not candidates:
            return None
        return candidates[-1]

    def segments(self) -> List[TranscriptSegment]:
        return list(self._segments)

    def history_sample(
        self,
        current_sec: int,
        window_sec: int,
        stride_sec: int,
        max_lines: int,
    ) -> List[TranscriptSegment]:
        history_end = max(0, current_sec - window_sec)
        history = [s for s in self._segments if s.t_sec <= history_end]
        if not history:
            return []
        buckets = {}
        for seg in history:
            key = seg.t_sec // stride_sec
            if key not in buckets:
                buckets[key] = seg
        sampled = [buckets[k] for k in sorted(buckets.keys())]
        if max_lines > 0:
            sampled = sampled[-max_lines:]
        return sampled

    def to_context(
        self,
        current_sec: int,
        window_sec: int,
        history_stride_sec: int = 120,
        history_max_lines: int = 40,
    ) -> str:
        if current_sec < 0:
            return "文字起こしはまだありません"
        history = self.history_sample(
            current_sec,
            window_sec,
            history_stride_sec,
            history_max_lines,
        )
        recent = self.window(current_sec, window_sec)
        if not history and not recent:
            return "文字起こしはまだありません"
        lines = []
        if history:
            lines.append("過去の要点:")
            lines.extend([f"[{s.t_sec:04d}s] {s.text}" for s in history])
        if recent:
            lines.append("直近の流れ:")
            lines.extend([f"[{s.t_sec:04d}s] {s.text}" for s in recent])
        return "\n".join(lines)
