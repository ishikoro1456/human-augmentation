from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


SegmentKind = Literal["cue", "speech"]


@dataclass(frozen=True)
class DemoSegment:
    kind: SegmentKind
    text: str
    audio: str = ""

    @property
    def is_speech(self) -> bool:
        return self.kind == "speech"

    @property
    def is_cue(self) -> bool:
        return self.kind == "cue"


@dataclass(frozen=True)
class DemoScript:
    script_id: str
    title: str
    language: str
    audio_dir: str
    segments: tuple[DemoSegment, ...]

    def speech_segments(self) -> list[DemoSegment]:
        return [segment for segment in self.segments if segment.is_speech]

    def spoken_context(self, *, upto_index: int, max_lines: int = 8) -> str:
        if max_lines <= 0:
            return "No spoken context yet."
        spoken = [
            segment.text
            for idx, segment in enumerate(self.segments)
            if idx <= int(upto_index) and segment.is_speech
        ]
        if not spoken:
            return "No spoken context yet."
        return "\n".join(spoken[-max_lines:])

    def resolve_audio_dir(self, *, script_path: Path) -> Path:
        base = script_path.parent
        return (base / self.audio_dir).resolve()


def load_demo_script(path: Path) -> DemoScript:
    raw = json.loads(path.read_text(encoding="utf-8"))
    script_id = str(raw.get("id", "")).strip()
    if not script_id:
        raise ValueError("demo script id is required")

    audio_dir = str(raw.get("audio_dir", "")).strip()
    if not audio_dir:
        raise ValueError("demo script audio_dir is required")

    segments_raw = raw.get("segments", [])
    if not isinstance(segments_raw, list) or not segments_raw:
        raise ValueError("demo script segments are required")

    segments: list[DemoSegment] = []
    for idx, item in enumerate(segments_raw):
        if not isinstance(item, dict):
            raise ValueError(f"segment[{idx}] must be an object")
        kind = str(item.get("kind", "")).strip()
        if kind not in ("cue", "speech"):
            raise ValueError(f"segment[{idx}] kind must be cue or speech")
        text = str(item.get("text", "")).strip()
        if not text:
            raise ValueError(f"segment[{idx}] text is required")
        audio = str(item.get("audio", "")).strip()
        if kind == "speech" and not audio:
            raise ValueError(f"segment[{idx}] audio is required for speech")
        segments.append(DemoSegment(kind=kind, text=text, audio=audio))

    return DemoScript(
        script_id=script_id,
        title=str(raw.get("title", "")).strip(),
        language=str(raw.get("language", "en")).strip() or "en",
        audio_dir=audio_dir,
        segments=tuple(segments),
    )
