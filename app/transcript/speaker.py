from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
import hashlib
import time
from pathlib import Path
from typing import List, Literal, Optional

from openai import OpenAI

from app.audio.player import AudioPlayer
from app.runtime.status import StatusStore
from app.transcript.timeline import TranscriptSegment, TranscriptTimeline
from app.tts.openai_tts import synthesize_to_file


@dataclass
class SpeakerState:
    spoken: List[TranscriptSegment]
    current: Optional[TranscriptSegment]
    current_audio_path: Path | None
    current_audio_started_at: float | None
    current_audio_duration_s: float | None


@dataclass(frozen=True)
class TranscriptEvent:
    kind: Literal["segment_start", "segment_end", "done", "error"]
    idx: int
    segment: TranscriptSegment | None = None
    audio_path: Path | None = None
    resume: threading.Event | None = None
    error: str | None = None


class TranscriptSpeaker:
    def __init__(
        self,
        timeline: TranscriptTimeline,
        client: OpenAI,
        player: AudioPlayer,
        cache_dir: Path,
        event_queue: queue.Queue[TranscriptEvent] | None = None,
        status: StatusStore | None = None,
        model: str = "gpt-4o-mini-tts",
        voice: str = "alloy",
        response_format: str = "mp3",
        start_sec: int = 0,
    ) -> None:
        self._timeline = timeline
        self._client = client
        self._player = player
        self._cache_dir = cache_dir
        self._model = model
        self._voice = voice
        self._response_format = response_format
        self._start_sec = start_sec
        self._status = status
        self._event_queue = event_queue
        self._lock = threading.Lock()
        self._state = SpeakerState(
            spoken=[],
            current=None,
            current_audio_path=None,
            current_audio_started_at=None,
            current_audio_duration_s=None,
        )

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def get_current(self) -> Optional[TranscriptSegment]:
        with self._lock:
            return self._state.current

    def get_current_playback(self) -> dict[str, object] | None:
        now = time.time()
        with self._lock:
            seg = self._state.current
            path = self._state.current_audio_path
            started_at = self._state.current_audio_started_at
            duration_s = self._state.current_audio_duration_s
        if seg is None or path is None or started_at is None or duration_s is None:
            return None
        elapsed_s = max(0.0, now - float(started_at))
        remaining_s = max(0.0, float(duration_s) - elapsed_s)
        return {
            "t_sec": int(seg.t_sec),
            "text": str(seg.text),
            "audio_path": str(path),
            "duration_s": round(float(duration_s), 3),
            "elapsed_s": round(float(elapsed_s), 3),
            "remaining_s": round(float(remaining_s), 3),
            "started_at_ts": round(float(started_at), 3),
        }

    def get_spoken_context(self) -> str:
        with self._lock:
            parts = [s.text for s in self._state.spoken]
            if self._state.current:
                parts.append(self._state.current.text)
            if not parts:
                return "文字起こしはまだありません"
            return "\n".join(parts)

    def _run(self) -> None:
        segments = self._timeline.segments()
        emitted = 0
        for idx, segment in enumerate(segments):
            if segment.t_sec < self._start_sec:
                continue
            cache_key = hashlib.sha1(
                (
                    f"{segment.t_sec}\n{segment.text}\n"
                    f"{self._model}\n{self._voice}\n{self._response_format}"
                ).encode("utf-8")
            ).hexdigest()[:12]
            cache_path = self._cache_dir / f"{segment.t_sec:06d}_{cache_key}.{self._response_format}"
            if not cache_path.exists():
                try:
                    synthesize_to_file(
                        client=self._client,
                        text=segment.text,
                        out_path=cache_path,
                        model=self._model,
                        voice=self._voice,
                        response_format=self._response_format,
                    )
                except Exception as exc:
                    print(f"TTS生成に失敗しました: {exc}")
                    if self._status:
                        self._status.log(f"TTS生成に失敗しました: {exc}")
                    if self._event_queue:
                        self._event_queue.put(
                            TranscriptEvent(
                                kind="error",
                                idx=idx,
                                segment=segment,
                                error=f"TTS生成に失敗しました: {exc}",
                            )
                        )
                    continue
            with self._lock:
                self._state.current = segment
            if self._status:
                self._status.set_transcript_current(
                    t_sec=segment.t_sec,
                    text=segment.text,
                    audio_path=cache_path,
                )
            if self._event_queue:
                self._event_queue.put(
                    TranscriptEvent(
                        kind="segment_start",
                        idx=idx,
                        segment=segment,
                        audio_path=cache_path,
                    )
                )
            try:
                duration_s = self._player.estimate_duration_s(cache_path)
                if duration_s is None and idx < (len(segments) - 1):
                    duration_s = float(max(0, segments[idx + 1].t_sec - segment.t_sec))
                started_at = time.time()
                with self._lock:
                    self._state.current_audio_path = cache_path
                    self._state.current_audio_started_at = started_at
                    self._state.current_audio_duration_s = duration_s
                self._player.play_music_blocking(cache_path)
            except Exception as exc:
                print(f"TTS再生に失敗しました: {exc}")
                if self._status:
                    self._status.log(f"TTS再生に失敗しました: {exc}")
                if self._event_queue:
                    self._event_queue.put(
                        TranscriptEvent(
                            kind="error",
                            idx=idx,
                            segment=segment,
                            audio_path=cache_path,
                            error=f"TTS再生に失敗しました: {exc}",
                        )
                    )
                continue
            with self._lock:
                self._state.spoken.append(segment)
                self._state.current = None
                self._state.current_audio_path = None
                self._state.current_audio_started_at = None
                self._state.current_audio_duration_s = None
            if self._status:
                self._status.on_transcript_spoken(text=segment.text)
                self._status.clear_transcript_current()
            if self._event_queue:
                resume = threading.Event()
                self._event_queue.put(
                    TranscriptEvent(
                        kind="segment_end",
                        idx=idx,
                        segment=segment,
                        audio_path=cache_path,
                        resume=resume,
                    )
                )
                resume.wait()
            emitted += 1
        if self._event_queue:
            self._event_queue.put(TranscriptEvent(kind="done", idx=emitted))
