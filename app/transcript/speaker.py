import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from app.audio.player import AudioPlayer
from app.transcript.timeline import TranscriptSegment, TranscriptTimeline
from app.tts.openai_tts import synthesize_to_file


@dataclass
class SpeakerState:
    spoken: List[TranscriptSegment]
    current: Optional[TranscriptSegment]


class TranscriptSpeaker:
    def __init__(
        self,
        timeline: TranscriptTimeline,
        client: OpenAI,
        player: AudioPlayer,
        cache_dir: Path,
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
        self._lock = threading.Lock()
        self._state = SpeakerState(spoken=[], current=None)

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def get_current(self) -> Optional[TranscriptSegment]:
        with self._lock:
            return self._state.current

    def get_spoken_context(self) -> str:
        with self._lock:
            if not self._state.spoken:
                return "文字起こしはまだありません"
            return "\n".join([s.text for s in self._state.spoken])

    def _run(self) -> None:
        segments = [s for s in self._timeline.segments() if s.t_sec >= self._start_sec]
        for idx, segment in enumerate(segments):
            cache_path = self._cache_dir / f"{idx:04d}.{self._response_format}"
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
                    continue
            with self._lock:
                self._state.current = segment
                self._state.spoken.append(segment)
            try:
                self._player.play(cache_path)
            except Exception as exc:
                print(f"TTS再生に失敗しました: {exc}")
