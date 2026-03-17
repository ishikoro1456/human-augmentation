from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from app.runtime.trace import TraceWriter


@dataclass
class TranscriptStatus:
    current_t_sec: Optional[int] = None
    current_text: str = ""
    spoken_count: int = 0
    spoken_tail: List[str] = field(default_factory=list)
    last_audio_path: Optional[Path] = None
    last_boundary_t_sec: Optional[int] = None
    last_boundary_text: str = ""
    last_boundary_ts: Optional[float] = None


@dataclass
class CalibrationStatus:
    still_summary: str = ""
    active_summary: str = ""
    warnings: List[str] = field(default_factory=list)
    finished_at: Optional[float] = None
    gesture_summaries: List[str] = field(default_factory=list)
    gesture_axis_map: str = ""
    gesture_finished_at: Optional[float] = None


@dataclass
class ImuStatus:
    last_motion_text: str = ""
    last_event: str = "none"
    last_ts: Optional[float] = None
    last_human_signal: str = ""
    last_human_signal_used: str = ""


@dataclass
class AgentStatus:
    last_choice_id: str = ""
    last_choice_text: str = ""
    last_reason: str = ""
    last_latency_ms: Optional[int] = None
    last_ts: Optional[float] = None


@dataclass
class AudioStatus:
    last_backchannel_path: Optional[Path] = None
    last_backchannel_played: Optional[bool] = None
    last_transcript_path: Optional[Path] = None
    speaker_playback_enabled: bool = True
    speaker_playback_started: bool = False
    speaker_last_audio_ts: Optional[float] = None
    speaker_last_rms: Optional[int] = None
    speaker_rms_mean_2s: Optional[float] = None
    speaker_rms_max_2s: Optional[int] = None


@dataclass
class UiStatus:
    guide: str = ""
    guide_ts: Optional[float] = None
    experiment_id: str = ""
    mode: str = ""
    talker_connected: bool = False
    talker_addr: str = ""
    human_menu: List[str] = field(default_factory=list)


@dataclass
class SessionStatus:
    started_at: float = field(default_factory=time.time)
    transcript: TranscriptStatus = field(default_factory=TranscriptStatus)
    calibration: CalibrationStatus = field(default_factory=CalibrationStatus)
    imu: ImuStatus = field(default_factory=ImuStatus)
    agent: AgentStatus = field(default_factory=AgentStatus)
    audio: AudioStatus = field(default_factory=AudioStatus)
    ui: UiStatus = field(default_factory=UiStatus)
    logs: List[str] = field(default_factory=list)


class StatusStore:
    def __init__(
        self,
        *,
        max_logs: int = 200,
        transcript_tail: int = 6,
        trace: TraceWriter | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._status = SessionStatus()
        self._max_logs = max_logs
        self._transcript_tail = transcript_tail
        self._trace = trace

    def snapshot(self) -> SessionStatus:
        with self._lock:
            return copy.deepcopy(self._status)

    def log(self, message: str) -> None:
        ts = time.time()
        with self._lock:
            self._status.logs.append(message)
            if len(self._status.logs) > self._max_logs:
                self._status.logs = self._status.logs[-self._max_logs :]
            trace = self._trace
        if trace:
            trace.log(message, source="status", ts=ts)

    def set_imu(self, *, motion_text: str, event: str, ts: float) -> None:
        with self._lock:
            self._status.imu.last_motion_text = motion_text
            self._status.imu.last_event = event
            self._status.imu.last_ts = ts

    def set_human_signal(self, *, text: str) -> None:
        with self._lock:
            self._status.imu.last_human_signal = text

    def set_human_signal_used(self, *, text: str) -> None:
        with self._lock:
            self._status.imu.last_human_signal_used = text

    def set_transcript_current(self, *, t_sec: int, text: str, audio_path: Path) -> None:
        with self._lock:
            self._status.transcript.current_t_sec = t_sec
            self._status.transcript.current_text = text
            self._status.transcript.last_audio_path = audio_path
            self._status.audio.last_transcript_path = audio_path

    def on_transcript_spoken(self, *, text: str) -> None:
        with self._lock:
            self._status.transcript.spoken_count += 1
            self._status.transcript.spoken_tail.append(text)
            if len(self._status.transcript.spoken_tail) > self._transcript_tail:
                self._status.transcript.spoken_tail = self._status.transcript.spoken_tail[
                    -self._transcript_tail :
                ]

    def clear_transcript_current(self) -> None:
        with self._lock:
            self._status.transcript.current_t_sec = None
            self._status.transcript.current_text = ""

    def set_transcript_boundary(self, *, t_sec: int, text: str, ts: float) -> None:
        with self._lock:
            self._status.transcript.last_boundary_t_sec = t_sec
            self._status.transcript.last_boundary_text = text
            self._status.transcript.last_boundary_ts = ts

    def set_calibration_summary(
        self,
        *,
        still_summary: str,
        active_summary: str,
        warnings: List[str],
        ts: float,
    ) -> None:
        with self._lock:
            self._status.calibration.still_summary = still_summary
            self._status.calibration.active_summary = active_summary
            self._status.calibration.warnings = list(warnings)
            self._status.calibration.finished_at = ts

    def set_gesture_calibration(
        self,
        *,
        summaries: List[str],
        axis_map: str,
        ts: float,
    ) -> None:
        with self._lock:
            self._status.calibration.gesture_summaries = list(summaries)
            self._status.calibration.gesture_axis_map = axis_map
            self._status.calibration.gesture_finished_at = ts

    def set_agent_decision(
        self,
        *,
        choice_id: str,
        choice_text: str,
        reason: str,
        latency_ms: int,
        ts: float,
    ) -> None:
        with self._lock:
            self._status.agent.last_choice_id = choice_id
            self._status.agent.last_choice_text = choice_text
            self._status.agent.last_reason = reason
            self._status.agent.last_latency_ms = latency_ms
            self._status.agent.last_ts = ts

    def set_ui_guide(self, *, text: str, ts: Optional[float] = None) -> None:
        now = time.time() if ts is None else float(ts)
        with self._lock:
            self._status.ui.guide = str(text)
            self._status.ui.guide_ts = now

    def set_experiment(self, *, experiment_id: str, mode: str) -> None:
        with self._lock:
            self._status.ui.experiment_id = str(experiment_id)
            self._status.ui.mode = str(mode)

    def set_talker_connection(self, *, connected: bool, addr: str = "", ts: Optional[float] = None) -> None:
        now = time.time() if ts is None else float(ts)
        with self._lock:
            self._status.ui.talker_connected = bool(connected)
            self._status.ui.talker_addr = str(addr)
            self._status.ui.guide_ts = now

    def set_human_menu(self, *, lines: List[str]) -> None:
        with self._lock:
            self._status.ui.human_menu = list(lines)

    def set_backchannel_playback(self, *, path: Path, played: bool) -> None:
        with self._lock:
            self._status.audio.last_backchannel_path = path
            self._status.audio.last_backchannel_played = played

    def clear_backchannel_playback(self) -> None:
        with self._lock:
            self._status.audio.last_backchannel_path = None
            self._status.audio.last_backchannel_played = None

    def set_speaker_audio(
        self,
        *,
        playback_enabled: bool,
        playback_started: bool,
        rms_last: Optional[int],
        rms_mean_2s: Optional[float],
        rms_max_2s: Optional[int],
        ts: float,
    ) -> None:
        with self._lock:
            self._status.audio.speaker_playback_enabled = bool(playback_enabled)
            self._status.audio.speaker_playback_started = bool(playback_started)
            self._status.audio.speaker_last_audio_ts = float(ts)
            self._status.audio.speaker_last_rms = None if rms_last is None else int(rms_last)
            self._status.audio.speaker_rms_mean_2s = None if rms_mean_2s is None else float(rms_mean_2s)
            self._status.audio.speaker_rms_max_2s = None if rms_max_2s is None else int(rms_max_2s)
