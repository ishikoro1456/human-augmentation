from __future__ import annotations

import json
import os
import queue
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

from openai import OpenAI

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:
    InMemorySaver = None

from app.agents.backchannel_graph import build_backchannel_graph
from app.audio.player import AudioPlayer
from app.core.catalog import load_catalog
from app.core.selector import find_audio_file
from app.core.types import BackchannelItem
from app.imu.buffer import ImuBuffer, ImuSample
from app.imu.device import DeviceProfile, probe_serial_format, read_device_imu_lines, resolve_serial_port
from app.imu.gesture_calibration import GestureCalibration, run_gesture_calibration
from app.imu.signal import detect_backchannel_signal
from app.imu.signal_store import HumanSignalStore
from app.runtime.status import StatusStore
from app.runtime.trace import TraceWriter

from .script import DemoScript


MANUAL_BACKCHANNEL_KEYS = (
    ("1", "01"),
    ("2", "02"),
    ("3", "03"),
    ("4", "04"),
    ("5", "05"),
    ("6", "06"),
    ("7", "07"),
    ("a", "08"),
    ("s", "09"),
    ("d", "10"),
    ("f", "11"),
    ("g", "12"),
    ("h", "13"),
)


def _extract_agent_reason(result: Dict[str, object]) -> str:
    selection = result.get("selection", {})
    if not isinstance(selection, dict):
        return ""
    reason = selection.get("decision_reason", "") or selection.get("reason", "")
    return str(reason) if reason else ""


@dataclass(frozen=True)
class DemoSpeechState:
    idx: int
    text: str
    audio_path: Path

    @property
    def t_sec(self) -> int:
        return int(self.idx)


class DemoSpeaker:
    def __init__(
        self,
        *,
        script: DemoScript,
        script_path: Path,
        player: AudioPlayer,
        status: StatusStore | None = None,
        trace: TraceWriter | None = None,
        interactive_cues: bool = True,
        cue_pause_sec: float = 0.2,
        emit: Callable[[str], None] | None = None,
    ) -> None:
        self._script = script
        self._audio_dir = script.resolve_audio_dir(script_path=script_path)
        self._player = player
        self._status = status
        self._trace = trace
        self._interactive_cues = bool(interactive_cues)
        self._cue_pause_sec = float(max(0.0, cue_pause_sec))
        self._emit = emit or print
        self._lock = threading.Lock()
        self._spoken: list[str] = []
        self._current: DemoSpeechState | None = None
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="demo-speaker")
        self._thread.start()

    def stop(self) -> None:
        self._player.stop()
        self._done.set()

    def wait(self) -> None:
        self._done.wait()

    def is_done(self) -> bool:
        return self._done.is_set()

    def get_current(self) -> DemoSpeechState | None:
        with self._lock:
            return self._current

    def get_spoken_context(self) -> str:
        with self._lock:
            parts = list(self._spoken)
            if self._current is not None:
                parts.append(self._current.text)
        if not parts:
            return "No spoken context yet."
        return "\n".join(parts[-8:])

    def _set_current(self, current: DemoSpeechState | None) -> None:
        with self._lock:
            self._current = current

    def _append_spoken(self, text: str) -> None:
        with self._lock:
            self._spoken.append(text)
            if len(self._spoken) > 32:
                self._spoken = self._spoken[-32:]

    def _run(self) -> None:
        try:
            for idx, segment in enumerate(self._script.segments):
                if segment.is_cue:
                    self._emit(f"[cue] {segment.text}")
                    if self._trace:
                        self._trace.write({"type": "demo_cue", "idx": int(idx), "text": segment.text})
                    if self._interactive_cues:
                        input("Enter で続けます > ")
                    elif self._cue_pause_sec > 0:
                        time.sleep(self._cue_pause_sec)
                    continue

                audio_path = self._audio_dir / segment.audio
                if not audio_path.exists():
                    raise FileNotFoundError(f"speech audio not found: {audio_path}")
                current = DemoSpeechState(idx=idx, text=segment.text, audio_path=audio_path)
                self._set_current(current)
                if self._status:
                    self._status.set_transcript_current(t_sec=idx, text=segment.text, audio_path=audio_path)
                    self._status.set_ui_guide(text="Please listen and react by nodding or shaking your head.")
                if self._trace:
                    self._trace.write(
                        {
                            "type": "demo_speech_start",
                            "idx": int(idx),
                            "text": segment.text,
                            "audio_path": str(audio_path),
                        }
                    )
                self._player.play_music_blocking(audio_path)
                self._append_spoken(segment.text)
                if self._status:
                    self._status.on_transcript_spoken(text=segment.text)
                    self._status.clear_transcript_current()
                if self._trace:
                    self._trace.write(
                        {
                            "type": "demo_speech_end",
                            "idx": int(idx),
                            "text": segment.text,
                            "audio_path": str(audio_path),
                        }
                    )
                self._set_current(None)
        finally:
            self._done.set()


class SensorOnlySpeaker:
    def __init__(
        self,
        *,
        status: StatusStore | None = None,
        trace: TraceWriter | None = None,
    ) -> None:
        self._status = status
        self._trace = trace
        self._done = threading.Event()
        self._current = DemoSpeechState(idx=0, text="Sensor-only listener reactions.", audio_path=Path())

    def start(self) -> None:
        if self._status:
            self._status.clear_transcript_current()
            self._status.set_ui_guide(text="Please nod or shake your head. The system replies in English.")
        if self._trace:
            self._trace.write(
                {
                    "type": "demo_sensor_only_start",
                    "text": self._current.text,
                }
            )

    def stop(self) -> None:
        self._done.set()

    def wait(self) -> None:
        self._done.wait()

    def is_done(self) -> bool:
        return self._done.is_set()

    def get_current(self) -> DemoSpeechState | None:
        return None if self._done.is_set() else self._current

    def get_spoken_context(self) -> str:
        return "Sensor-only mode. Use only the listener gesture."


class MeasurementControl:
    def __init__(self, *, enabled: bool) -> None:
        self._lock = threading.Lock()
        self._enabled = bool(enabled)

    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool) -> bool:
        value = bool(enabled)
        with self._lock:
            self._enabled = value
        return value

    def toggle(self) -> bool:
        with self._lock:
            self._enabled = not self._enabled
            return self._enabled


def _paused_signal() -> Dict[str, object]:
    return {
        "present": False,
        "gesture_hint": "other",
        "reason": "paused",
    }


def _manual_key_map(items: list[BackchannelItem], audio_dir: Path) -> Dict[str, tuple[BackchannelItem, Path]]:
    items_by_id = {item.id: item for item in items}
    mapping: Dict[str, tuple[BackchannelItem, Path]] = {}
    for key, item_id in MANUAL_BACKCHANNEL_KEYS:
        item = items_by_id.get(item_id)
        if item is None:
            continue
        audio_path = find_audio_file(audio_dir, item)
        if audio_path is None:
            continue
        mapping[key] = (item, audio_path)
    return mapping


def _manual_key_guide(mapping: Dict[str, tuple[BackchannelItem, Path]]) -> str:
    positive: list[str] = []
    negative: list[str] = []
    for key, item_id in MANUAL_BACKCHANNEL_KEYS:
        entry = mapping.get(key)
        if entry is None:
            continue
        item, _ = entry
        label = f"{key}:{item.text}"
        if item.directory == "positive":
            positive.append(label)
        else:
            negative.append(label)
    parts = []
    if positive:
        parts.append("positive " + " ".join(positive))
    if negative:
        parts.append("negative " + " ".join(negative))
    parts.append("Enter: ON/OFF")
    parts.append("q: 終了")
    return " | ".join(parts)


def _drain_signal_events(signal_events: queue.Queue) -> None:
    while True:
        try:
            signal_events.get_nowait()
        except queue.Empty:
            return


def _set_measurement_enabled(
    control: MeasurementControl,
    *,
    enabled: bool,
    player: AudioPlayer,
    signal_events: queue.Queue,
    emit: Callable[[str], None],
    trace: TraceWriter | None,
) -> bool:
    current = control.is_enabled()
    if current == bool(enabled):
        return current
    next_state = control.set_enabled(enabled)
    player.stop()
    _drain_signal_events(signal_events)
    if trace:
        trace.write({"type": "demo_measurement_toggle", "enabled": bool(next_state)})
    if next_state:
        emit("測定を ON にしました。頷くか首を振ると英語で返します。")
    else:
        emit("測定を OFF にしました。メガネを外せます。")
    return next_state


def _play_selected_backchannel(
    *,
    player: AudioPlayer,
    item: BackchannelItem,
    audio_path: Path,
    reason: str,
    interrupt: bool,
    status: StatusStore | None,
    trace: TraceWriter | None,
    debug_agent: bool = False,
) -> bool:
    restore_volume: float | None = None
    if player.is_music_playing():
        restore_volume = player.get_music_volume()
        player.set_music_volume(0.35)
    played = player.play_effect(audio_path, interrupt=interrupt)
    if status:
        status.set_backchannel_playback(path=audio_path, played=played)
        status.log(f"{reason}: {item.id} {item.text}")
    if trace:
        trace.write(
            {
                "type": "demo_backchannel_play",
                "reason": reason,
                "selected_id": item.id,
                "selected_text": item.text,
                "audio_path": str(audio_path),
                "played": bool(played),
            }
        )
    if debug_agent and not status:
        print(f"{reason}: {item.id} {item.text}")
    if played:
        while player.is_effect_playing():
            time.sleep(0.02)
    if restore_volume is not None:
        player.set_music_volume(restore_volume)
    return bool(played)


class _RawTtyReader:
    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._old = None

    def __enter__(self) -> "_RawTtyReader":
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def read_key(self, timeout_sec: float = 0.1) -> str:
        ready, _, _ = select.select([self._fd], [], [], max(0.0, float(timeout_sec)))
        if not ready:
            return ""
        raw = os.read(self._fd, 1)
        if not raw:
            return ""
        if raw == b"\x03":
            raise KeyboardInterrupt
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return ""


def _sensor_only_control_loop(
    *,
    control: MeasurementControl,
    speaker: SensorOnlySpeaker,
    player: AudioPlayer,
    signal_events: queue.Queue,
    manual_map: Dict[str, tuple[BackchannelItem, Path]],
    emit: Callable[[str], None],
    trace: TraceWriter | None,
    status: StatusStore | None,
    debug_agent: bool,
) -> None:
    try:
        with _RawTtyReader() as reader:
            while not speaker.is_done():
                key = reader.read_key(timeout_sec=0.1)
                if not key:
                    continue
                if key in ("\r", "\n"):
                    _set_measurement_enabled(
                        control,
                        enabled=not control.is_enabled(),
                        player=player,
                        signal_events=signal_events,
                        emit=emit,
                        trace=trace,
                    )
                    continue
                norm = key.lower()
                if norm == "q":
                    speaker.stop()
                    player.stop()
                    emit("停止します。")
                    return
                entry = manual_map.get(norm)
                if entry is None:
                    continue
                item, audio_path = entry
                _play_selected_backchannel(
                    player=player,
                    item=item,
                    audio_path=audio_path,
                    reason=f"manual key {norm}",
                    interrupt=True,
                    status=status,
                    trace=trace,
                    debug_agent=debug_agent,
                )
    except (EOFError, termios.error):
        _set_measurement_enabled(
            control,
            enabled=True,
            player=player,
            signal_events=signal_events,
            emit=emit,
            trace=trace,
        )
        emit("標準入力が読めないので、測定を ON のまま続けます。")
        return
    except KeyboardInterrupt:
        speaker.stop()
        player.stop()
        return


def _imu_loop_device(
    *,
    profile: DeviceProfile,
    override_port: str,
    override_baud: int | None,
    buffer: ImuBuffer,
    debug: bool = False,
    status: StatusStore | None = None,
) -> None:
    last_log = 0.0
    last_status = 0.0
    log_fn = status.log if status else None
    for reading in read_device_imu_lines(
        profile,
        override_port=override_port,
        override_baud=override_baud,
        on_log=log_fn,
    ):
        now = time.time()
        buffer.add(
            ImuSample(
                ts=now,
                ax=reading.ax,
                ay=reading.ay,
                az=reading.az,
                gx=reading.gx,
                gy=reading.gy,
                gz=reading.gz,
                has_acc=reading.has_acc,
            )
        )
        if status and (now - last_status) > 0.2:
            status.set_imu(motion_text=buffer.format_status_line(now=now), event="raw", ts=now)
            last_status = now
        if debug and (now - last_log) > 1.0:
            message = f"IMU受信中: {buffer.format_status_line(now=now)}"
            if status:
                status.log(message)
            else:
                print(message)
            last_log = now


def _human_signal_loop(
    buffer: ImuBuffer,
    store: HumanSignalStore,
    *,
    abs_threshold: float,
    max_age_s: float,
    min_consecutive_above: int,
    nod_axis: str,
    shake_axis: str,
    event_queue: queue.Queue,
    control: MeasurementControl | None = None,
    debug: bool = False,
    status: StatusStore | None = None,
) -> None:
    last_present: bool | None = None
    episode_fired = False
    last_debug: bool | None = None
    while True:
        now = time.time()
        if control is not None and not control.is_enabled():
            paused = _paused_signal()
            store.update(ts=now, signal=paused)
            if status:
                status.set_human_signal(text="停止中")
            episode_fired = False
            last_present = False
            time.sleep(0.1)
            continue
        imu_bundle = buffer.build_bundle(now=now, raw_window_sec=2.0, raw_max_points=24, stats_windows_sec=[1.0, 5.0])
        signal = detect_backchannel_signal(
            imu_bundle,
            calibration=None,
            abs_threshold=abs_threshold,
            max_age_s=max_age_s,
            min_consecutive_above=min_consecutive_above,
            nod_axis=nod_axis,
            shake_axis=shake_axis,
        )
        store.update(ts=now, signal=signal)
        present = bool(signal.get("present", False))
        hint = signal.get("gesture_hint")
        eligible = bool(present) and isinstance(hint, str) and hint in ("nod", "shake")

        if last_present is None:
            last_present = present
        if not present:
            episode_fired = False
        elif last_present is False and present:
            episode_fired = False

        if eligible and not episode_fired:
            event_queue.put({"ts": now, "signal": dict(signal)})
            episode_fired = True
        if status:
            status.set_human_signal(text=str(signal.get("reason", "")))
        if debug and last_debug != eligible:
            print(f"IMUサイン: {signal.get('reason', '')}")
            last_debug = eligible
        last_present = present
        time.sleep(0.1)


def _gesture_intensity(
    *,
    hint: str,
    signal: Dict[str, object],
    gesture_calib: GestureCalibration | None,
    nod_axis: str,
    shake_axis: str,
) -> Dict[str, object] | None:
    if gesture_calib is None or hint not in ("nod", "shake"):
        return None
    axis = nod_axis if hint == "nod" else shake_axis
    axis_mean_map = signal.get("axis_abs_mean_1s", {})
    if not isinstance(axis_mean_map, dict):
        return None
    current_axis_mean = axis_mean_map.get(axis)
    if not isinstance(current_axis_mean, (int, float)):
        return None

    weak_ex = gesture_calib.examples.get(f"{hint}_weak")
    strong_ex = gesture_calib.examples.get(f"{hint}_strong")
    weak_ref = None if weak_ex is None else weak_ex.axis_abs_mean.get(axis)
    strong_ref = None if strong_ex is None else strong_ex.axis_abs_mean.get(axis)
    if not isinstance(weak_ref, (int, float)) or not isinstance(strong_ref, (int, float)):
        return None
    if float(strong_ref) <= float(weak_ref):
        return None
    norm = (float(current_axis_mean) - float(weak_ref)) / (float(strong_ref) - float(weak_ref))
    norm = max(0.0, min(1.0, norm))
    level = max(1, min(5, int(round(1 + norm * 4))))
    return {
        "hint": hint,
        "axis": axis,
        "axis_abs_mean_1s": round(float(current_axis_mean), 3),
        "weak_ref": round(float(weak_ref), 3),
        "strong_ref": round(float(strong_ref), 3),
        "norm_0to1": round(float(norm), 3),
        "level_1to5": int(level),
    }


def _pick_demo_fallback_item(
    *,
    items: list[BackchannelItem],
    hint: str,
    gesture_intensity: Dict[str, object] | None,
    avoid_ids: list[str],
) -> BackchannelItem | None:
    directory = "positive" if hint == "nod" else "negative"
    target_level = gesture_intensity.get("level_1to5") if isinstance(gesture_intensity, dict) else None
    if isinstance(target_level, (int, float)):
        level = int(target_level)
    else:
        level = 1
    target_strength = min((1, 3, 5), key=lambda value: abs(value - level))
    avoid = {str(item_id) for item_id in avoid_ids}

    candidates = [item for item in items if item.directory == directory and item.id not in avoid]
    if not candidates:
        candidates = [item for item in items if item.directory == directory]
    if not candidates:
        return None

    exact = [item for item in candidates if item.strength == target_strength and item.nod == target_strength]
    if exact:
        return exact[0]

    near = sorted(
        candidates,
        key=lambda item: (abs(int(item.strength) - target_strength), abs(int(item.nod) - target_strength), item.id),
    )
    return near[0] if near else None


def _demo_backchannel_loop(
    signal_events: queue.Queue,
    *,
    control: MeasurementControl | None,
    graph: object | None,
    items: list,
    audio_dir: Path,
    imu_buffer: ImuBuffer,
    gesture_calib: GestureCalibration | None,
    nod_axis: str,
    shake_axis: str,
    speaker: DemoSpeaker,
    player: AudioPlayer,
    thread_id: str,
    agent_interval_sec: float,
    backchannel_cooldown_sec: float,
    debug_agent: bool = False,
    status: StatusStore | None = None,
    trace: TraceWriter | None = None,
) -> None:
    last_agent_call = 0.0
    last_backchannel_play = 0.0
    last_backchannel_text = ""
    recent_ids: list[str] = []
    recent_texts: list[str] = []

    while not speaker.is_done():
        if control is not None and not control.is_enabled():
            _drain_signal_events(signal_events)
            time.sleep(0.1)
            continue
        try:
            ev = signal_events.get(timeout=0.1)
        except queue.Empty:
            continue

        signal = ev.get("signal", {}) if isinstance(ev, dict) else {}
        ts = ev.get("ts") if isinstance(ev, dict) else None
        if not isinstance(signal, dict) or not isinstance(ts, (int, float)):
            continue
        if player.is_effect_playing():
            continue
        if (time.time() - last_backchannel_play) < float(backchannel_cooldown_sec):
            continue
        if (time.time() - last_agent_call) < float(agent_interval_sec):
            continue

        current = speaker.get_current()
        if current is None:
            continue

        hint = str(signal.get("gesture_hint", "other"))
        if hint not in ("nod", "shake"):
            continue

        imu_bundle = imu_buffer.build_bundle(
            now=float(ts),
            raw_window_sec=2.0,
            raw_max_points=8,
            stats_windows_sec=[1.0, 5.0, 30.0],
        )
        imu_bundle["human_signal"] = dict(signal)
        gesture_intensity = _gesture_intensity(
            hint=hint,
            signal=signal,
            gesture_calib=gesture_calib,
            nod_axis=nod_axis,
            shake_axis=shake_axis,
        )
        if gesture_intensity is not None:
            imu_bundle["gesture_intensity"] = gesture_intensity
        transcript_context = speaker.get_spoken_context()
        directory_allowlist = ["positive"] if hint == "nod" else ["negative"]
        avoid_ids = recent_ids[-2:] if recent_ids else []

        audio_state: Dict[str, object] = {
            "transcript_playing": player.is_music_playing(),
            "backchannel_playing": player.is_effect_playing(),
            "decision_point": "demo_on_human_signal",
        }
        recent_backchannel: Dict[str, object] = {
            "seconds_ago": None if last_backchannel_play == 0.0 else round(time.time() - last_backchannel_play, 3),
            "text": last_backchannel_text,
            "history_ids": recent_ids[-6:],
            "history_texts": recent_texts[-6:],
        }

        try:
            if graph is None:
                latency_ms = 0
                last_agent_call = time.time()
                fallback_item = _pick_demo_fallback_item(
                    items=items,
                    hint=hint,
                    gesture_intensity=gesture_intensity,
                    avoid_ids=avoid_ids,
                )
                selected_id = "" if fallback_item is None else fallback_item.id
                reason = f"sensor-only: {hint}"
            else:
                t0 = time.time()
                result = graph.invoke(
                    {
                        "utterance": current.text,
                        "imu": imu_bundle,
                        "imu_text": json.dumps(imu_bundle, ensure_ascii=False),
                        "audio_state": audio_state,
                        "recent_backchannel": recent_backchannel,
                        "utterance_t_sec": current.t_sec,
                        "transcript_context": transcript_context,
                        "directory_allowlist": directory_allowlist,
                        "avoid_ids": avoid_ids,
                        "candidates": [],
                        "selection": {},
                        "selected_id": "",
                        "errors": [],
                    },
                    config={"configurable": {"thread_id": thread_id}},
                )
                latency_ms = int(round((time.time() - t0) * 1000))
                last_agent_call = time.time()
                selected_id = str(result.get("selected_id", ""))
                reason = _extract_agent_reason(result)
            if trace:
                trace.write(
                    {
                        "type": "demo_agent_result",
                        "utterance": current.text,
                        "gesture_hint": hint,
                        "selected_id": selected_id,
                        "reason": reason,
                        "latency_ms": int(latency_ms),
                    }
                )
            if selected_id in ("", "NONE"):
                fallback_item = _pick_demo_fallback_item(
                    items=items,
                    hint=hint,
                    gesture_intensity=gesture_intensity,
                    avoid_ids=avoid_ids,
                )
                if fallback_item is None:
                    if status:
                        status.clear_backchannel_playback()
                        status.set_agent_decision(
                            choice_id="NONE",
                            choice_text="",
                            reason=reason,
                            latency_ms=latency_ms,
                            ts=time.time(),
                        )
                    continue
                selected_id = fallback_item.id
                reason = f"demo fallback: {reason or hint}"

            selected_item = next((item for item in items if item.id == selected_id), None)
            if selected_item is None:
                continue
            audio_path = find_audio_file(audio_dir, selected_item)
            if not audio_path:
                continue
            if status:
                status.set_agent_decision(
                    choice_id=selected_item.id,
                    choice_text=selected_item.text,
                    reason=reason,
                    latency_ms=latency_ms,
                    ts=time.time(),
                )
            played = _play_selected_backchannel(
                player=player,
                item=selected_item,
                audio_path=audio_path,
                reason=f"agent {hint}",
                interrupt=False,
                status=status,
                trace=trace,
                debug_agent=debug_agent,
            )
            if played:
                last_backchannel_play = time.time()
                last_backchannel_text = selected_item.text
                recent_ids.append(selected_item.id)
                recent_texts.append(selected_item.text)
                if len(recent_ids) > 12:
                    recent_ids = recent_ids[-12:]
                if len(recent_texts) > 12:
                    recent_texts = recent_texts[-12:]
        except Exception as exc:
            if status:
                status.set_agent_decision(
                    choice_id="NONE",
                    choice_text="",
                    reason=f"エラーのため今回は返しません: {exc}",
                    latency_ms=0,
                    ts=time.time(),
                )
            if debug_agent and not status:
                print(f"相槌の判断でエラーが起きました: {exc}")


def run_demo_session(
    *,
    script: DemoScript | None,
    script_path: Path | None,
    catalog_path: Path,
    audio_dir: Path,
    device_profile: DeviceProfile,
    port: str,
    baud: int | None,
    model: str,
    thread_id: str,
    status: StatusStore | None = None,
    trace: TraceWriter | None = None,
    debug_imu: bool = False,
    debug_agent: bool = False,
    debug_signal: bool = False,
    agent_interval_sec: float = 0.2,
    backchannel_cooldown_sec: float = 0.6,
    gesture_calibration: bool = True,
    gesture_weak_sec: float = 2.0,
    gesture_strong_sec: float = 2.0,
    gesture_start_delay_sec: float = 1.0,
    gesture_rest_sec: float = 1.0,
    human_signal_abs_threshold: float = 8.0,
    human_signal_max_age_s: float = 1.5,
    human_signal_min_consecutive: int = 3,
    imu_nod_axis: str = "gy",
    imu_shake_axis: str = "gz",
    interactive_cues: bool = True,
    cue_pause_sec: float = 0.2,
) -> None:
    def emit(message: str) -> None:
        if status:
            status.log(message)
            status.set_ui_guide(text=message)
        else:
            print(message)
            if trace:
                trace.log(message, source="print")

    items = load_catalog(catalog_path)
    experiment_id = script.script_id if script is not None else "sensor_only_en"
    if status:
        status.set_experiment(experiment_id=experiment_id, mode="demo")

    if trace:
        trace.write(
            {
                "type": "demo_session_start",
                "script_id": experiment_id,
                "script_path": "" if script_path is None else str(script_path),
                "device_id": device_profile.device_id,
                "input_kind": device_profile.input_kind,
                "catalog_path": str(catalog_path),
                "audio_dir": str(audio_dir),
                "model": model,
                "sensor_only": bool(script is None),
            }
        )

    graph: object | None = None
    if script is not None:
        client = OpenAI()
        checkpointer = InMemorySaver() if InMemorySaver else None
        graph = build_backchannel_graph(client, model, items).compile(checkpointer=checkpointer)

    resolved_port = resolve_serial_port(device_profile, override_port=port)
    resolved_baud = int(device_profile.baud if baud is None else baud)
    emit(f"IMUポート: {resolved_port}")
    if device_profile.input_kind == "gyro_xyz":
        probe = probe_serial_format(port=resolved_port, baud=resolved_baud, seconds=1.5)
        if probe.detected_format == "accel_xyz":
            raise ValueError(
                "このデバイスは加速度 3 軸の可能性が高いです。"
                "gyro_xyz では扱えないので、6 軸ファームウェアへ更新してください。"
            )
        if probe.detected_format == "six_axis":
            raise ValueError(
                "このデバイスは 6 軸を出しているようです。"
                "devices.json の input_kind を six_axis に直してください。"
            )
        emit(f"3 軸 probe: {probe.reason}")

    imu_buffer = ImuBuffer(max_seconds=600.0)
    stdin_toggle = script is None and sys.stdin.isatty()
    measurement_control = MeasurementControl(enabled=(script is not None or not stdin_toggle))
    threading.Thread(
        target=_imu_loop_device,
        kwargs={
            "profile": device_profile,
            "override_port": resolved_port,
            "override_baud": resolved_baud,
            "buffer": imu_buffer,
            "debug": debug_imu,
            "status": status,
        },
        daemon=True,
        name="demo-imu-reader",
    ).start()

    gesture_calib: GestureCalibration | None = None
    if gesture_calibration:
        gesture_calib = run_gesture_calibration(
            imu_buffer,
            weak_sec=gesture_weak_sec,
            strong_sec=gesture_strong_sec,
            start_delay_sec=gesture_start_delay_sec,
            rest_sec=gesture_rest_sec,
            log=emit,
        )
        if gesture_calib is not None:
            suggest = gesture_calib.axis_suggest
            nod_axis = suggest.get("nod_axis") or device_profile.nod_axis or imu_nod_axis
            shake_axis = suggest.get("shake_axis") or device_profile.shake_axis or imu_shake_axis
        else:
            suggest = {}
            nod_axis = device_profile.nod_axis or imu_nod_axis
            shake_axis = device_profile.shake_axis or imu_shake_axis
        if status and gesture_calib is not None:
            axis_map = (
                f"suggest nod={suggest.get('nod_axis','-')}, "
                f"shake={suggest.get('shake_axis','-')}"
            )
            status.set_gesture_calibration(
                summaries=gesture_calib.summaries(),
                axis_map=axis_map,
                ts=gesture_calib.finished_at,
            )
    else:
        nod_axis = device_profile.nod_axis or imu_nod_axis
        shake_axis = device_profile.shake_axis or imu_shake_axis

    signal_store = HumanSignalStore()
    signal_events: queue.Queue = queue.Queue()
    threading.Thread(
        target=_human_signal_loop,
        args=(imu_buffer, signal_store),
        kwargs={
            "abs_threshold": human_signal_abs_threshold,
            "max_age_s": human_signal_max_age_s,
            "min_consecutive_above": human_signal_min_consecutive,
            "nod_axis": nod_axis,
            "shake_axis": shake_axis,
            "event_queue": signal_events,
            "control": measurement_control,
            "debug": debug_signal and (status is None),
            "status": status,
        },
        daemon=True,
        name="demo-human-signal",
    ).start()

    player = AudioPlayer()
    manual_map = _manual_key_map(items, audio_dir)
    if script is None or script_path is None:
        speaker = SensorOnlySpeaker(status=status, trace=trace)
    else:
        speaker = DemoSpeaker(
            script=script,
            script_path=script_path,
            player=player,
            status=status,
            trace=trace,
            interactive_cues=interactive_cues,
            cue_pause_sec=cue_pause_sec,
            emit=emit,
        )
    threading.Thread(
        target=_demo_backchannel_loop,
        args=(signal_events,),
        kwargs={
            "control": measurement_control,
            "graph": graph,
            "items": items,
            "audio_dir": audio_dir,
            "imu_buffer": imu_buffer,
            "gesture_calib": gesture_calib,
            "nod_axis": nod_axis,
            "shake_axis": shake_axis,
            "speaker": speaker,
            "player": player,
            "thread_id": thread_id,
            "agent_interval_sec": agent_interval_sec,
            "backchannel_cooldown_sec": backchannel_cooldown_sec,
            "debug_agent": debug_agent,
            "status": status,
            "trace": trace,
        },
        daemon=True,
        name="demo-backchannel",
    ).start()

    speaker.start()
    if script is None:
        if stdin_toggle:
            emit("準備できました。Enter で測定 ON/OFF、q で終了です。")
            emit(_manual_key_guide(manual_map))
            threading.Thread(
                target=_sensor_only_control_loop,
                kwargs={
                    "control": measurement_control,
                    "speaker": speaker,
                    "player": player,
                    "signal_events": signal_events,
                    "manual_map": manual_map,
                    "emit": emit,
                    "trace": trace,
                    "status": status,
                    "debug_agent": debug_agent,
                },
                daemon=True,
                name="demo-sensor-toggle",
            ).start()
        else:
            measurement_control.set_enabled(True)
            emit("センサだけで反応します。頷くか、首を振ってください。終了は Ctrl+C です。")
    else:
        emit("デモを始めます。cue が出たら Enter で進めてください。")
    try:
        speaker.wait()
    except KeyboardInterrupt:
        speaker.stop()
        player.stop()
        emit("停止しました。")
        return
    emit("デモが終わりました。")
