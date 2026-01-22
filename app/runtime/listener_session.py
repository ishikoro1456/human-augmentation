from __future__ import annotations

import json
import queue
import re
import socket
import threading
import time
import uuid
import base64
import math
import struct
import traceback
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from openai import OpenAI

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # langgraph のバージョン差異に備える
    InMemorySaver = None

from app.agents.backchannel_graph import build_backchannel_graph
from app.audio.ffplay_stream import FfplayConfig, FfplayStreamPlayer
from app.audio.player import AudioPlayer
from app.core.catalog import load_catalog
from app.core.selector import find_audio_file
from app.imu.buffer import ImuBuffer, ImuSample
from app.imu.calibration import ImuCalibration, normalize_activity, run_calibration
from app.imu.gesture_calibration import GestureCalibration, run_gesture_calibration
from app.imu.reader import read_imu_lines
from app.imu.signal import detect_backchannel_signal
from app.imu.signal_store import HumanSignalStore
from app.net.jsonl import iter_jsonl_messages, send_jsonl
from app.runtime.status import StatusStore
from app.runtime.trace import TraceWriter
from app.transcript.live_buffer import LiveTranscriptBuffer


def _extract_agent_reason(result: Dict[str, object]) -> str:
    selection = result.get("selection", {})
    if not isinstance(selection, dict):
        return ""
    reason = selection.get("decision_reason", "") or selection.get("reason", "")
    return str(reason) if reason else ""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _rms_s16le(raw: bytes) -> int:
    if not raw:
        return 0
    count = int(len(raw) // 2)
    if count <= 0:
        return 0
    total = 0
    for (s,) in struct.iter_unpack("<h", raw[: count * 2]):
        total += int(s) * int(s)
    return int(math.sqrt(total / count))


def _extract_transcript_text(resp: object) -> str:
    if isinstance(resp, str):
        return resp.strip()
    # OpenAI python SDK returns a typed object for json response_format
    text = getattr(resp, "text", None)
    if text is not None:
        return str(text).strip()
    if isinstance(resp, dict):
        t = resp.get("text", "")
        return str(t).strip() if t else ""
    return str(resp).strip()


def _stt_response_format_for_model(model: str) -> str:
    m = str(model or "").lower()
    if "gpt-4o" in m and "transcribe" in m:
        return "json"
    return "text"


def imu_loop(
    port: str,
    baud: int,
    buffer: ImuBuffer,
    debug: bool = False,
    status: StatusStore | None = None,
) -> None:
    last_log = 0.0
    last_status = 0.0
    log_fn = status.log if status else None
    for data in read_imu_lines(port, baud, on_log=log_fn):
        now = time.time()
        ax, ay, az, gx, gy, gz = data
        buffer.add(ImuSample(ts=now, ax=ax, ay=ay, az=az, gx=gx, gy=gy, gz=gz))
        if status:
            if now - last_status > 0.2:
                status.set_imu(motion_text=buffer.format_status_line(now=now), event="raw", ts=now)
                last_status = now
        if debug:
            if now - last_log > 1.0:
                message = f"IMU受信中: {buffer.format_status_line(now=now)}"
                if status:
                    status.log(message)
                else:
                    print(message)
                last_log = now


def human_signal_loop(
    buffer: ImuBuffer,
    store: HumanSignalStore,
    *,
    calibration: ImuCalibration | None,
    gyro_sigma: float,
    abs_threshold: float,
    max_age_s: float,
    min_consecutive_above: int,
    nod_axis: str,
    shake_axis: str,
    tick_sec: float = 0.1,
    event_queue: "queue.Queue[Dict[str, object]] | None" = None,
    debug: bool = False,
    status: StatusStore | None = None,
    trace: TraceWriter | None = None,
    trace_interval_sec: float = 1.0,
) -> None:
    last_eligible: bool | None = None
    last_present: bool | None = None
    episode_fired = False
    last_trace_ts = 0.0
    last_trace_present: bool | None = None
    last_trace_hint: str | None = None
    while True:
        now = time.time()
        imu_bundle = buffer.build_bundle(
            now=now,
            raw_window_sec=2.0,
            raw_max_points=24,
            stats_windows_sec=[1.0],
        )
        signal = detect_backchannel_signal(
            imu_bundle,
            calibration=calibration,
            gyro_sigma=gyro_sigma,
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

        recent_counts_30s: Dict[str, int] = {}
        recent_rate_per_min_30s: Dict[str, float] = {}
        try:
            recent_counts_30s = store.count_by_gesture(max_age_s=30.0, now=now)
            recent_rate_per_min_30s = {k: round(float(v) * 2.0, 2) for k, v in recent_counts_30s.items()}
        except Exception:
            recent_counts_30s = {}
            recent_rate_per_min_30s = {}

        if trace and (
            (now - last_trace_ts) >= float(max(0.2, trace_interval_sec))
            or present != last_trace_present
            or (str(hint) if isinstance(hint, str) else "") != (last_trace_hint or "")
        ):
            imu_latest = imu_bundle.get("latest", {})
            imu_activity = imu_bundle.get("activity_1s", {})
            trace.write(
                {
                    "type": "imu_signal_tick",
                    "present": bool(present),
                    "eligible": bool(eligible),
                    "gesture_hint": str(hint) if isinstance(hint, str) else "",
                    "reason": str(signal.get("reason", "") or ""),
                    "last_sample_age_s": imu_bundle.get("last_sample_age_s"),
                    "sample_rate_hz": imu_bundle.get("sample_rate_hz"),
                    "latest": dict(imu_latest) if isinstance(imu_latest, dict) else {},
                    "activity_1s": dict(imu_activity) if isinstance(imu_activity, dict) else {},
                    "gyro_mag_max_1s": signal.get("gyro_mag_max_1s"),
                    "threshold": signal.get("threshold"),
                    "threshold_base": signal.get("threshold_base"),
                    "run_max": signal.get("run_max"),
                    "min_consecutive_above": signal.get("min_consecutive_above"),
                    "dominant_axis": signal.get("dominant_axis"),
                    "axis_map": signal.get("axis_map"),
                    "motion_features": signal.get("motion_features"),
                    "gesture_counts_30s": recent_counts_30s,
                    "gesture_rate_per_min_30s": recent_rate_per_min_30s,
                },
                ts=now,
            )
            last_trace_ts = float(now)
            last_trace_present = bool(present)
            last_trace_hint = str(hint) if isinstance(hint, str) else ""

        if last_present is None:
            last_present = present
        if not present:
            episode_fired = False
        elif last_present is False and present:
            episode_fired = False

        if event_queue is not None:
            if eligible and not episode_fired:
                event_queue.put({"ts": now, "signal": dict(signal)})
                episode_fired = True
        if status:
            status.set_human_signal(text=str(signal.get("reason", "")))
        if debug and not status:
            if last_eligible is None or eligible != last_eligible:
                print(f"IMUサイン: {signal.get('reason', '')}")
            last_eligible = eligible
        else:
            last_eligible = eligible
        last_present = present
        time.sleep(tick_sec)


@dataclass
class TalkerConnection:
    sock: socket.socket | None = None
    addr: str = ""
    lock: object = field(default_factory=threading.Lock)


@dataclass
class TalkerGuideState:
    text: str = ""
    updated_ts: float = 0.0
    lock: object = field(default_factory=threading.Lock)

    def set(self, text: str, *, ts: float | None = None) -> None:
        now = time.time() if ts is None else float(ts)
        with self.lock:
            self.text = str(text)
            self.updated_ts = float(now)

    def snapshot(self) -> str:
        with self.lock:
            return str(self.text)


def transcript_server_loop(
    *,
    host: str,
    port: int,
    experiment_id: str,
    event_queue: "queue.Queue[Dict[str, object]]",
    conn_state: TalkerConnection,
    log: callable,
    status: StatusStore | None = None,
    guide_state: TalkerGuideState | None = None,
    trace: TraceWriter | None = None,
) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, int(port)))
    server.listen(1)
    log(f"話し手接続: {host}:{int(port)} で待ち受けます。")
    if trace:
        trace.write({"type": "talker_listen", "host": host, "port": int(port)})
    while True:
        conn, addr = server.accept()
        addr_str = f"{addr[0]}:{addr[1]}"
        try:
            with conn_state.lock:
                conn_state.sock = conn
                conn_state.addr = addr_str
            log(f"話し手接続: 接続 {addr_str}")
            if status:
                status.set_talker_connection(connected=True, addr=addr_str, ts=time.time())
            if trace:
                trace.write({"type": "talker_connected", "addr": addr_str})
            try:
                send_jsonl(conn, {"type": "session", "experiment_id": str(experiment_id), "ts_ms": int(_now_ms())})
                if trace:
                    trace.write({"type": "session_sent", "addr": addr_str})
                if guide_state is not None:
                    guide = str(guide_state.snapshot() or "").strip()
                    if guide:
                        send_jsonl(conn, {"type": "guide", "text": guide, "ts_ms": int(_now_ms())})
            except Exception as exc:
                log(f"話し手接続: session を送れませんでした: {exc}")
                if trace:
                    trace.write({"type": "session_send_error", "error": str(exc)})
            for msg in iter_jsonl_messages(conn):
                msg = dict(msg)
                msg["_received_ts"] = round(time.time(), 3)
                event_queue.put(msg)
        except Exception as exc:
            log(f"話し手接続: エラー {exc}")
            if trace:
                trace.write({"type": "talker_error", "error": str(exc)})
        finally:
            with conn_state.lock:
                if conn_state.sock is conn:
                    conn_state.sock = None
                    conn_state.addr = ""
            try:
                conn.close()
            except Exception:
                pass
            log("話し手接続: 切断")
            if status:
                status.set_talker_connection(connected=False, addr="", ts=time.time())
            if trace:
                trace.write({"type": "talker_disconnected"})
            try:
                event_queue.put({"type": "talker_disconnected", "addr": addr_str, "ts_ms": int(_now_ms())})
            except Exception:
                pass


@dataclass
class SpeakerAudioState:
    speaking: bool = False
    silence_ms: int = 0
    last_update_ts: float = 0.0


def run_listener_session(
    *,
    catalog_path: Path,
    audio_dir: Path,
    listen_host: str,
    listen_port: int,
    port: str,
    baud: int,
    model: str,
    thread_id: str,
    experiment_id: str,
    status: StatusStore | None = None,
    trace: TraceWriter | None = None,
    debug_imu: bool = False,
    debug_agent: bool = False,
    debug_signal: bool = False,
    agent_interval_sec: float = 0.0,
    backchannel_cooldown_sec: float = 0.0,
    calibration_still_sec: float = 0.0,
    calibration_active_sec: float = 10.0,
    calibration_start_delay_sec: float = 3.0,
    calibration_between_sec: float = 0.0,
    calibration_wait_for_imu_sec: float = 15.0,
    human_signal_gyro_sigma: float = 3.0,
    human_signal_abs_threshold: float = 8.0,
    human_signal_max_age_s: float = 1.5,
    human_signal_min_consecutive: int = 3,
    human_signal_hold_sec: float = 3.0,
    imu_nod_axis: str = "gy",
    imu_shake_axis: str = "gz",
    gesture_calibration: bool = True,
    gesture_weak_sec: float = 2.0,
    gesture_strong_sec: float = 2.0,
    gesture_start_delay_sec: float = 2.0,
    gesture_rest_sec: float = 2.0,
    auto_imu_axis_map: bool = True,
    boundary_silence_ms: int = 150,
    context_max_lines: int = 10,
    early_call_delay_sec: float = 0.2,
    speaker_playback: bool = True,
    speaker_playback_bin: str = "ffplay",
    stt_model: str = "gpt-4o-transcribe",
    stt_language: str = "ja",
    stt_prompt: str = "",
    stt_segments_dir: Path = Path("data/stt_segments_listener"),
    vad_frame_ms: int = 20,
    vad_pre_roll_ms: int = 200,
    vad_silence_end_ms: int = 500,
    vad_min_speech_ms: int = 300,
    vad_start_voice_frames: int = 2,
    vad_min_voice_ms: int = 80,
    vad_calib_sec: float = 1.0,
    vad_threshold_rms: int = 0,
    vad_threshold_mult: float = 3.0,
    vad_max_segment_ms: int = 20000,
    send_backchannel_to_talker: bool = True,
    local_backchannel_play: bool = False,
    mode: str = "llm",
    human_choice_count: int = 9,
    human_choice_ids: str = "",
) -> None:
    def emit(message: str) -> None:
        if status:
            status.set_ui_guide(text=message)
            status.log(message)
        else:
            print(message)
            if trace:
                trace.log(message, source="print")

    def log_only(message: str) -> None:
        if status:
            status.log(message)
        else:
            print(message)
            if trace:
                trace.log(message, source="print")

    started_at_ts = status.snapshot().started_at if status else time.time()

    items = load_catalog(catalog_path)

    if status:
        status.set_experiment(experiment_id=str(experiment_id), mode=str(mode))

    if trace:
        trace.write(
            {
                "type": "listener_session_start",
                "thread_id": thread_id,
                "experiment_id": str(experiment_id),
                "model": model,
                "mode": str(mode),
                "listen": {"host": listen_host, "port": int(listen_port)},
                "port": port,
                "baud": int(baud),
                "human_signal_hold_sec": float(human_signal_hold_sec),
                "human_signal_gyro_sigma": float(human_signal_gyro_sigma),
                "human_signal_abs_threshold": float(human_signal_abs_threshold),
                "human_signal_max_age_s": float(human_signal_max_age_s),
                "human_signal_min_consecutive": int(human_signal_min_consecutive),
                "imu_axes": {"nod_axis": imu_nod_axis, "shake_axis": imu_shake_axis},
                "gesture_calibration": bool(gesture_calibration),
                "stt": {
                    "model": str(stt_model),
                    "language": str(stt_language),
                    "prompt_enabled": bool(stt_prompt),
                    "segments_dir": str(stt_segments_dir),
                },
                "vad": {
                    "frame_ms": int(vad_frame_ms),
                    "pre_roll_ms": int(vad_pre_roll_ms),
                    "silence_end_ms": int(vad_silence_end_ms),
                    "min_speech_ms": int(vad_min_speech_ms),
                    "start_voice_frames": int(vad_start_voice_frames),
                    "min_voice_ms": int(vad_min_voice_ms),
                    "calib_sec": float(vad_calib_sec),
                    "threshold_rms": int(vad_threshold_rms),
                    "threshold_mult": float(vad_threshold_mult),
                    "max_segment_ms": int(vad_max_segment_ms),
                },
            }
        )

    client = OpenAI()
    checkpointer = InMemorySaver() if InMemorySaver else None
    graph = build_backchannel_graph(client, model, items).compile(checkpointer=checkpointer)

    imu_buffer = ImuBuffer(max_seconds=600.0)
    threading.Thread(target=imu_loop, args=(port, baud, imu_buffer, debug_imu, status), daemon=True).start()

    # 文字起こしの受信は、IMUの計測より先に待ち受けを開始しておく（talker が先に起動しても接続できるように）
    talker_conn = TalkerConnection()
    talker_guide_state = TalkerGuideState()
    talker_guide_state.set("聞き手が話し手の接続を待っています。まだ話さずに待ってください。")
    net_events: queue.Queue = queue.Queue()
    threading.Thread(
        target=transcript_server_loop,
        kwargs={
            "host": listen_host,
            "port": int(listen_port),
            "experiment_id": str(experiment_id),
            "event_queue": net_events,
            "conn_state": talker_conn,
            "log": log_only,
            "status": status,
            "guide_state": talker_guide_state,
            "trace": trace,
        },
        daemon=True,
    ).start()

    last_talker_guide_text = ""
    last_talker_guide_ts = 0.0

    def _send_to_talker(payload: Dict[str, object]) -> None:
        if not send_backchannel_to_talker:
            return
        with talker_conn.lock:
            sock = talker_conn.sock
        if sock is None:
            return
        try:
            send_jsonl(sock, payload)
        except Exception:
            return

    def _broadcast_guide_to_talker(message: str) -> None:
        nonlocal last_talker_guide_text, last_talker_guide_ts
        now_ts = time.time()
        msg = str(message or "").strip()
        if not msg:
            return
        if msg == last_talker_guide_text and (now_ts - last_talker_guide_ts) < 0.3:
            return
        talker_guide_state.set(msg, ts=now_ts)
        _send_to_talker({"type": "guide", "text": msg, "ts_ms": int(_now_ms())})
        last_talker_guide_text = msg
        last_talker_guide_ts = float(now_ts)

    emit("話し手の接続を待っています。話し手アプリを起動してください。")
    _broadcast_guide_to_talker("聞き手が話し手の接続を待っています。まだ話さずに待ってください。")

    # 話し手音声（ネットワーク）を受け取り、すぐ再生しつつ、無音区切りで文字起こしする
    transcript_events: queue.Queue = queue.Queue()
    speaker_state = SpeakerAudioState(speaking=False, silence_ms=0, last_update_ts=time.time())
    speaker_state_seen = False

    stt_queue: queue.Queue = queue.Queue()
    stt_segments_dir.mkdir(parents=True, exist_ok=True)

    def _write_wav(path: Path, *, pcm_s16le: bytes, sample_rate: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm_s16le)

    def _stt_worker() -> None:
        while True:
            job = stt_queue.get()
            if not isinstance(job, dict):
                continue
            seg_id = job.get("segment_id")
            pcm = job.get("pcm_s16le")
            start_ms = job.get("start_ts_ms")
            end_ms = job.get("end_ts_ms")
            sample_rate = job.get("sample_rate")
            enqueued_ts_ms = job.get("enqueued_ts_ms")
            end_reason = job.get("end_reason")
            if not isinstance(seg_id, int):
                continue
            if not isinstance(pcm, (bytes, bytearray)):
                continue
            if not isinstance(start_ms, int) or not isinstance(end_ms, int):
                continue
            if not isinstance(sample_rate, int):
                continue

            wav_path = stt_segments_dir / f"seg_{seg_id:04d}_{start_ms}_{end_ms}.wav"
            dequeued_ts_ms = int(_now_ms())
            queue_wait_ms: int | None = None
            if isinstance(enqueued_ts_ms, int):
                queue_wait_ms = int(dequeued_ts_ms - int(enqueued_ts_ms))
            pcm_bytes = int(len(pcm))
            duration_s = round((float(pcm_bytes) / float(sample_rate * 2)) if int(sample_rate) > 0 else 0.0, 3)
            if trace:
                trace.write(
                    {
                        "type": "stt_job_start",
                        "segment_id": int(seg_id),
                        "wav_path": str(wav_path),
                        "pcm_bytes": int(pcm_bytes),
                        "duration_s": float(duration_s),
                        "queue_wait_ms": None if queue_wait_ms is None else int(queue_wait_ms),
                        "enqueued_ts_ms": int(enqueued_ts_ms) if isinstance(enqueued_ts_ms, int) else None,
                        "dequeued_ts_ms": int(dequeued_ts_ms),
                        "end_reason": str(end_reason) if isinstance(end_reason, str) else None,
                        "model": str(stt_model),
                    }
                )
            try:
                stage = "write_wav"
                _write_wav(wav_path, pcm_s16le=bytes(pcm), sample_rate=sample_rate)
                if trace:
                    trace.write(
                        {
                            "type": "stt_wav_written",
                            "segment_id": int(seg_id),
                            "wav_path": str(wav_path),
                            "pcm_bytes": int(pcm_bytes),
                            "duration_s": float(duration_s),
                        }
                    )
                stage = "request"
                req_started = time.time()
                resp_format = _stt_response_format_for_model(stt_model)
                if trace:
                    trace.write(
                        {
                            "type": "stt_request_start",
                            "segment_id": int(seg_id),
                            "model": str(stt_model),
                            "response_format": str(resp_format),
                            "language": str(stt_language) if stt_language else "",
                            "prompt_enabled": bool(stt_prompt),
                        }
                    )
                with wav_path.open("rb") as f:
                    params: Dict[str, object] = {"model": stt_model, "file": f, "response_format": resp_format}
                    if stt_language:
                        params["language"] = stt_language
                    if stt_prompt:
                        params["prompt"] = stt_prompt
                    resp = client.audio.transcriptions.create(**params)
                text = _extract_transcript_text(resp)
                req_ms = int(round((time.time() - req_started) * 1000.0))
                if trace:
                    trace.write(
                        {
                            "type": "stt_request_end",
                            "segment_id": int(seg_id),
                            "latency_ms": int(req_ms),
                            "text_len": int(len(text)),
                        }
                    )
            except Exception as exc:
                emit(f"文字起こしに失敗しました: {exc}")
                if trace:
                    trace.write(
                        {
                            "type": "stt_error",
                            "segment_id": int(seg_id),
                            "stage": str(stage) if "stage" in locals() else "",
                            "wav_path": str(wav_path),
                            "error": str(exc),
                        }
                    )
                continue

            if text:
                transcript_events.put(
                    {
                        "type": "segment_final",
                        "segment_id": int(seg_id),
                        "text": text,
                        "start_ts_ms": int(start_ms),
                        "end_ts_ms": int(end_ms),
                        "ts_ms": int(end_ms),
                    }
                )
                if trace:
                    trace.write(
                        {
                            "type": "stt_segment",
                            "segment_id": int(seg_id),
                            "text": text,
                            "text_len": int(len(text)),
                            "wav_path": str(wav_path),
                            "duration_s": float(duration_s),
                        }
                    )

    threading.Thread(target=_stt_worker, daemon=True).start()

    speaker_sample_rate = 16000
    speaker_frame_ms = int(max(5, vad_frame_ms))
    speaker_player = FfplayStreamPlayer(
        FfplayConfig(ffplay_bin=str(speaker_playback_bin), sample_rate=int(speaker_sample_rate), channels=1)
    )
    speaker_player_started = False
    speaker_player_warned = False

    # 無音区切りの状態
    noise_rms = 200.0
    calib_until = time.time() + max(0.0, float(vad_calib_sec))
    speaking = False
    start_voice_streak = 0
    silence_frames = 0
    pre_roll: list[bytes] = []
    segment_frames: list[bytes] = []
    segment_start_ms = 0
    seg_id = 0
    seg_voice_frames = 0
    seg_rms_sum = 0
    seg_rms_max = 0

    def _recompute_vad_derived() -> tuple[int, int, int, int, int]:
        frame_ms = int(max(5, speaker_frame_ms))
        frame_samples = int(speaker_sample_rate * frame_ms / 1000)
        frame_bytes = frame_samples * 2
        pre_roll_frames = max(0, int(int(vad_pre_roll_ms) / frame_ms))
        silence_end_frames = max(1, int(int(vad_silence_end_ms) / frame_ms))
        min_speech_frames = max(1, int(int(vad_min_speech_ms) / frame_ms))
        max_segment_frames = 0
        if int(vad_max_segment_ms) > 0:
            max_segment_frames = max(1, int(int(vad_max_segment_ms) / frame_ms))
        return frame_bytes, pre_roll_frames, silence_end_frames, min_speech_frames, max_segment_frames

    def _audio_loop() -> None:
        nonlocal speaker_state_seen, speaker_sample_rate, speaker_frame_ms
        nonlocal speaker_player, speaker_player_started, speaker_player_warned
        nonlocal noise_rms, calib_until, speaking, start_voice_streak, silence_frames, pre_roll, segment_frames, segment_start_ms, seg_id
        nonlocal seg_voice_frames, seg_rms_sum, seg_rms_max

        frame_bytes, pre_roll_frames, silence_end_frames, min_speech_frames, max_segment_frames = _recompute_vad_derived()
        if trace:
            trace.write(
                {
                    "type": "speaker_audio_loop_start",
                    "speaker_playback": bool(speaker_playback),
                    "sample_rate": int(speaker_sample_rate),
                    "frame_ms": int(speaker_frame_ms),
                    "frame_bytes": int(frame_bytes),
                }
            )
        recv_stats_started = time.time()
        recv_frames = 0
        recv_bytes = 0
        recv_rms_sum = 0
        recv_rms_max = 0
        last_rms_mean_2s: float | None = None
        last_rms_max_2s: int | None = None
        last_status_update = 0.0

        def _flush_segment(*, end_ts_ms: int, reason: str) -> None:
            nonlocal speaking, start_voice_streak, silence_frames, pre_roll, segment_frames, segment_start_ms, seg_id
            nonlocal seg_voice_frames, seg_rms_sum, seg_rms_max

            if not segment_frames:
                speaking = False
                start_voice_streak = 0
                silence_frames = 0
                pre_roll = []
                seg_voice_frames = 0
                seg_rms_sum = 0
                seg_rms_max = 0
                return

            frame_ms = int(max(5, speaker_frame_ms))
            seg_frames = int(len(segment_frames))
            seg_ms_est = int(seg_frames * frame_ms)
            voice_ms = int(int(seg_voice_frames) * frame_ms)
            should_send = (seg_frames >= int(min_speech_frames)) and (voice_ms >= int(max(0, vad_min_voice_ms)))
            pcm_bytes_est = int(seg_frames * int(frame_bytes))
            rms_mean = round((float(seg_rms_sum) / float(seg_frames)) if seg_frames > 0 else 0.0, 1)
            thr_rms = (
                int(vad_threshold_rms)
                if int(vad_threshold_rms) > 0
                else int(max(300.0, float(noise_rms) * float(vad_threshold_mult)))
            )
            noise_rms_now = round(float(noise_rms), 1)

            if should_send:
                seg_id += 1
                enqueued_ts_ms = int(_now_ms())
                pcm = b"".join(segment_frames)
                stt_queue.put(
                    {
                        "segment_id": int(seg_id),
                        "pcm_s16le": pcm,
                        "start_ts_ms": int(segment_start_ms),
                        "end_ts_ms": int(end_ts_ms),
                        "sample_rate": int(speaker_sample_rate),
                        "enqueued_ts_ms": int(enqueued_ts_ms),
                        "end_reason": str(reason),
                    }
                )
                stt_queue_size = 0
                try:
                    stt_queue_size = int(stt_queue.qsize())
                except Exception:
                    stt_queue_size = 0
                if trace:
                    trace.write(
                        {
                            "type": "vad_segment",
                            "segment_id": int(seg_id),
                            "start_ts_ms": int(segment_start_ms),
                            "end_ts_ms": int(end_ts_ms),
                            "frames": int(seg_frames),
                            "voice_frames": int(seg_voice_frames),
                            "voice_ms": int(voice_ms),
                            "segment_ms_est": int(seg_ms_est),
                            "rms_mean": float(rms_mean),
                            "rms_max": int(seg_rms_max),
                            "pcm_bytes": int(len(pcm)),
                            "sample_rate": int(speaker_sample_rate),
                            "frame_ms": int(frame_ms),
                            "thr_rms": int(thr_rms),
                            "noise_rms": float(noise_rms_now),
                            "stt_queue_size": int(stt_queue_size),
                            "enqueued_ts_ms": int(enqueued_ts_ms),
                            "end_reason": str(reason),
                        }
                    )
            else:
                if trace:
                    trace.write(
                        {
                            "type": "vad_segment_drop",
                            "start_ts_ms": int(segment_start_ms),
                            "end_ts_ms": int(end_ts_ms),
                            "frames": int(seg_frames),
                            "voice_frames": int(seg_voice_frames),
                            "voice_ms": int(voice_ms),
                            "segment_ms_est": int(seg_ms_est),
                            "rms_mean": float(rms_mean),
                            "rms_max": int(seg_rms_max),
                            "pcm_bytes_est": int(pcm_bytes_est),
                            "sample_rate": int(speaker_sample_rate),
                            "frame_ms": int(frame_ms),
                            "thr_rms": int(thr_rms),
                            "noise_rms": float(noise_rms_now),
                            "reason": "too_short_or_quiet",
                            "end_reason": str(reason),
                            "min_speech_frames": int(min_speech_frames),
                            "vad_min_voice_ms": int(max(0, vad_min_voice_ms)),
                        }
                    )

            speaking = False
            start_voice_streak = 0
            silence_frames = 0
            segment_frames = []
            pre_roll = []
            seg_voice_frames = 0
            seg_rms_sum = 0
            seg_rms_max = 0

        vad_tick_started = time.time()
        vad_tick_frames = 0
        vad_tick_voice_frames = 0
        vad_tick_rms_sum = 0
        vad_tick_rms_max = 0
        vad_tick_last_rms = 0
        vad_tick_last_thr = 0

        def _trace_vad_tick(now_ts: float, *, rms: int, thr: int, is_voice: bool, chunk_ts_ms: int) -> None:
            nonlocal vad_tick_started, vad_tick_frames, vad_tick_voice_frames, vad_tick_rms_sum, vad_tick_rms_max
            nonlocal vad_tick_last_rms, vad_tick_last_thr

            vad_tick_frames += 1
            vad_tick_rms_sum += int(rms)
            vad_tick_rms_max = max(int(vad_tick_rms_max), int(rms))
            if is_voice:
                vad_tick_voice_frames += 1
            vad_tick_last_rms = int(rms)
            vad_tick_last_thr = int(thr)
            if (not trace) or (now_ts - float(vad_tick_started)) < 1.0:
                return
            mean = (float(vad_tick_rms_sum) / float(vad_tick_frames)) if vad_tick_frames > 0 else 0.0
            trace.write(
                {
                    "type": "vad_tick",
                    "ts_ms": int(chunk_ts_ms),
                    "frames": int(vad_tick_frames),
                    "voice_frames": int(vad_tick_voice_frames),
                    "rms_mean": round(float(mean), 1),
                    "rms_max": int(vad_tick_rms_max),
                    "rms_last": int(vad_tick_last_rms),
                    "thr_last": int(vad_tick_last_thr),
                    "noise_rms": round(float(noise_rms), 1),
                    "speaking": bool(speaking),
                    "silence_frames": int(silence_frames),
                    "start_voice_streak": int(start_voice_streak),
                    "segment_frames_len": int(len(segment_frames)),
                    "pre_roll_len": int(len(pre_roll)),
                    "frame_ms": int(max(5, speaker_frame_ms)),
                    "sample_rate": int(speaker_sample_rate),
                }
            )
            vad_tick_started = float(now_ts)
            vad_tick_frames = 0
            vad_tick_voice_frames = 0
            vad_tick_rms_sum = 0
            vad_tick_rms_max = 0

        last_err_ts = 0.0
        last_err = ""
        last_msg_type = ""
        last_raw_bytes = 0

        while True:
            try:
                msg = net_events.get()
                if not isinstance(msg, dict):
                    continue
                msg_type = str(msg.get("type", ""))
                last_msg_type = msg_type

                if msg_type == "talker_disconnected":
                    end_ts_obj = msg.get("ts_ms")
                    end_ts_ms = int(end_ts_obj) if isinstance(end_ts_obj, (int, float)) else int(_now_ms())
                    if speaking and segment_frames:
                        _flush_segment(end_ts_ms=int(end_ts_ms), reason="talker_disconnected")
                    speaking = False
                    start_voice_streak = 0
                    silence_frames = 0
                    pre_roll = []
                    segment_frames = []
                    seg_voice_frames = 0
                    seg_rms_sum = 0
                    seg_rms_max = 0
                    speaker_state.speaking = False
                    speaker_state.silence_ms = 0
                    speaker_state.last_update_ts = time.time()
                    speaker_state_seen = True
                    continue

                if msg_type == "hello":
                    end_ts_obj = msg.get("ts_ms")
                    end_ts_ms = int(end_ts_obj) if isinstance(end_ts_obj, (int, float)) else int(_now_ms())
                    if speaking and segment_frames:
                        _flush_segment(end_ts_ms=int(end_ts_ms), reason="hello")
                    speaking = False
                    start_voice_streak = 0
                    silence_frames = 0
                    pre_roll = []
                    segment_frames = []
                    seg_voice_frames = 0
                    seg_rms_sum = 0
                    seg_rms_max = 0
                    audio = msg.get("audio", {})
                    if isinstance(audio, dict):
                        sr = audio.get("sample_rate")
                        fm = audio.get("frame_ms")
                        if isinstance(sr, int) and sr > 0:
                            speaker_sample_rate = int(sr)
                        if isinstance(fm, int) and fm >= 5:
                            speaker_frame_ms = int(fm)
                        speaker_player.close()
                        speaker_player = FfplayStreamPlayer(
                            FfplayConfig(ffplay_bin=str(speaker_playback_bin), sample_rate=int(speaker_sample_rate), channels=1)
                        )
                        speaker_player_started = False
                        speaker_player_warned = False
                        frame_bytes, pre_roll_frames, silence_end_frames, min_speech_frames, max_segment_frames = _recompute_vad_derived()
                    continue

                if msg_type != "audio_chunk":
                    continue

                data_b64 = msg.get("data_b64", "")
                if not isinstance(data_b64, str) or not data_b64:
                    continue
                try:
                    raw = base64.b64decode(data_b64)
                except Exception:
                    continue
                last_raw_bytes = int(len(raw))
                chunk_ts_obj = msg.get("ts_ms")
                chunk_ts_ms = int(chunk_ts_obj) if isinstance(chunk_ts_obj, (int, float)) else int(_now_ms())
                recv_bytes += int(len(raw))

                if speaker_playback:
                    if not speaker_player.is_running():
                        speaker_player_started = bool(speaker_player.start())
                        if (not speaker_player_started) and (not speaker_player_warned):
                            tail = speaker_player.stderr_tail()
                            detail = tail[-1] if tail else ""
                            warn_msg = "ffplay で話し手音声を再生できないので、省略します。"
                            if detail:
                                warn_msg += f" (ffplay: {detail})"
                            emit(warn_msg)
                            speaker_player_warned = True
                    else:
                        speaker_player_started = True
                    if speaker_player_started:
                        ok = bool(speaker_player.write(raw))
                        if not ok:
                            speaker_player_started = False
                            if not speaker_player_warned:
                                tail = speaker_player.stderr_tail()
                                detail = tail[-1] if tail else ""
                                warn_msg = "ffplay が止まったので、話し手音声の再生を再起動します。"
                                if detail:
                                    warn_msg += f" (ffplay: {detail})"
                                emit(warn_msg)
                                speaker_player_warned = True

                # フレーム処理（複数フレームがまとまって届いても扱う / サイズが合わなければ丸ごと1フレーム扱い）
                frames: list[bytes] = []
                if int(frame_bytes) > 0 and len(raw) >= int(frame_bytes) and (len(raw) % int(frame_bytes) == 0):
                    i = 0
                    while i + int(frame_bytes) <= len(raw):
                        frames.append(raw[i : i + int(frame_bytes)])
                        i += int(frame_bytes)
                else:
                    # サイズが合わない場合は、今届いたサイズからフレーム長を推定してみる
                    if len(raw) >= 2 and (len(raw) % 2 == 0) and int(speaker_sample_rate) > 0:
                        est_samples = int(len(raw) // 2)
                        est_ms = int(round(float(est_samples) * 1000.0 / float(speaker_sample_rate)))
                        if est_ms >= 5 and est_ms != int(speaker_frame_ms):
                            speaker_frame_ms = int(est_ms)
                            frame_bytes, pre_roll_frames, silence_end_frames, min_speech_frames, max_segment_frames = _recompute_vad_derived()
                            if trace:
                                trace.write(
                                    {
                                        "type": "speaker_frame_resync",
                                        "raw_bytes": int(len(raw)),
                                        "sample_rate": int(speaker_sample_rate),
                                        "frame_ms": int(speaker_frame_ms),
                                        "frame_bytes": int(frame_bytes),
                                    }
                                )
                    frames = [raw]

                for idx, frame in enumerate(frames):
                    rms_msg = msg.get("rms")
                    rms = int(rms_msg) if (idx == 0 and isinstance(rms_msg, (int, float))) else _rms_s16le(frame)
                    recv_frames += 1
                    recv_rms_sum += int(rms)
                    recv_rms_max = max(int(recv_rms_max), int(rms))

                    # しきい値の推定（無音のときだけ更新）
                    thr = (
                        int(vad_threshold_rms)
                        if int(vad_threshold_rms) > 0
                        else int(max(300.0, noise_rms * float(vad_threshold_mult)))
                    )
                    is_voice = rms >= thr
                    if time.time() < calib_until:
                        if not is_voice:
                            noise_rms = (noise_rms * 0.95) + (rms * 0.05)
                    elif (not speaking) and (not is_voice):
                        noise_rms = (noise_rms * 0.995) + (rms * 0.005)

                    pre_roll.append(frame)
                    if len(pre_roll) > pre_roll_frames:
                        pre_roll = pre_roll[-pre_roll_frames:]

                    now_ts = time.time()
                    if not speaking:
                        if is_voice:
                            start_voice_streak += 1
                        else:
                            start_voice_streak = 0

                        if start_voice_streak >= int(max(1, vad_start_voice_frames)):
                            speaking = True
                            initial_voice_frames = int(start_voice_streak)
                            start_voice_streak = 0
                            silence_frames = 0
                            seg_voice_frames = int(initial_voice_frames)
                            seg_rms_sum = int(rms)
                            seg_rms_max = int(rms)
                            segment_frames = list(pre_roll)
                            segment_start_ms = int(chunk_ts_ms)
                            speaker_state.speaking = True
                            speaker_state.silence_ms = 0
                            speaker_state.last_update_ts = now_ts
                            speaker_state_seen = True
                        else:
                            speaker_state.speaking = False
                            speaker_state.silence_ms = int(max(0, speaker_state.silence_ms + int(speaker_frame_ms)))
                            speaker_state.last_update_ts = now_ts
                            speaker_state_seen = True
                        _trace_vad_tick(
                            now_ts,
                            rms=int(rms),
                            thr=int(thr),
                            is_voice=bool(is_voice),
                            chunk_ts_ms=int(chunk_ts_ms),
                        )
                        continue

                    # speaking 中
                    segment_frames.append(frame)
                    if is_voice:
                        seg_voice_frames += 1
                        seg_rms_sum += int(rms)
                        seg_rms_max = max(int(seg_rms_max), int(rms))
                        silence_frames = 0
                        speaker_state.speaking = True
                        speaker_state.silence_ms = 0
                        speaker_state.last_update_ts = now_ts
                        speaker_state_seen = True
                    else:
                        seg_rms_sum += int(rms)
                        seg_rms_max = max(int(seg_rms_max), int(rms))
                        silence_frames += 1
                        speaker_state.speaking = False
                        speaker_state.silence_ms = int(silence_frames * int(speaker_frame_ms))
                        speaker_state.last_update_ts = now_ts
                        speaker_state_seen = True

                        if silence_frames >= silence_end_frames:
                            _trace_vad_tick(
                                now_ts,
                                rms=int(rms),
                                thr=int(thr),
                                is_voice=bool(is_voice),
                                chunk_ts_ms=int(chunk_ts_ms),
                            )
                            _flush_segment(end_ts_ms=int(chunk_ts_ms), reason="silence")
                            continue

                    if int(max_segment_frames) > 0 and int(len(segment_frames)) >= int(max_segment_frames):
                        tail = segment_frames[-pre_roll_frames:] if int(pre_roll_frames) > 0 else []
                        _trace_vad_tick(
                            now_ts,
                            rms=int(rms),
                            thr=int(thr),
                            is_voice=bool(is_voice),
                            chunk_ts_ms=int(chunk_ts_ms),
                        )
                        _flush_segment(end_ts_ms=int(chunk_ts_ms), reason="max_segment")
                        pre_roll = list(tail)
                        continue
                    _trace_vad_tick(
                        now_ts,
                        rms=int(rms),
                        thr=int(thr),
                        is_voice=bool(is_voice),
                        chunk_ts_ms=int(chunk_ts_ms),
                    )

                    if status:
                        now2 = time.time()
                        if (now2 - last_status_update) >= 0.2:
                            status.set_speaker_audio(
                                playback_enabled=bool(speaker_playback),
                                playback_started=bool(speaker_player_started),
                                rms_last=int(rms),
                                rms_mean_2s=last_rms_mean_2s,
                                rms_max_2s=last_rms_max_2s,
                                ts=now2,
                            )
                            last_status_update = float(now2)
                    if trace and (time.time() - recv_stats_started) >= 2.0:
                        mean = (float(recv_rms_sum) / float(recv_frames)) if recv_frames > 0 else 0.0
                        last_rms_mean_2s = float(mean)
                        last_rms_max_2s = int(recv_rms_max)
                        trace.write(
                            {
                                "type": "audio_recv_stats",
                                "frames": int(recv_frames),
                                "bytes": int(recv_bytes),
                                "rms_mean": round(float(mean), 1),
                                "rms_max": int(recv_rms_max),
                                "speaker_playback": bool(speaker_playback),
                                "speaker_player_started": bool(speaker_player_started),
                            }
                        )
                        recv_stats_started = time.time()
                        recv_frames = 0
                        recv_bytes = 0
                        recv_rms_sum = 0
                        recv_rms_max = 0
            except Exception as exc:
                now = time.time()
                err = str(exc)
                if (now - float(last_err_ts)) >= 1.0 or err != last_err:
                    emit(f"話し手音声の処理でエラーが起きました: {exc}")
                    if trace:
                        trace.write(
                            {
                                "type": "speaker_audio_loop_error",
                                "error": str(exc),
                                "last_msg_type": str(last_msg_type),
                                "last_raw_bytes": int(last_raw_bytes),
                                "traceback": traceback.format_exc()[-4000:],
                            }
                        )
                    last_err_ts = float(now)
                    last_err = err
                time.sleep(0.05)
                continue

    threading.Thread(target=_audio_loop, daemon=True).start()

    signal_store = HumanSignalStore()
    signal_events: queue.Queue = queue.Queue()

    def _talker_guide_for_calibration(message: str) -> str:
        msg = str(message or "").strip()
        if not msg:
            return ""

        # 測定の完了に近いメッセージ
        if "計測が終わりました" in msg or "軸の推定" in msg:
            return "聞き手の測定が終わりました。もうすぐ始めます。"

        # 残り秒数
        remaining = ""
        remaining_label = ""
        m = re.search(r"残り\s*(\d+)\s*秒", msg)
        if m:
            remaining = m.group(1)
            remaining_label = "残り"
        if not remaining:
            m = re.search(r"(\d+)\s*秒後", msg)
            if m:
                remaining = m.group(1)
                remaining_label = "あと"
        if not remaining:
            m = re.search(r"次の計測まで\s*(\d+)\s*秒", msg)
            if m:
                remaining = m.group(1)
                remaining_label = "あと"
        if not remaining:
            m = re.search(r"（\s*(\d+)\s*秒\s*）", msg)
            if m:
                remaining = m.group(1)
                remaining_label = "残り"

        phase = "聞き手が測定中です。話さずに待ってください。"
        if ("計測を始めます" in msg) and ("秒後" in msg):
            phase = "聞き手が測定の準備中です。話さずに待ってください。"
        if ("静止" in msg) or ("still" in msg):
            phase = "聞き手が静止の測定中です。話さずに待ってください。"
        elif ("普段どおり" in msg) or ("自然に動いて" in msg) or ("active" in msg):
            phase = "聞き手が普段の動きの測定中です。話さずに待ってください。"
        elif "頷" in msg or "nod" in msg:
            phase = "聞き手が頷きの測定中です。話さずに待ってください。"
        elif ("首を横" in msg) or ("首振" in msg) or "shake" in msg:
            phase = "聞き手が首振りの測定中です。話さずに待ってください。"
        elif ("休憩" in msg) or ("次の計測まで" in msg):
            phase = "聞き手が測定の休憩中です。話さずに待ってください。"
        elif ("準備" in msg) or ("次:" in msg):
            phase = "聞き手が測定の準備中です。話さずに待ってください。"

        if remaining:
            label = remaining_label or "残り"
            return f"{phase} {label} {remaining} 秒"
        return phase

    def _emit_calibration(message: str) -> None:
        emit(message)
        talker_msg = _talker_guide_for_calibration(message)
        if talker_msg:
            _broadcast_guide_to_talker(talker_msg)

    # 話し手と接続してから計測に入る（talker 側の待ち時間を揃えるため）
    while True:
        with talker_conn.lock:
            connected = talker_conn.sock is not None
        if connected:
            break
        time.sleep(0.1)
    emit("接続できました。これから計測を始めます。話さずに待ってください。")
    _broadcast_guide_to_talker("接続できました。これから計測を始めます。話さずに待ってください。")

    imu_calibration: ImuCalibration | None = run_calibration(
        imu_buffer,
        still_sec=calibration_still_sec,
        active_sec=calibration_active_sec,
        start_delay_sec=calibration_start_delay_sec,
        between_phases_sec=calibration_between_sec,
        wait_for_imu_sec=calibration_wait_for_imu_sec,
        log=_emit_calibration,
    )
    if status and imu_calibration is not None:
        status.set_calibration_summary(
            still_summary="" if imu_calibration.still is None else imu_calibration.still.summary(),
            active_summary="" if imu_calibration.active is None else imu_calibration.active.summary(),
            warnings=imu_calibration.warnings,
            ts=imu_calibration.finished_at,
        )

    gesture_calib: GestureCalibration | None = None
    if gesture_calibration:
        gesture_calib = run_gesture_calibration(
            imu_buffer,
            weak_sec=gesture_weak_sec,
            strong_sec=gesture_strong_sec,
            start_delay_sec=gesture_start_delay_sec,
            rest_sec=gesture_rest_sec,
            log=_emit_calibration,
        )

    imu_nod_axis_effective = imu_nod_axis
    imu_shake_axis_effective = imu_shake_axis
    if gesture_calib and auto_imu_axis_map and gesture_calib.axis_suggest:
        imu_nod_axis_effective = gesture_calib.axis_suggest.get("nod_axis", imu_nod_axis_effective)
        imu_shake_axis_effective = gesture_calib.axis_suggest.get("shake_axis", imu_shake_axis_effective)
    if status and gesture_calib is not None:
        suggest = gesture_calib.axis_suggest
        axis_map = (
            f"suggest nod={suggest.get('nod_axis','-')}, shake={suggest.get('shake_axis','-')} / "
            f"effective nod={imu_nod_axis_effective}, shake={imu_shake_axis_effective}"
        )
        status.set_gesture_calibration(
            summaries=gesture_calib.summaries(),
            axis_map=axis_map,
            ts=gesture_calib.finished_at,
        )

    player = AudioPlayer()

    threading.Thread(
        target=human_signal_loop,
        args=(imu_buffer, signal_store),
        kwargs={
            "calibration": imu_calibration,
            "gyro_sigma": human_signal_gyro_sigma,
            "abs_threshold": human_signal_abs_threshold,
            "max_age_s": human_signal_max_age_s,
            "min_consecutive_above": human_signal_min_consecutive,
            "nod_axis": imu_nod_axis_effective,
            "shake_axis": imu_shake_axis_effective,
            "event_queue": signal_events,
            "debug": debug_signal and (status is None),
            "status": status,
            "trace": trace,
        },
        daemon=True,
    ).start()

    emit("計測完了です。話し手が話すのを待ってください。")
    _broadcast_guide_to_talker("計測が終わりました。話してください。")

    transcript = LiveTranscriptBuffer(max_lines=300)
    last_pause_like_boundary = False

    last_backchannel_play = 0.0
    last_backchannel_text = ""
    recent_backchannel_ids: list[str] = []
    warned_no_talker = False

    def send_backchannel(
        *,
        selected_id: str,
        selected_text: str,
        reason: str,
        latency_ms: int,
        call_id: str,
        planned: bool,
    ) -> bool:
        nonlocal warned_no_talker
        if not send_backchannel_to_talker:
            return False
        payload: Dict[str, object] = {
            "type": "backchannel",
            "experiment_id": str(experiment_id),
            "id": str(selected_id),
            "text": str(selected_text),
            "reason": str(reason),
            "planned": bool(planned),
            "latency_ms": int(latency_ms),
            "call_id": str(call_id),
            "ts_ms": int(_now_ms()),
        }
        with talker_conn.lock:
            sock = talker_conn.sock
            addr = talker_conn.addr
            if sock is None:
                if not warned_no_talker:
                    emit("話し手(talker)が未接続なので、相槌を送れません。")
                    warned_no_talker = True
                    if trace:
                        trace.write({"type": "backchannel_send_skipped", "reason": "no_talker"})
                return False
            try:
                send_jsonl(sock, payload)
                if trace:
                    ev = dict(payload)
                    ev["type"] = "backchannel_sent"
                    ev["addr"] = addr
                    trace.write(ev)
                return True
            except Exception as exc:
                emit(f"相槌の送信に失敗しました: {exc}")
                try:
                    sock.close()
                except Exception:
                    pass
                talker_conn.sock = None
                talker_conn.addr = ""
                return False

    def play_backchannel(
        *,
        selected_id: str,
        selected_text: str,
        audio_path: Path | None,
        reason: str,
        latency_ms: int,
        call_id: str,
        planned: bool,
    ) -> bool:
        nonlocal last_backchannel_play, last_backchannel_text, recent_backchannel_ids
        if status:
            status.set_agent_decision(
                choice_id=selected_id,
                choice_text=selected_text,
                reason=reason,
                latency_ms=latency_ms,
                ts=time.time(),
            )
        if (not status) or debug_agent:
            log_only(f"選択: {selected_id} {selected_text}".strip())
            if debug_agent and reason:
                log_only(f"理由: {reason}")

        sent = send_backchannel(
            selected_id=selected_id,
            selected_text=selected_text,
            reason=reason,
            latency_ms=latency_ms,
            call_id=call_id,
            planned=planned,
        )
        played_local = False
        if local_backchannel_play and audio_path is not None:
            played_local = player.play_effect(audio_path, interrupt=True)
        if trace:
            trace.write(
                {
                    "type": "backchannel_play",
                    "call_id": call_id,
                    "thread_id": thread_id,
                    "selected_id": selected_id,
                    "selected_text": selected_text,
                    "audio_path": "" if audio_path is None else str(audio_path),
                    "sent_to_talker": bool(sent),
                    "played_local": bool(played_local),
                    "planned": bool(planned),
                }
            )
        if status and audio_path is not None:
            status.set_backchannel_playback(path=audio_path, played=(sent or played_local))
        if local_backchannel_play and played_local and ((not status) or debug_agent):
            log_only(f"再生(ローカル): {audio_path}")
        if (not sent) and (not played_local):
            return False
        last_backchannel_play = time.time()
        last_backchannel_text = selected_text
        recent_backchannel_ids.append(str(selected_id))
        if len(recent_backchannel_ids) > 20:
            del recent_backchannel_ids[:-20]
        return True

    mode_norm = str(mode or "llm").strip().lower()
    if mode_norm not in ("llm", "human", "none"):
        log_only(f"mode が不明なので llm にします: {mode}")
        mode_norm = "llm"

    if mode_norm == "none":
        log_only("モード: none（相槌を返しません）")
    elif mode_norm == "human":
        log_only("モード: human（キー入力で相槌を送ります）")

        items_by_id = {it.id: it for it in items}
        ids: list[str] = []
        raw_ids = [x.strip() for x in str(human_choice_ids or "").split(",") if x.strip()]
        if raw_ids:
            ids = [x for x in raw_ids if x in items_by_id]
        if not ids:
            by_dir: Dict[str, list] = {}
            dir_order: list[str] = []
            for it in items:
                if it.directory not in by_dir:
                    by_dir[it.directory] = []
                    dir_order.append(it.directory)
                by_dir[it.directory].append(it)

            # まずは各カテゴリから「中くらい」を1つずつ選ぶ
            for d in dir_order:
                cand = next((it for it in by_dir.get(d, []) if getattr(it, "strength", 0) == 3), None)
                if cand is None:
                    cand = by_dir.get(d, [None])[0]
                if cand is not None:
                    ids.append(cand.id)

            # 足りなければ、残りをカタログ順に埋める
            for it in items:
                if len(ids) >= int(max(1, human_choice_count)):
                    break
                if it.id not in ids:
                    ids.append(it.id)
        ids = ids[: int(max(1, human_choice_count))]
        key_map = {str(i + 1): items_by_id[x] for i, x in enumerate(ids) if x in items_by_id}

        def _human_help_lines() -> list[str]:
            lines: list[str] = []
            for k, it in key_map.items():
                lines.append(f"{k}: {it.text} (id={it.id})")
            lines.append("入力: 1-9 / id / all / help / q")
            lines.append("補足: Enter で確定します")
            return lines

        def _human_all_lines() -> list[str]:
            by_dir: Dict[str, list] = {}
            dir_order: list[str] = []
            for it in items:
                if it.directory not in by_dir:
                    by_dir[it.directory] = []
                    dir_order.append(it.directory)
                by_dir[it.directory].append(it)
            lines: list[str] = []
            for d in dir_order:
                lines.append(f"{d}:")
                for it in by_dir.get(d, []):
                    lines.append(f"  {it.id}: {it.text} (strength={it.strength})")
            return lines

        if status:
            status.set_human_menu(lines=_human_help_lines())
            status.set_ui_guide(text="人間モード: 番号か id を入力して Enter で確定します")
        else:
            log_only("\n".join(_human_help_lines()))

        def _human_input_loop() -> None:
            while True:
                try:
                    line = input().strip()
                except EOFError:
                    time.sleep(0.1)
                    continue
                if not line:
                    continue
                if line in ("q", "quit", "exit"):
                    log_only("人間入力を終了します。")
                    return
                if line in ("h", "help", "?"):
                    if status:
                        status.set_human_menu(lines=_human_help_lines())
                        status.set_ui_guide(text="人間モード: 番号か id を入力して Enter で確定します")
                    else:
                        log_only("\n".join(_human_help_lines()))
                    continue
                if line in ("a", "all", "list"):
                    if status:
                        status.set_human_menu(lines=_human_all_lines())
                        status.set_ui_guide(text="人間モード: 全候補を表示しています")
                    else:
                        log_only("\n".join(_human_all_lines()))
                    continue

                item = key_map.get(line) or items_by_id.get(line)
                if item is None:
                    log_only("見つかりません。help で一覧を見てください。")
                    continue

                ap = find_audio_file(audio_dir, item) if local_backchannel_play else None
                play_backchannel(
                    selected_id=item.id,
                    selected_text=item.text,
                    audio_path=ap,
                    reason="human",
                    latency_ms=0,
                    call_id=("human-" + uuid.uuid4().hex[:10]),
                    planned=False,
                )

        threading.Thread(target=_human_input_loop, daemon=True).start()
    else:
        log_only("モード: llm（IMU + モデルで相槌を決めます）")

    while True:
        boundary_event = False
        boundary_text: str | None = None

        # 文字起こしイベントを処理（STT結果）
        try:
            while True:
                ev = transcript_events.get_nowait()
                if not isinstance(ev, dict):
                    continue
                ev_type = str(ev.get("type", ""))
                if ev_type == "segment_final":
                    text = str(ev.get("text", "") or "").strip()
                    speaker_ts_ms = ev.get("ts_ms")
                    if not isinstance(speaker_ts_ms, int):
                        speaker_ts_ms = None
                    if text:
                        transcript.add(text=text, speaker_ts_ms=speaker_ts_ms)
                        boundary_text = text
                        boundary_event = True
                        if status:
                            status.on_transcript_spoken(text=text)
                            t_sec = int(max(0.0, time.time() - started_at_ts))
                            status.set_transcript_boundary(t_sec=t_sec, text=text, ts=time.time())
                        if trace:
                            trace.write({"type": "transcript_segment", "text": text, "speaker_ts_ms": speaker_ts_ms})
        except queue.Empty:
            pass

        now = time.time()

        # 話し手が少し黙っているなら、区切りに近い合図として扱う（区切り扱いは1回だけ）
        speaker_silence_ms_live = int(max(0, speaker_state.silence_ms))
        if speaker_state_seen and (not speaker_state.speaking):
            speaker_silence_ms_live += int(max(0.0, now - speaker_state.last_update_ts) * 1000)
        speaker_pause_like_boundary = (
            bool(speaker_state_seen)
            and (not speaker_state.speaking)
            and (speaker_silence_ms_live >= int(max(0, boundary_silence_ms)))
        )
        pause_boundary_event = bool(speaker_pause_like_boundary) and (not bool(last_pause_like_boundary))
        last_pause_like_boundary = bool(speaker_pause_like_boundary)
        if pause_boundary_event:
            boundary_event = True

        if mode_norm != "llm":
            time.sleep(0.05)
            continue

        # IMUイベントを処理（最新だけ残す）
        latest_signal_event: Dict[str, object] | None = None
        try:
            while True:
                ev = signal_events.get_nowait()
                ts = ev.get("ts") if isinstance(ev, dict) else None
                sig = ev.get("signal") if isinstance(ev, dict) else None
                if isinstance(ts, (int, float)) and isinstance(sig, dict):
                    latest_signal_event = {"ts": float(ts), "signal": dict(sig)}
        except queue.Empty:
            pass

        # IMUの合図が来たら、すぐに1回だけ判断する
        if latest_signal_event is None:
            time.sleep(0.01)
            continue

        sig = latest_signal_event.get("signal")
        ts = latest_signal_event.get("ts")
        if not isinstance(sig, dict) or not isinstance(ts, (int, float)):
            time.sleep(0.01)
            continue

        seconds_since_signal = max(0.0, now - float(ts))

        # IMUの要約
        imu_bundle = imu_buffer.build_bundle(
            now=now,
            raw_window_sec=2.0,
            raw_max_points=8,
            stats_windows_sec=[1.0, 5.0],
        )
        if imu_calibration is not None:
            imu_bundle["calibration"] = imu_calibration.to_dict()
            activity_1s = imu_bundle.get("activity_1s", {})
            if isinstance(activity_1s, dict):
                imu_bundle["normalized_activity"] = normalize_activity(activity_1s, imu_calibration)
        if gesture_calib is not None:
            imu_bundle["gesture_calibration"] = gesture_calib.to_dict()
            imu_bundle["imu_axis_map_effective"] = {
                "nod_axis": imu_nod_axis_effective,
                "shake_axis": imu_shake_axis_effective,
            }

        human_signal = dict(sig)
        human_signal["present"] = True
        human_signal["age_since_signal_s"] = round(float(seconds_since_signal), 3)
        imu_bundle["human_signal"] = human_signal

        if debug_signal or debug_agent:
            reason = str(human_signal.get("reason", ""))
            log_only(f"IMUサイン: {reason} (age={float(seconds_since_signal):.2f}s)")

        directory_allowlist: list[str] = []
        hint = human_signal.get("gesture_hint")
        if isinstance(hint, str):
            if hint == "nod":
                directory_allowlist = ["positive"]
            elif hint == "shake":
                directory_allowlist = ["negative"]

        transcript_context = transcript.context(max_lines=context_max_lines).strip()
        if not transcript_context:
            transcript_context = "文字起こしはまだありません"

        latest_received_ts = transcript.latest_received_ts()
        transcript_latest_age_s: float | None = None
        if latest_received_ts is not None:
            transcript_latest_age_s = max(0.0, now - float(latest_received_ts))

        timing: Dict[str, object] = {
            "is_boundary": bool(boundary_event),
            "seconds_since_signal": round(float(seconds_since_signal), 3),
            "speaker_speaking": bool(speaker_state.speaking),
            "speaker_silence_ms": int(speaker_silence_ms_live),
            "speaker_pause_like_boundary": bool(speaker_pause_like_boundary),
            "boundary_silence_ms": int(max(0, boundary_silence_ms)),
            "transcript_latest_age_s": None
            if transcript_latest_age_s is None
            else round(float(transcript_latest_age_s), 3),
        }

        utterance = str(boundary_text or "").strip()
        if (not utterance) and transcript_latest_age_s is not None and transcript_latest_age_s <= 2.0:
            utterance = transcript.latest_text() or ""
        utterance_t_sec = int(max(0.0, now - started_at_ts))

        call_id = uuid.uuid4().hex[:12]
        avoid_ids: list[str] = []
        if recent_backchannel_ids:
            last_id = recent_backchannel_ids[-1]
            streak = 1
            for prev_id in reversed(recent_backchannel_ids[:-1]):
                if prev_id != last_id:
                    break
                streak += 1
            if streak >= 2:
                avoid_ids = [last_id]
        if trace:
            trace.write(
                {
                    "type": "agent_call",
                    "call_id": call_id,
                    "thread_id": thread_id,
                    "utterance": utterance,
                    "utterance_t_sec": utterance_t_sec,
                    "transcript_context": transcript_context,
                    "timing": dict(timing),
                    "directory_allowlist": list(directory_allowlist),
                    "avoid_ids": list(avoid_ids),
                    "imu": imu_bundle,
                }
            )

        t0 = time.time()
        try:
            result = graph.invoke(
                {
                    "utterance": utterance,
                    "imu": imu_bundle,
                    "imu_text": json.dumps(imu_bundle, ensure_ascii=False),
                    "utterance_t_sec": utterance_t_sec,
                    "transcript_context": transcript_context,
                    "timing": timing,
                    "directory_allowlist": directory_allowlist,
                    "avoid_ids": list(avoid_ids),
                    "candidates": [],
                    "selection": {},
                    "selected_id": "",
                    "errors": [],
                },
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception as exc:
            emit(f"相槌の判断でエラーが起きました: {exc}")
            if trace:
                trace.write({"type": "agent_error", "call_id": call_id, "error": str(exc)})
            time.sleep(0.01)
            continue

        latency_ms = int(round((time.time() - t0) * 1000))

        selected_id = str(result.get("selected_id", ""))
        reason = _extract_agent_reason(result)
        if trace:
            trace.write(
                {
                    "type": "agent_result",
                    "call_id": call_id,
                    "thread_id": thread_id,
                    "latency_ms": int(latency_ms),
                    "selected_id": selected_id,
                    "reason": reason,
                    "selection": result.get("selection", {}),
                    "errors": result.get("errors", []),
                }
            )

        if selected_id == "NONE":
            if status:
                status.clear_backchannel_playback()
                status.set_agent_decision(
                    choice_id="NONE",
                    choice_text="",
                    reason=reason,
                    latency_ms=latency_ms,
                    ts=time.time(),
                )
            if debug_agent:
                log_only(f"選択: NONE ({reason})" if reason else "選択: NONE")
            time.sleep(0.01)
            continue

        selected_item = next((item for item in items if item.id == selected_id), None)
        if not selected_item:
            emit("相槌の選択に失敗しました。")
            time.sleep(0.01)
            continue

        audio_path: Path | None = None
        if local_backchannel_play:
            audio_path = find_audio_file(audio_dir, selected_item)
            if not audio_path:
                emit("音声ファイルが見つからないので、ローカル再生はできません。")

        play_backchannel(
            selected_id=selected_item.id,
            selected_text=selected_item.text,
            audio_path=audio_path,
            reason=reason,
            latency_ms=latency_ms,
            call_id=call_id,
            planned=False,
        )
        time.sleep(0.01)
        continue
