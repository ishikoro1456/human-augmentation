from __future__ import annotations

import json
import queue
import socket
import threading
import time
import uuid
import base64
import audioop
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
) -> None:
    last_eligible: bool | None = None
    last_present: bool | None = None
    episode_fired = False
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


def transcript_server_loop(
    *,
    host: str,
    port: int,
    experiment_id: str,
    event_queue: "queue.Queue[Dict[str, object]]",
    conn_state: TalkerConnection,
    log: callable,
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
        try:
            with conn_state.lock:
                conn_state.sock = conn
                conn_state.addr = f"{addr[0]}:{addr[1]}"
            log(f"話し手接続: 接続 {addr[0]}:{addr[1]}")
            if trace:
                trace.write({"type": "talker_connected", "addr": f"{addr[0]}:{addr[1]}"})
            try:
                send_jsonl(conn, {"type": "session", "experiment_id": str(experiment_id), "ts_ms": int(_now_ms())})
                if trace:
                    trace.write({"type": "session_sent", "addr": f"{addr[0]}:{addr[1]}"})
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
            if trace:
                trace.write({"type": "talker_disconnected"})


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
    agent_interval_sec: float = 1.0,
    backchannel_cooldown_sec: float = 2.0,
    calibration_still_sec: float = 10.0,
    calibration_active_sec: float = 20.0,
    calibration_start_delay_sec: float = 3.0,
    calibration_between_sec: float = 3.0,
    calibration_wait_for_imu_sec: float = 15.0,
    human_signal_gyro_sigma: float = 3.0,
    human_signal_abs_threshold: float = 8.0,
    human_signal_max_age_s: float = 1.5,
    human_signal_min_consecutive: int = 3,
    human_signal_hold_sec: float = 3.0,
    imu_nod_axis: str = "gy",
    imu_shake_axis: str = "gz",
    gesture_calibration: bool = False,
    gesture_weak_sec: float = 2.0,
    gesture_strong_sec: float = 2.0,
    gesture_start_delay_sec: float = 2.0,
    gesture_rest_sec: float = 2.0,
    auto_imu_axis_map: bool = True,
    boundary_silence_ms: int = 350,
    context_max_lines: int = 10,
    early_call_delay_sec: float = 0.2,
    deadline_guard_sec: float = 0.3,
    speaker_playback: bool = True,
    speaker_playback_bin: str = "ffplay",
    stt_model: str = "whisper-1",
    stt_language: str = "ja",
    stt_prompt: str = "",
    stt_segments_dir: Path = Path("data/stt_segments_listener"),
    vad_frame_ms: int = 20,
    vad_pre_roll_ms: int = 200,
    vad_silence_end_ms: int = 500,
    vad_min_speech_ms: int = 300,
    vad_calib_sec: float = 1.0,
    vad_threshold_rms: int = 0,
    vad_threshold_mult: float = 3.0,
    send_backchannel_to_talker: bool = True,
    local_backchannel_play: bool = False,
    mode: str = "llm",
    human_choice_count: int = 9,
    human_choice_ids: str = "",
) -> None:
    def emit(message: str) -> None:
        if status:
            status.log(message)
        else:
            print(message)
            if trace:
                trace.log(message, source="print")

    started_at_ts = status.snapshot().started_at if status else time.time()

    items = load_catalog(catalog_path)

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
            }
        )

    client = OpenAI()
    checkpointer = InMemorySaver() if InMemorySaver else None
    graph = build_backchannel_graph(client, model, items).compile(checkpointer=checkpointer)

    imu_buffer = ImuBuffer(max_seconds=600.0)
    threading.Thread(target=imu_loop, args=(port, baud, imu_buffer, debug_imu, status), daemon=True).start()

    # 文字起こしの受信は、IMUの計測より先に待ち受けを開始しておく（talker が先に起動しても接続できるように）
    talker_conn = TalkerConnection()
    net_events: queue.Queue = queue.Queue()
    threading.Thread(
        target=transcript_server_loop,
        kwargs={
            "host": listen_host,
            "port": int(listen_port),
            "experiment_id": str(experiment_id),
            "event_queue": net_events,
            "conn_state": talker_conn,
            "log": emit,
            "trace": trace,
        },
        daemon=True,
    ).start()

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
            if not isinstance(seg_id, int):
                continue
            if not isinstance(pcm, (bytes, bytearray)):
                continue
            if not isinstance(start_ms, int) or not isinstance(end_ms, int):
                continue
            if not isinstance(sample_rate, int):
                continue

            wav_path = stt_segments_dir / f"seg_{seg_id:04d}_{start_ms}_{end_ms}.wav"
            try:
                _write_wav(wav_path, pcm_s16le=bytes(pcm), sample_rate=sample_rate)
                with wav_path.open("rb") as f:
                    params: Dict[str, object] = {
                        "model": stt_model,
                        "file": f,
                        "response_format": "text",
                    }
                    if stt_language:
                        params["language"] = stt_language
                    if stt_prompt:
                        params["prompt"] = stt_prompt
                    text = client.audio.transcriptions.create(**params)
                text = str(text).strip()
            except Exception as exc:
                emit(f"文字起こしに失敗しました: {exc}")
                if trace:
                    trace.write({"type": "stt_error", "segment_id": int(seg_id), "error": str(exc)})
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
                    trace.write({"type": "stt_segment", "segment_id": int(seg_id), "text": text})

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
    silence_frames = 0
    pre_roll: list[bytes] = []
    segment_frames: list[bytes] = []
    segment_start_ms = 0
    seg_id = 0

    def _recompute_vad_derived() -> tuple[int, int, int, int]:
        frame_ms = int(max(5, speaker_frame_ms))
        frame_samples = int(speaker_sample_rate * frame_ms / 1000)
        frame_bytes = frame_samples * 2
        pre_roll_frames = max(0, int(int(vad_pre_roll_ms) / frame_ms))
        silence_end_frames = max(1, int(int(vad_silence_end_ms) / frame_ms))
        min_speech_frames = max(1, int(int(vad_min_speech_ms) / frame_ms))
        return frame_bytes, pre_roll_frames, silence_end_frames, min_speech_frames

    def _audio_loop() -> None:
        nonlocal speaker_state_seen, speaker_sample_rate, speaker_frame_ms
        nonlocal speaker_player, speaker_player_started, speaker_player_warned
        nonlocal noise_rms, calib_until, speaking, silence_frames, pre_roll, segment_frames, segment_start_ms, seg_id

        frame_bytes, pre_roll_frames, silence_end_frames, min_speech_frames = _recompute_vad_derived()

        while True:
            msg = net_events.get()
            if not isinstance(msg, dict):
                continue
            msg_type = str(msg.get("type", ""))

            if msg_type == "hello":
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
                    frame_bytes, pre_roll_frames, silence_end_frames, min_speech_frames = _recompute_vad_derived()
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

            if speaker_playback:
                if not speaker_player_started:
                    speaker_player_started = bool(speaker_player.start())
                    if (not speaker_player_started) and (not speaker_player_warned):
                        emit("ffplay が見つからないので、話し手音声の再生を省略します。")
                        speaker_player_warned = True
                if speaker_player_started:
                    speaker_player.write(raw)

            # フレーム処理（念のため、複数フレームがまとまって届いても扱う）
            i = 0
            while i + frame_bytes <= len(raw) and frame_bytes > 0:
                frame = raw[i : i + frame_bytes]
                i += frame_bytes
                rms_msg = msg.get("rms")
                rms = int(rms_msg) if (i == frame_bytes and isinstance(rms_msg, (int, float))) else int(audioop.rms(frame, 2))

                # しきい値の推定（無音のときだけ更新）
                if time.time() < calib_until:
                    noise_rms = (noise_rms * 0.95) + (rms * 0.05)
                elif not speaking:
                    noise_rms = (noise_rms * 0.995) + (rms * 0.005)
                thr = int(vad_threshold_rms) if int(vad_threshold_rms) > 0 else int(max(300.0, noise_rms * float(vad_threshold_mult)))
                is_voice = rms >= thr

                pre_roll.append(frame)
                if len(pre_roll) > pre_roll_frames:
                    pre_roll = pre_roll[-pre_roll_frames:]

                now_ts = time.time()
                if not speaking:
                    if is_voice:
                        speaking = True
                        silence_frames = 0
                        segment_frames = list(pre_roll)
                        segment_start_ms = int(msg.get("ts_ms", _now_ms()))
                        speaker_state.speaking = True
                        speaker_state.silence_ms = 0
                        speaker_state.last_update_ts = now_ts
                        speaker_state_seen = True
                    else:
                        speaker_state.speaking = False
                        speaker_state.silence_ms = int(max(0, speaker_state.silence_ms + int(speaker_frame_ms)))
                        speaker_state.last_update_ts = now_ts
                        speaker_state_seen = True
                    continue

                # speaking 中
                segment_frames.append(frame)
                if is_voice:
                    silence_frames = 0
                    speaker_state.speaking = True
                    speaker_state.silence_ms = 0
                    speaker_state.last_update_ts = now_ts
                    speaker_state_seen = True
                    continue

                silence_frames += 1
                speaker_state.speaking = False
                speaker_state.silence_ms = int(silence_frames * int(speaker_frame_ms))
                speaker_state.last_update_ts = now_ts
                speaker_state_seen = True

                if silence_frames < silence_end_frames:
                    continue

                # 区切り
                speaking = False
                seg_end_ms = int(msg.get("ts_ms", _now_ms()))
                silence_frames = 0

                if len(segment_frames) >= min_speech_frames:
                    seg_id += 1
                    stt_queue.put(
                        {
                            "segment_id": int(seg_id),
                            "pcm_s16le": b"".join(segment_frames),
                            "start_ts_ms": int(segment_start_ms),
                            "end_ts_ms": int(seg_end_ms),
                            "sample_rate": int(speaker_sample_rate),
                        }
                    )
                segment_frames = []
                pre_roll = []

    threading.Thread(target=_audio_loop, daemon=True).start()

    signal_store = HumanSignalStore()
    signal_events: queue.Queue = queue.Queue()

    imu_calibration: ImuCalibration | None = run_calibration(
        imu_buffer,
        still_sec=calibration_still_sec,
        active_sec=calibration_active_sec,
        start_delay_sec=calibration_start_delay_sec,
        between_phases_sec=calibration_between_sec,
        wait_for_imu_sec=calibration_wait_for_imu_sec,
        log=emit,
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
            log=emit,
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
        },
        daemon=True,
    ).start()

    emit("聞き手アプリを開始しました。IMUの合図に反応して相槌を返します。")

    transcript = LiveTranscriptBuffer(max_lines=300)
    last_pause_like_boundary = False

    last_agent_call = 0.0
    last_backchannel_play = 0.0
    last_backchannel_text = ""
    recent_ids: list[str] = []
    recent_texts: list[str] = []
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
        nonlocal last_backchannel_play, last_backchannel_text, recent_ids, recent_texts
        if status:
            status.set_agent_decision(
                choice_id=selected_id,
                choice_text=selected_text,
                reason=reason,
                latency_ms=latency_ms,
                ts=time.time(),
            )
        if (not status) or debug_agent:
            emit(f"選択: {selected_id} {selected_text}".strip())
            if debug_agent and reason:
                emit(f"理由: {reason}")

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
            played_local = player.play_effect(audio_path)
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
        if local_backchannel_play and (not played_local):
            emit("再生中の音があるので、相槌の再生をスキップします。")
        if local_backchannel_play and played_local and ((not status) or debug_agent):
            emit(f"再生(ローカル): {audio_path}")
        if (not sent) and (not played_local):
            return False
        last_backchannel_play = time.time()
        last_backchannel_text = selected_text
        recent_ids.append(selected_id)
        recent_texts.append(selected_text)
        if len(recent_ids) > 12:
            recent_ids = recent_ids[-12:]
        if len(recent_texts) > 12:
            recent_texts = recent_texts[-12:]
        if local_backchannel_play:
            while player.is_effect_playing():
                time.sleep(0.02)
        return True

    mode_norm = str(mode or "llm").strip().lower()
    if mode_norm not in ("llm", "human", "none"):
        emit(f"mode が不明なので llm にします: {mode}")
        mode_norm = "llm"

    if mode_norm == "none":
        emit("モード: none（相槌を返しません）")
    elif mode_norm == "human":
        emit("モード: human（キー入力で相槌を送ります）")

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

        def _human_help() -> str:
            lines = []
            for k, it in key_map.items():
                lines.append(f"{k}: {it.text} (id={it.id}, category={it.directory}, strength={it.strength})")
            lines.append("入力: 1-9 / id / all / help / q")
            lines.append("補足: Enter で確定します")
            return "\n".join(lines)

        def _human_all() -> str:
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
            return "\n".join(lines)

        emit(_human_help())

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
                    emit("人間入力を終了します。")
                    return
                if line in ("h", "help", "?"):
                    emit(_human_help())
                    continue
                if line in ("a", "all", "list"):
                    emit(_human_all())
                    continue

                item = key_map.get(line) or items_by_id.get(line)
                if item is None:
                    emit("見つかりません。help で一覧を見てください。")
                    continue

                if time.time() - last_backchannel_play < backchannel_cooldown_sec:
                    emit("クールダウン中です。少し待ってください。")
                    continue

                if trace:
                    trace.write({"type": "human_choice", "input": line, "selected_id": item.id, "selected_text": item.text})

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
        emit("モード: llm（IMU + モデルで相槌を決めます）")

    pending: Dict[str, object] | None = None

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

        if latest_signal_event is not None:
            sig = latest_signal_event.get("signal", {})
            ts = latest_signal_event.get("ts")
            if isinstance(sig, dict) and isinstance(ts, (int, float)):
                hint = str(sig.get("gesture_hint", "other"))
                pending = {
                    "signal_ts": float(ts),
                    "deadline_ts": float(ts) + float(human_signal_hold_sec),
                    "early_call_ts": float(ts) + float(early_call_delay_sec),
                    "deadline_call_ts": float(ts) + max(0.0, float(human_signal_hold_sec) - float(deadline_guard_sec)),
                    "early_called": False,
                    "deadline_called": False,
                    "wait_used": False,
                    "wait_until_ts": 0.0,
                    "planned": None,
                    "planned_after_ts": 0.0,
                    "planned_armed": False,
                    "planned_armed_ts": 0.0,
                    "signal": dict(sig),
                }
                if trace:
                    trace.write(
                        {
                            "type": "pending_set",
                            "hint": hint,
                            "signal_ts": round(float(ts), 3),
                            "deadline_ts": round(float(pending["deadline_ts"]), 3),
                            "early_call_ts": round(float(pending["early_call_ts"]), 3),
                            "deadline_call_ts": round(float(pending["deadline_call_ts"]), 3),
                            "signal": dict(sig),
                        }
                    )
                if debug_signal or debug_agent:
                    emit(f"保留: {hint} (最大{float(human_signal_hold_sec):g}秒)")

        # planned を boundary で armed にする
        if pending is not None and (boundary_event or speaker_pause_like_boundary):
            if pending.get("planned") is not None and not bool(pending.get("planned_armed", False)):
                pending["planned_armed"] = True
                pending["planned_armed_ts"] = time.time()
                if trace:
                    trace.write(
                        {
                            "type": "planned_armed",
                            "signal_ts": pending.get("signal_ts"),
                            "armed_ts": round(float(pending["planned_armed_ts"]), 3),
                            "planned": pending.get("planned"),
                        }
                    )

        # planned が armed なら再生する
        if pending is not None and bool(pending.get("planned_armed", False)) and isinstance(pending.get("planned"), dict):
            if player.is_effect_playing():
                time.sleep(0.01)
                continue
            if now - last_backchannel_play < backchannel_cooldown_sec:
                time.sleep(0.01)
                continue
            planned = pending.get("planned")
            planned_after_ts = pending.get("planned_after_ts", 0.0)
            if isinstance(planned_after_ts, (int, float)) and now < float(planned_after_ts):
                time.sleep(0.01)
                continue
            pid = planned.get("selected_id")
            ptext = planned.get("selected_text")
            paudio = planned.get("audio_path")
            preason = planned.get("reason", "")
            plat = planned.get("latency_ms", 0)
            pcall = planned.get("call_id", "")
            if (
                isinstance(pid, str)
                and isinstance(ptext, str)
                and isinstance(pcall, str)
                and isinstance(plat, int)
            ):
                audio_path = Path(paudio) if isinstance(paudio, str) and paudio else None
                played = play_backchannel(
                    selected_id=pid,
                    selected_text=ptext,
                    audio_path=audio_path,
                    reason=str(preason),
                    latency_ms=int(plat),
                    call_id=pcall,
                    planned=True,
                )
                if played:
                    pending = None
                continue
            pending = None
            continue

        # 期限切れで保留を落とす（planned 再生を優先する）
        if pending is not None:
            deadline_ts = pending.get("deadline_ts")
            if isinstance(deadline_ts, (int, float)) and now > float(deadline_ts):
                if debug_signal or debug_agent:
                    emit("保留: 期限切れで見送ります")
                if trace:
                    trace.write(
                        {
                            "type": "pending_expired",
                            "signal_ts": pending.get("signal_ts"),
                            "deadline_ts": pending.get("deadline_ts"),
                            "early_called": bool(pending.get("early_called", False)),
                            "deadline_called": bool(pending.get("deadline_called", False)),
                            "wait_used": bool(pending.get("wait_used", False)),
                            "planned": pending.get("planned"),
                            "planned_armed": bool(pending.get("planned_armed", False)),
                        }
                    )
                pending = None

        # 呼び出し要否
        if pending is None:
            time.sleep(0.01)
            continue

        decision_point = ""
        is_boundary = False
        has_signal = True
        seconds_since_signal = None
        wait_allowed = False
        wait_budget_ms = 0

        signal_ts = pending.get("signal_ts")
        if isinstance(signal_ts, (int, float)):
            seconds_since_signal = max(0.0, now - float(signal_ts))

        if boundary_event:
            decision_point = "boundary"
            is_boundary = True
        else:
            wait_until_ts = pending.get("wait_until_ts", 0.0)
            if isinstance(wait_until_ts, (int, float)) and now < float(wait_until_ts):
                time.sleep(0.01)
                continue
            early_called = bool(pending.get("early_called", False))
            deadline_called = bool(pending.get("deadline_called", False))
            early_call_ts = pending.get("early_call_ts")
            deadline_call_ts = pending.get("deadline_call_ts")
            mark_pending_key: str | None = None
            if (not early_called) and isinstance(early_call_ts, (int, float)) and now >= float(early_call_ts):
                decision_point = "signal_early"
                mark_pending_key = "early_called"
            elif (not deadline_called) and isinstance(deadline_call_ts, (int, float)) and now >= float(deadline_call_ts):
                decision_point = "deadline_no_boundary"
                mark_pending_key = "deadline_called"
            else:
                continue

            if mark_pending_key and pending is not None:
                pending[mark_pending_key] = True

        if player.is_effect_playing():
            time.sleep(0.01)
            continue
        if now - last_backchannel_play < backchannel_cooldown_sec:
            time.sleep(0.01)
            continue
        if now - last_agent_call < agent_interval_sec:
            time.sleep(0.01)
            continue

        deadline_ts = pending.get("deadline_ts")
        if isinstance(deadline_ts, (int, float)):
            remaining_ms = int(max(0.0, float(deadline_ts) - now) * 1000)
            wait_budget_ms = remaining_ms
            wait_used = bool(pending.get("wait_used", False))
            wait_allowed = (not wait_used) and (decision_point == "signal_early") and remaining_ms > 0

        # IMUの要約
        imu_bundle = imu_buffer.build_bundle(
            now=now,
            raw_window_sec=2.0,
            raw_max_points=8,
            stats_windows_sec=[1.0, 5.0, 30.0, 120.0, 600.0],
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

        # 合図は pending の signal を優先
        human_signal = dict(pending.get("signal", {}) if isinstance(pending.get("signal"), dict) else {})
        human_signal["present"] = True
        human_signal["hold_sec"] = float(human_signal_hold_sec)
        if seconds_since_signal is not None:
            human_signal["age_since_signal_s"] = round(float(seconds_since_signal), 3)
        imu_bundle["human_signal"] = human_signal

        if debug_signal or debug_agent:
            reason = str(human_signal.get("reason", ""))
            age_s = human_signal.get("age_since_signal_s")
            if isinstance(age_s, (int, float)):
                emit(f"IMUサイン: {reason} (age={float(age_s):.2f}s)")
            else:
                emit(f"IMUサイン: {reason}")

        directory_allowlist: list[str] = []
        hint = human_signal.get("gesture_hint")
        if isinstance(hint, str):
            if hint == "nod":
                directory_allowlist = ["positive"]
            elif hint == "shake":
                directory_allowlist = ["negative"]

        avoid_ids = recent_ids[-2:] if recent_ids else []

        transcript_context = transcript.context(max_lines=context_max_lines).strip()
        if not transcript_context:
            transcript_context = "文字起こしはまだありません"

        timing: Dict[str, object] = {
            "is_boundary": bool(is_boundary),
            "has_signal": bool(has_signal),
            "decision_point": decision_point,
            "seconds_since_signal": None if seconds_since_signal is None else round(float(seconds_since_signal), 3),
            "deadline_sec": float(human_signal_hold_sec),
            "wait_allowed": bool(wait_allowed),
            "wait_budget_ms": int(wait_budget_ms),
            "speaker_speaking": bool(speaker_state.speaking),
            "speaker_silence_ms": int(speaker_silence_ms_live),
            "speaker_pause_like_boundary": bool(speaker_pause_like_boundary),
            "boundary_silence_ms": int(max(0, boundary_silence_ms)),
        }

        audio_state: Dict[str, object] = {
            "speaker_speaking": bool(speaker_state.speaking),
            "backchannel_playing": player.is_effect_playing(),
            "decision_point": decision_point,
            "is_boundary": bool(is_boundary),
            "speaker_pause_like_boundary": bool(speaker_pause_like_boundary),
        }
        recent_backchannel: Dict[str, object] = {
            "seconds_ago": None if last_backchannel_play == 0.0 else round(now - last_backchannel_play, 3),
            "text": last_backchannel_text,
            "history_ids": recent_ids[-6:],
            "history_texts": recent_texts[-6:],
        }

        utterance = boundary_text or transcript.latest_text() or ""
        utterance_t_sec = int(max(0.0, now - started_at_ts))

        call_id = uuid.uuid4().hex[:12]
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
                    "audio_state": dict(audio_state),
                    "recent_backchannel": dict(recent_backchannel),
                    "directory_allowlist": list(directory_allowlist),
                    "avoid_ids": list(avoid_ids),
                    "imu": imu_bundle,
                }
            )

        t0 = time.time()
        result = graph.invoke(
            {
                "utterance": utterance,
                "imu": imu_bundle,
                "imu_text": json.dumps(imu_bundle, ensure_ascii=False),
                "audio_state": audio_state,
                "recent_backchannel": recent_backchannel,
                "utterance_t_sec": utterance_t_sec,
                "transcript_context": transcript_context,
                "timing": timing,
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
        decision_action = ""
        dec = result.get("decision", {})
        if isinstance(dec, dict):
            decision_action = str(dec.get("action", "")) if dec.get("action") is not None else ""
        if trace:
            trace.write(
                {
                    "type": "agent_result",
                    "call_id": call_id,
                    "thread_id": thread_id,
                    "latency_ms": int(latency_ms),
                    "selected_id": selected_id,
                    "reason": reason,
                    "decision": result.get("decision", {}),
                    "selection": result.get("selection", {}),
                    "errors": result.get("errors", []),
                }
            )

        if selected_id == "NONE":
            if status:
                status.clear_backchannel_playback()
                status.set_agent_decision(choice_id="NONE", choice_text="", reason=reason, latency_ms=latency_ms, ts=time.time())
            if debug_agent:
                emit(f"選択: NONE ({reason})" if reason else "選択: NONE")
            pending = None
            continue

        selected_item = next((item for item in items if item.id == selected_id), None)
        if not selected_item:
            emit("相槌の選択に失敗しました。")
            pending = None
            continue

        audio_path: Path | None = None
        if local_backchannel_play:
            audio_path = find_audio_file(audio_dir, selected_item)
            if not audio_path:
                emit("音声ファイルが見つからないので、ローカル再生はできません。")

        if decision_action == "WAIT" and pending is not None:
            wait_ms = 0
            if isinstance(dec, dict):
                wm = dec.get("wait_ms", 0)
                if isinstance(wm, (int, float)):
                    wait_ms = int(wm)
            pending["wait_used"] = True
            pending["planned"] = {
                "call_id": call_id,
                "selected_id": selected_item.id,
                "selected_text": selected_item.text,
                "audio_path": "" if audio_path is None else str(audio_path),
                "reason": reason,
                "latency_ms": int(latency_ms),
                "planned_at_ts": round(time.time(), 3),
                "wait_ms": int(wait_ms),
            }
            pending["planned_after_ts"] = time.time() + max(0.0, float(wait_ms) / 1000.0)
            # モデルが遅いときでも、planned をすぐ期限切れにしないための猶予
            try:
                cur_deadline = float(pending.get("deadline_ts", 0.0))
            except Exception:
                cur_deadline = 0.0
            pending["deadline_ts"] = max(cur_deadline, float(pending["planned_after_ts"]) + 1.0)
            pending["planned_armed"] = False
            pending["planned_armed_ts"] = 0.0
            if trace:
                trace.write(
                    {
                        "type": "planned_set",
                        "call_id": call_id,
                        "thread_id": thread_id,
                        "signal_ts": pending.get("signal_ts"),
                        "planned": pending.get("planned"),
                    }
                )
            if status:
                status.clear_backchannel_playback()
                status.set_agent_decision(
                    choice_id="WAIT",
                    choice_text=selected_item.text,
                    reason=("区切り待ち: " + reason) if reason else "区切り待ち",
                    latency_ms=latency_ms,
                    ts=time.time(),
                )
            if debug_agent:
                emit(f"選択: WAIT (区切りで返す) {selected_item.text} {reason}".strip())
            continue

        played = play_backchannel(
            selected_id=selected_item.id,
            selected_text=selected_item.text,
            audio_path=audio_path,
            reason=reason,
            latency_ms=latency_ms,
            call_id=call_id,
            planned=False,
        )
        if played:
            pending = None
