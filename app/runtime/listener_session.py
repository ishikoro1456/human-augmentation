from __future__ import annotations

import json
import queue
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from openai import OpenAI

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # langgraph のバージョン差異に備える
    InMemorySaver = None

from app.agents.backchannel_graph import build_backchannel_graph
from app.audio.player import AudioPlayer
from app.core.catalog import load_catalog
from app.core.selector import find_audio_file
from app.imu.buffer import ImuBuffer, ImuSample
from app.imu.calibration import ImuCalibration, normalize_activity, run_calibration
from app.imu.gesture_calibration import GestureCalibration, run_gesture_calibration
from app.imu.reader import read_imu_lines
from app.imu.signal import detect_backchannel_signal
from app.imu.signal_store import HumanSignalStore
from app.net.jsonl import iter_jsonl_messages
from app.runtime.status import StatusStore
from app.runtime.trace import TraceWriter
from app.transcript.live_buffer import LiveTranscriptBuffer


def _extract_agent_reason(result: Dict[str, object]) -> str:
    selection = result.get("selection", {})
    if not isinstance(selection, dict):
        return ""
    reason = selection.get("decision_reason", "") or selection.get("reason", "")
    return str(reason) if reason else ""


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


def transcript_server_loop(
    *,
    host: str,
    port: int,
    event_queue: "queue.Queue[Dict[str, object]]",
    log: callable,
    trace: TraceWriter | None = None,
) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, int(port)))
    server.listen(1)
    log(f"文字起こし受信: {host}:{int(port)} で待ち受けます。")
    if trace:
        trace.write({"type": "transcript_listen", "host": host, "port": int(port)})
    while True:
        conn, addr = server.accept()
        try:
            log(f"文字起こし受信: 接続 {addr[0]}:{addr[1]}")
            if trace:
                trace.write({"type": "transcript_connected", "addr": f"{addr[0]}:{addr[1]}"})
            for msg in iter_jsonl_messages(conn):
                msg = dict(msg)
                msg["_received_ts"] = round(time.time(), 3)
                event_queue.put(msg)
        except Exception as exc:
            log(f"文字起こし受信: エラー {exc}")
            if trace:
                trace.write({"type": "transcript_error", "error": str(exc)})
        finally:
            try:
                conn.close()
            except Exception:
                pass
            log("文字起こし受信: 切断")
            if trace:
                trace.write({"type": "transcript_disconnected"})


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
                "model": model,
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

    transcript_events: queue.Queue = queue.Queue()
    threading.Thread(
        target=transcript_server_loop,
        kwargs={
            "host": listen_host,
            "port": int(listen_port),
            "event_queue": transcript_events,
            "log": emit,
            "trace": trace,
        },
        daemon=True,
    ).start()

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
    speaker_state = SpeakerAudioState(speaking=False, silence_ms=0, last_update_ts=time.time())
    speaker_state_seen = False

    last_agent_call = 0.0
    last_backchannel_play = 0.0
    last_backchannel_text = ""
    recent_ids: list[str] = []
    recent_texts: list[str] = []

    def play_backchannel(
        *,
        selected_id: str,
        selected_text: str,
        audio_path: Path,
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

        played = player.play_effect(audio_path)
        if trace:
            trace.write(
                {
                    "type": "backchannel_play",
                    "call_id": call_id,
                    "thread_id": thread_id,
                    "selected_id": selected_id,
                    "selected_text": selected_text,
                    "audio_path": str(audio_path),
                    "played": bool(played),
                    "planned": bool(planned),
                }
            )
        if status:
            status.set_backchannel_playback(path=audio_path, played=played)
        if not played:
            emit("再生中の音があるので、相槌の再生をスキップします。")
            return False
        if (not status) or debug_agent:
            emit(f"再生: {audio_path}")
        last_backchannel_play = time.time()
        last_backchannel_text = selected_text
        recent_ids.append(selected_id)
        recent_texts.append(selected_text)
        if len(recent_ids) > 12:
            recent_ids = recent_ids[-12:]
        if len(recent_texts) > 12:
            recent_texts = recent_texts[-12:]
        while player.is_effect_playing():
            time.sleep(0.02)
        return True

    pending: Dict[str, object] | None = None

    while True:
        boundary_event = False
        boundary_text: str | None = None

        # 文字起こしイベントを処理
        try:
            while True:
                ev = transcript_events.get_nowait()
                if not isinstance(ev, dict):
                    continue
                ev_type = str(ev.get("type", ""))
                if ev_type == "speech_state":
                    speaking = bool(ev.get("speaking", False))
                    silence_ms = ev.get("silence_ms", 0)
                    if not isinstance(silence_ms, (int, float)):
                        silence_ms = 0
                    speaker_state.speaking = speaking
                    speaker_state.silence_ms = int(max(0, float(silence_ms)))
                    speaker_state.last_update_ts = time.time()
                    speaker_state_seen = True
                elif ev_type == "segment_final":
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

        # 期限切れで保留を落とす
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

        # planned を boundary で armed にする
        if pending is not None and boundary_event:
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
                continue
            if now - last_backchannel_play < backchannel_cooldown_sec:
                continue
            planned = pending.get("planned")
            planned_after_ts = pending.get("planned_after_ts", 0.0)
            if isinstance(planned_after_ts, (int, float)) and now < float(planned_after_ts):
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
                and isinstance(paudio, str)
                and isinstance(pcall, str)
                and isinstance(plat, int)
            ):
                audio_path = Path(paudio)
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

        # 呼び出し要否
        if pending is None:
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
            continue
        if now - last_backchannel_play < backchannel_cooldown_sec:
            continue
        if now - last_agent_call < agent_interval_sec:
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
                directory_allowlist = ["understanding", "agreement"]
            elif hint == "shake":
                directory_allowlist = ["question", "disagreement"]

        avoid_ids = recent_ids[-2:] if recent_ids else []

        transcript_context = transcript.context(max_lines=context_max_lines).strip()
        if not transcript_context:
            transcript_context = "文字起こしはまだありません"

        speaker_silence_ms_live = int(max(0, speaker_state.silence_ms))
        if speaker_state_seen and (not speaker_state.speaking):
            speaker_silence_ms_live += int(max(0.0, now - speaker_state.last_update_ts) * 1000)
        speaker_pause_like_boundary = (
            bool(speaker_state_seen)
            and (not speaker_state.speaking)
            and (speaker_silence_ms_live >= int(max(0, boundary_silence_ms)))
        )

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

        audio_path = find_audio_file(audio_dir, selected_item)
        if not audio_path:
            emit("音声ファイルが見つかりません。")
            pending = None
            continue

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
                "audio_path": str(audio_path),
                "reason": reason,
                "latency_ms": int(latency_ms),
                "planned_at_ts": round(time.time(), 3),
                "wait_ms": int(wait_ms),
            }
            pending["planned_after_ts"] = time.time() + max(0.0, float(wait_ms) / 1000.0)
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
