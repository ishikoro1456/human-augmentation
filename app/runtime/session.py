import json
import queue
import threading
import time
import uuid
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
from app.runtime.status import StatusStore
from app.runtime.trace import TraceWriter
from app.transcript.speaker import TranscriptSpeaker
from app.transcript.timeline import TranscriptTimeline


def _extract_agent_reason(result: Dict[str, object]) -> str:
    """エージェントの結果からreasonを抽出する"""
    selection = result.get("selection", {})
    if not isinstance(selection, dict):
        return ""

    # decision_reason または reason を取得
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
        buffer.add(
            ImuSample(
                ts=now,
                ax=ax,
                ay=ay,
                az=az,
                gx=gx,
                gy=gy,
                gz=gz,
            )
        )
        if status:
            if now - last_status > 0.2:
                status.set_imu(
                    motion_text=buffer.format_status_line(now=now),
                    event="raw",
                    ts=now,
                )
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

        # 同じ「動きのまとまり」の途中で eligible が揺れても、イベントは1回だけにする
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


def backchannel_loop_on_signal(
    signal_events: "queue.Queue[Dict[str, object]]",
    *,
    graph: object,
    items: list,
    audio_dir: Path,
    imu_buffer: ImuBuffer,
    imu_calibration: ImuCalibration | None,
    gesture_calib: GestureCalibration | None,
    imu_nod_axis: str,
    imu_shake_axis: str,
    speaker: TranscriptSpeaker,
    player: AudioPlayer,
    thread_id: str,
    agent_interval_sec: float,
    backchannel_cooldown_sec: float,
    human_signal_hold_sec: float,
    duck_music_volume: float = 0.35,
    debug_agent: bool = False,
    status: StatusStore | None = None,
) -> None:
    def _emit(message: str) -> None:
        if status:
            status.log(message)
        else:
            print(message)

    last_agent_call = 0.0
    last_backchannel_play = 0.0
    last_backchannel_text = ""
    recent_ids: list[str] = []
    recent_texts: list[str] = []
    pending: Dict[str, object] | None = None

    while True:
        try:
            ev = signal_events.get(timeout=0.1)
            if isinstance(ev, dict) and ev.get("kind") == "stop":
                return
            if isinstance(ev, dict):
                ts = ev.get("ts")
                signal = ev.get("signal")
                if isinstance(ts, (int, float)) and isinstance(signal, dict):
                    pending = {"ts": float(ts), "signal": dict(signal)}
            # 最新だけ残す
            while True:
                try:
                    ev2 = signal_events.get_nowait()
                except queue.Empty:
                    break
                if isinstance(ev2, dict) and ev2.get("kind") == "stop":
                    return
                ts2 = ev2.get("ts") if isinstance(ev2, dict) else None
                sig2 = ev2.get("signal") if isinstance(ev2, dict) else None
                if isinstance(ts2, (int, float)) and isinstance(sig2, dict):
                    pending = {"ts": float(ts2), "signal": dict(sig2)}
        except queue.Empty:
            pass

        if pending is None:
            continue

        ts = pending.get("ts")
        signal = pending.get("signal")
        if not isinstance(ts, (int, float)) or not isinstance(signal, dict):
            pending = None
            continue

        now = time.time()
        age = max(0.0, now - float(ts))
        if age > float(human_signal_hold_sec):
            pending = None
            continue

        # いまは返せないが、hold の間は保持しておく
        if now - last_backchannel_play < backchannel_cooldown_sec:
            continue
        if now - last_agent_call < agent_interval_sec:
            continue
        if player.is_effect_playing():
            continue

        current = speaker.get_current()
        utterance = current.text if current else ""
        utterance_t_sec = current.t_sec if current else ""
        transcript_context = speaker.get_spoken_context()

        imu_bundle = imu_buffer.build_bundle(
            now=float(ts),
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
                "nod_axis": imu_nod_axis,
                "shake_axis": imu_shake_axis,
            }

        human_signal = dict(signal)
        human_signal["signal_ts"] = round(float(ts), 3)
        human_signal["age_s"] = round(age, 3)
        imu_bundle["human_signal"] = human_signal
        if status:
            reason = str(human_signal.get("reason", ""))
            status.set_human_signal_used(text=f"{reason} (age={age:.2f}s)")

        if gesture_calib is not None:
            hint = human_signal.get("gesture_hint")
            if isinstance(hint, str) and hint in ("nod", "shake"):
                axis = imu_nod_axis if hint == "nod" else imu_shake_axis
                current_axis_mean = None
                axis_mean_map = human_signal.get("axis_abs_mean_1s", {})
                if isinstance(axis_mean_map, dict):
                    v = axis_mean_map.get(axis)
                    if isinstance(v, (int, float)):
                        current_axis_mean = float(v)
                weak_ex = gesture_calib.examples.get(f"{hint}_weak")
                strong_ex = gesture_calib.examples.get(f"{hint}_strong")
                weak_ref = None if weak_ex is None else weak_ex.axis_abs_mean.get(axis)
                strong_ref = None if strong_ex is None else strong_ex.axis_abs_mean.get(axis)
                if (
                    current_axis_mean is not None
                    and isinstance(weak_ref, (int, float))
                    and isinstance(strong_ref, (int, float))
                    and float(strong_ref) > float(weak_ref)
                ):
                    norm = (current_axis_mean - float(weak_ref)) / (float(strong_ref) - float(weak_ref))
                    norm = max(0.0, min(1.0, norm))
                    level = int(round(1 + norm * 4))
                    imu_bundle["gesture_intensity"] = {
                        "hint": hint,
                        "axis": axis,
                        "axis_abs_mean_1s": round(current_axis_mean, 3),
                        "weak_ref": round(float(weak_ref), 3),
                        "strong_ref": round(float(strong_ref), 3),
                        "norm_0to1": round(norm, 3),
                        "level_1to5": level,
                    }

        audio_state: Dict[str, object] = {
            "transcript_playing": player.is_music_playing(),
            "backchannel_playing": player.is_effect_playing(),
            "decision_point": "on_human_signal",
        }
        recent_backchannel: Dict[str, object] = {
            "seconds_ago": None
            if last_backchannel_play == 0.0
            else round(now - last_backchannel_play, 3),
            "text": last_backchannel_text,
            "history_ids": recent_ids[-6:],
            "history_texts": recent_texts[-6:],
        }

        hint = human_signal.get("gesture_hint")
        directory_allowlist: list[str] | None = None
        if isinstance(hint, str):
            if hint == "nod":
                directory_allowlist = ["understanding", "agreement"]
            elif hint == "shake":
                directory_allowlist = ["question", "disagreement"]
        avoid_ids = recent_ids[-2:] if recent_ids else []

        try:
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
                    "directory_allowlist": directory_allowlist or [],
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
                if debug_agent and not status:
                    _emit(f"選択: NONE ({reason})" if reason else "選択: NONE")
                pending = None
                continue

            selected_item = next((item for item in items if item.id == selected_id), None)
            if not selected_item:
                pending = None
                continue

            audio_path = find_audio_file(audio_dir, selected_item)
            if not audio_path:
                pending = None
                continue

            if status:
                status.set_agent_decision(
                    choice_id=selected_item.id,
                    choice_text=selected_item.text,
                    reason=reason,
                    latency_ms=latency_ms,
                    ts=time.time(),
                )

            restore_volume: float | None = None
            if player.is_music_playing():
                restore_volume = player.get_music_volume()
                player.set_music_volume(float(duck_music_volume))

            played = player.play_effect(audio_path)
            if status:
                status.set_backchannel_playback(path=audio_path, played=played)
            if played:
                last_backchannel_play = time.time()
                last_backchannel_text = selected_item.text
                recent_ids.append(selected_item.id)
                recent_texts.append(selected_item.text)
                if len(recent_ids) > 12:
                    recent_ids = recent_ids[-12:]
                if len(recent_texts) > 12:
                    recent_texts = recent_texts[-12:]
                while player.is_effect_playing():
                    time.sleep(0.02)
            if restore_volume is not None:
                player.set_music_volume(restore_volume)
            pending = None
        except Exception as exc:
            if status:
                status.clear_backchannel_playback()
                status.set_agent_decision(
                    choice_id="NONE",
                    choice_text="",
                    reason=f"エラーのため今回は返しません: {exc}",
                    latency_ms=0,
                    ts=time.time(),
                )
            pending = None


def run_session(
    catalog_path: Path,
    audio_dir: Path,
    port: str,
    baud: int,
    model: str,
    thread_id: str,
    status: StatusStore | None = None,
    trace: TraceWriter | None = None,
    debug_imu: bool = False,
    transcript_path: Optional[Path] = None,
    transcript_start_sec: int = 0,
    debug_transcript: bool = False,
    debug_agent: bool = False,
    debug_signal: bool = False,
    tts_model: str = "gpt-4o-mini-tts",
    tts_voice: str = "alloy",
    tts_format: str = "mp3",
    tts_cache_dir: Optional[Path] = None,
    agent_interval_sec: float = 1.0,
    backchannel_cooldown_sec: float = 2.0,
    calibration_still_sec: float = 10.0,
    calibration_active_sec: float = 20.0,
    calibration_start_delay_sec: float = 0.0,
    calibration_between_sec: float = 0.0,
    calibration_wait_for_imu_sec: float = 15.0,
    require_human_signal: bool = True,
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
    gesture_start_delay_sec: float = 1.0,
    gesture_rest_sec: float = 1.0,
    auto_imu_axis_map: bool = True,
    startup_wait_sec: float = 0.0,
    min_gesture_count: int = 3,
) -> None:
    def emit(message: str) -> None:
        if status:
            status.log(message)
        else:
            print(message)
            if trace:
                trace.log(message, source="print")

    items = load_catalog(catalog_path)
    timeline = None
    if transcript_path and transcript_path.exists():
        timeline = TranscriptTimeline.from_file(transcript_path)

    if trace:
        trace.write(
            {
                "type": "session_start",
                "thread_id": thread_id,
                "model": model,
                "transcript_path": None if transcript_path is None else str(transcript_path),
                "transcript_start_sec": int(transcript_start_sec),
                "port": port,
                "baud": int(baud),
                "require_human_signal": bool(require_human_signal),
                "human_signal_hold_sec": float(human_signal_hold_sec),
                "human_signal_gyro_sigma": float(human_signal_gyro_sigma),
                "human_signal_abs_threshold": float(human_signal_abs_threshold),
                "human_signal_max_age_s": float(human_signal_max_age_s),
                "human_signal_min_consecutive": int(human_signal_min_consecutive),
                "imu_axes": {
                    "nod_axis": imu_nod_axis,
                    "shake_axis": imu_shake_axis,
                },
                "gesture_calibration": bool(gesture_calibration),
                "debug": {
                    "imu": bool(debug_imu),
                    "transcript": bool(debug_transcript),
                    "agent": bool(debug_agent),
                    "signal": bool(debug_signal),
                },
            }
        )

    client = OpenAI()
    checkpointer = InMemorySaver() if InMemorySaver else None
    graph = build_backchannel_graph(client, model, items).compile(checkpointer=checkpointer)

    imu_buffer = ImuBuffer(max_seconds=600.0)
    threading.Thread(
        target=imu_loop,
        args=(port, baud, imu_buffer, debug_imu, status),
        daemon=True,
    ).start()

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

    if startup_wait_sec > 0:
        emit(f"{int(round(startup_wait_sec))}秒後に文字起こしの読み上げを始めます。")
        time.sleep(startup_wait_sec)

    player = AudioPlayer()
    last_agent_call = 0.0
    last_backchannel_play = 0.0
    last_backchannel_text = ""

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

        restore_volume: float | None = None
        try:
            if player.is_music_playing():
                restore_volume = player.get_music_volume()
                player.set_music_volume(float(duck_music_volume))

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
                        "transcript_was_playing": restore_volume is not None,
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
        finally:
            if restore_volume is not None:
                player.set_music_volume(restore_volume)

    if not timeline:
        emit("文字起こしが見つからないので、相槌エージェントを待機します。")
        while True:
            time.sleep(0.2)
    transcript_events: queue.Queue = queue.Queue()
    cache_dir = tts_cache_dir or Path("data/runtime/tts_cache")
    speaker = TranscriptSpeaker(
        timeline=timeline,
        client=client,
        player=player,
        cache_dir=cache_dir,
        event_queue=transcript_events,
        status=status,
        model=tts_model,
        voice=tts_voice,
        response_format=tts_format,
        start_sec=transcript_start_sec,
    )
    try:
        while True:
            signal_events.get_nowait()
    except queue.Empty:
        pass
    speaker.start()

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

    if require_human_signal:
        emit(
            "文字起こしの読み上げを開始しました。"
            f"相槌はIMUサインを検出したら最大{float(human_signal_hold_sec):g}秒以内に判断します。"
            "基本は区切りで判断します。"
        )
    else:
        emit("文字起こしの読み上げを開始しました。相槌はチャンク境界(読み上げ直後)で判断します。")

    recent_ids: list[str] = []
    recent_texts: list[str] = []

    last_transcript_text: str = ""
    last_transcript_t_sec: int = 0
    last_count_display: float = 0.0
    gesture_accumulation_sec = 30.0  # デバッグ用の表示窓
    context_window_sec = 60
    history_stride_sec = 120
    history_max_lines = 10
    deadline_guard_sec = 0.3
    max_wait_for_boundary_sec = float(human_signal_hold_sec)
    duck_music_volume = 0.35

    pending: Dict[str, object] | None = None  # IMUサインの保留（最大 human_signal_hold_sec ）

    transcript_done = False

    while not transcript_done:
        boundary_seg_text: str | None = None
        boundary_seg_t_sec: int | None = None

        # transcript_events を処理（コンテクスト更新用）
        try:
            event = transcript_events.get(timeout=0.05)
            if event.kind == "segment_start":
                if debug_transcript and event.segment and not status:
                    print(f"文字起こし(再生中): [{event.segment.t_sec:04d}s] {event.segment.text}")
            elif event.kind == "error":
                if event.error:
                    emit(event.error)
            elif event.kind == "done":
                emit("文字起こしの読み上げが終わりました。")
                signal_events.put({"kind": "stop"})
                transcript_done = True
            elif event.kind == "segment_end":
                seg = event.segment
                resume = event.resume
                if seg is not None:
                    last_transcript_text = seg.text
                    last_transcript_t_sec = seg.t_sec
                    boundary_seg_text = seg.text
                    boundary_seg_t_sec = seg.t_sec
                    if status:
                        status.set_transcript_boundary(t_sec=seg.t_sec, text=seg.text, ts=time.time())
                    if debug_transcript and not status:
                        print(f"文字起こし(直後): [{seg.t_sec:04d}s] {seg.text}")
                if (
                    require_human_signal
                    and pending is not None
                    and pending.get("planned") is not None
                    and not bool(pending.get("planned_armed", False))
                ):
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
                # segment_end で止まっている再生を即座に再開（IMUトリガー待ちで止まらないように）
                if resume is not None:
                    resume.set()
        except queue.Empty:
            pass

        # IMUイベントを処理（最新だけ残す）
        latest_signal_event: Dict[str, object] | None = None
        try:
            while True:
                ev = signal_events.get_nowait()
                if isinstance(ev, dict) and ev.get("kind") == "stop":
                    continue
                ts = ev.get("ts") if isinstance(ev, dict) else None
                sig = ev.get("signal") if isinstance(ev, dict) else None
                if isinstance(ts, (int, float)) and isinstance(sig, dict):
                    latest_signal_event = {"ts": float(ts), "signal": dict(sig)}
        except queue.Empty:
            pass

        if require_human_signal and latest_signal_event is not None:
            sig = latest_signal_event.get("signal", {})
            ts = latest_signal_event.get("ts")
            if isinstance(sig, dict) and isinstance(ts, (int, float)):
                hint = str(sig.get("gesture_hint", "other"))
                pending = {
                    "signal_ts": float(ts),
                    "deadline_ts": float(ts) + float(human_signal_hold_sec),
                    "deadline_call_ts": float(ts)
                    + max(0.0, float(human_signal_hold_sec) - float(deadline_guard_sec)),
                    "deadline_called": False,
                    "wait_used": False,
                    "wait_until_ts": 0.0,
                    "planned": None,
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
                            "deadline_call_ts": round(float(pending["deadline_call_ts"]), 3),
                            "signal": dict(sig),
                        }
                    )
                if debug_signal or debug_agent:
                    emit(f"保留: {hint} (3秒以内に判断)")

        # 定期的にカウントを表示（1秒ごと）
        now_for_count = time.time()
        if require_human_signal and (debug_signal or debug_agent) and (now_for_count - last_count_display) >= 1.0:
            gesture_counts = signal_store.count_by_gesture(
                max_age_s=gesture_accumulation_sec,
                now=now_for_count,
            )
            gesture_points = signal_store.points_by_gesture(
                max_age_s=gesture_accumulation_sec,
                now=now_for_count,
            )
            total = sum(gesture_counts.values())
            if total > 0:
                emit(
                    f"現在のカウント: {gesture_counts} / "
                    f"ポイント: {gesture_points} "
                    f"(保留: {'あり' if pending else 'なし'})"
                )
            last_count_display = now_for_count

        now = time.time()
        if require_human_signal and pending is not None:
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
                            "deadline_called": bool(pending.get("deadline_called", False)),
                            "wait_used": bool(pending.get("wait_used", False)),
                            "planned": pending.get("planned"),
                            "planned_armed": bool(pending.get("planned_armed", False)),
                        }
                    )
                pending = None

        decision_point = ""
        is_boundary = False
        has_signal = False
        seconds_since_signal = None
        wait_allowed = False
        wait_budget_ms = 0
        boundary_remaining_s: float | None = None
        playback_info = speaker.get_current_playback()
        if isinstance(playback_info, dict):
            rs = playback_info.get("remaining_s")
            if isinstance(rs, (int, float)):
                boundary_remaining_s = float(rs)

        if require_human_signal and pending is not None:
            planned = pending.get("planned")
            planned_armed = bool(pending.get("planned_armed", False))
            if isinstance(planned, dict) and planned_armed:
                if player.is_effect_playing():
                    continue
                if now - last_backchannel_play < backchannel_cooldown_sec:
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

        # 今回の呼び出し要否を決める
        mark_pending_key: str | None = None
        if require_human_signal:
            if pending is None:
                continue
            has_signal = True
            signal_ts = pending.get("signal_ts")
            if isinstance(signal_ts, (int, float)):
                seconds_since_signal = max(0.0, now - float(signal_ts))
            is_boundary = boundary_seg_text is not None
            if is_boundary:
                decision_point = "boundary"
            else:
                if pending.get("planned") is not None:
                    continue
                wait_until_ts = pending.get("wait_until_ts", 0.0)
                if isinstance(wait_until_ts, (int, float)) and now < float(wait_until_ts):
                    continue
                deadline_called = bool(pending.get("deadline_called", False))
                deadline_call_ts = pending.get("deadline_call_ts")
                if (not deadline_called) and isinstance(deadline_call_ts, (int, float)) and now >= float(deadline_call_ts):
                    decision_point = "deadline_no_boundary"
                    mark_pending_key = "deadline_called"
                else:
                    continue
        else:
            if boundary_seg_text is None:
                continue
            decision_point = "boundary"
            is_boundary = True
            has_signal = False

        # いま呼べない条件は待つ（保留は残す）
        if player.is_effect_playing():
            continue
        if now - last_backchannel_play < backchannel_cooldown_sec:
            continue
        if now - last_agent_call < agent_interval_sec:
            continue
        if mark_pending_key and require_human_signal and pending is not None:
            pending[mark_pending_key] = True

        if require_human_signal and pending is not None:
            wait_used = bool(pending.get("wait_used", False))
            wait_allowed = False
            deadline_ts = pending.get("deadline_ts")
            if isinstance(deadline_ts, (int, float)):
                remaining_ms = int(max(0.0, float(deadline_ts) - now) * 1000)
                wait_budget_ms = remaining_ms
                if boundary_remaining_s is not None:
                    wait_budget_ms = min(wait_budget_ms, int(max(0.0, boundary_remaining_s * 1000)))
                    wait_allowed = (
                        (not wait_used)
                        and (decision_point == "deadline_no_boundary")
                        and (boundary_remaining_s <= float(max_wait_for_boundary_sec))
                        and (int(boundary_remaining_s * 1000) <= remaining_ms)
                    )

        # 発話とコンテクストを作る
        current_seg = speaker.get_current()
        utterance = last_transcript_text
        utterance_t_sec = last_transcript_t_sec
        current_sec = last_transcript_t_sec
        if is_boundary:
            if boundary_seg_text is not None:
                utterance = boundary_seg_text
            if boundary_seg_t_sec is not None:
                utterance_t_sec = boundary_seg_t_sec
                current_sec = boundary_seg_t_sec
        else:
            if current_seg is not None:
                utterance = current_seg.text
                utterance_t_sec = current_seg.t_sec
                current_sec = current_seg.t_sec

        transcript_context = timeline.to_context(
            current_sec,
            window_sec=context_window_sec,
            history_stride_sec=history_stride_sec,
            history_max_lines=history_max_lines,
        )
        if not is_boundary and current_seg is not None:
            transcript_context = (
                f"{transcript_context}\n\n"
                "再生中の文（区切りではありません）:\n"
                f"{current_seg.text}"
            )

        try:

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

            sig = signal_store.snapshot()
            present_recent = sig.is_recent(now=now, hold_sec=human_signal_hold_sec)
            base_signal: Dict[str, object] = {}
            if require_human_signal and pending is not None:
                ps = pending.get("signal")
                if isinstance(ps, dict):
                    base_signal = dict(ps)
            if not base_signal and isinstance(sig.latest, dict):
                base_signal = dict(sig.latest)
            if (not require_human_signal) and present_recent and sig.last_present_signal:
                base_signal = dict(sig.last_present_signal)

            human_signal = dict(base_signal) if base_signal else {}
            raw_present = bool(sig.latest.get("present", False)) if sig.latest else False
            human_signal["present_raw"] = raw_present
            human_signal["present"] = bool(has_signal) if require_human_signal else bool(present_recent)
            human_signal["present_latched"] = bool(has_signal) if require_human_signal else bool(present_recent)
            human_signal["hold_sec"] = float(human_signal_hold_sec)
            if require_human_signal and pending is not None and isinstance(pending.get("signal_ts"), (int, float)):
                human_signal["signal_ts"] = round(float(pending.get("signal_ts", 0.0)), 3)
                if seconds_since_signal is not None:
                    human_signal["age_since_signal_s"] = round(float(seconds_since_signal), 3)
            elif sig.last_present_at is not None:
                human_signal["age_since_present_s"] = round(max(0.0, now - sig.last_present_at), 3)
            imu_bundle["human_signal"] = human_signal
            if status:
                reason = str(human_signal.get("reason", ""))
                age_s = human_signal.get("age_since_signal_s") or human_signal.get("age_since_present_s")
                if isinstance(age_s, (int, float)):
                    status.set_human_signal_used(text=f"{reason} (age={float(age_s):.2f}s)")
                else:
                    status.set_human_signal_used(text=reason)
            if debug_signal or debug_agent:
                reason = str(human_signal.get("reason", ""))
                age_s = human_signal.get("age_since_signal_s") or human_signal.get("age_since_present_s")
                if isinstance(age_s, (int, float)):
                    emit(f"IMUサイン: {reason} (age={float(age_s):.2f}s)")
                else:
                    emit(f"IMUサイン: {reason}")

            if gesture_calib is not None:
                hint = human_signal.get("gesture_hint")
                if isinstance(hint, str) and hint in ("nod", "shake"):
                    axis = imu_nod_axis_effective if hint == "nod" else imu_shake_axis_effective
                    current_axis_mean = None
                    axis_mean_map = human_signal.get("axis_abs_mean_1s", {})
                    if isinstance(axis_mean_map, dict):
                        v = axis_mean_map.get(axis)
                        if isinstance(v, (int, float)):
                            current_axis_mean = float(v)
                    weak_ex = gesture_calib.examples.get(f"{hint}_weak")
                    strong_ex = gesture_calib.examples.get(f"{hint}_strong")
                    weak_ref = None if weak_ex is None else weak_ex.axis_abs_mean.get(axis)
                    strong_ref = None if strong_ex is None else strong_ex.axis_abs_mean.get(axis)
                    if (
                        current_axis_mean is not None
                        and isinstance(weak_ref, (int, float))
                        and isinstance(strong_ref, (int, float))
                        and float(strong_ref) > float(weak_ref)
                    ):
                        norm = (current_axis_mean - float(weak_ref)) / (float(strong_ref) - float(weak_ref))
                        norm = max(0.0, min(1.0, norm))
                        level = int(round(1 + norm * 4))
                        imu_bundle["gesture_intensity"] = {
                            "hint": hint,
                            "axis": axis,
                            "axis_abs_mean_1s": round(current_axis_mean, 3),
                            "weak_ref": round(float(weak_ref), 3),
                            "strong_ref": round(float(strong_ref), 3),
                            "norm_0to1": round(norm, 3),
                            "level_1to5": level,
                        }
            audio_state: Dict[str, object] = {
                "transcript_playing": player.is_music_playing(),
                "backchannel_playing": player.is_effect_playing(),
                "decision_point": decision_point,
                "is_boundary": is_boundary,
            }
            recent_backchannel: Dict[str, object] = {
                "seconds_ago": None
                if last_backchannel_play == 0.0
                else round(now - last_backchannel_play, 3),
                "text": last_backchannel_text,
                "history_ids": recent_ids[-6:],
                "history_texts": recent_texts[-6:],
            }

            directory_allowlist: list[str] = []
            if require_human_signal:
                hint = human_signal.get("gesture_hint")
                if isinstance(hint, str):
                    if hint == "nod":
                        directory_allowlist = ["understanding", "agreement"]
                    elif hint == "shake":
                        directory_allowlist = ["question", "disagreement"]
            avoid_ids = recent_ids[-2:] if recent_ids else []

            timing: Dict[str, object] = {
                "is_boundary": is_boundary,
                "has_signal": has_signal,
                "decision_point": decision_point,
                "seconds_since_signal": None if seconds_since_signal is None else round(float(seconds_since_signal), 3),
                "deadline_sec": float(human_signal_hold_sec) if require_human_signal else None,
                "wait_allowed": bool(wait_allowed),
                "wait_budget_ms": int(wait_budget_ms),
                "segment_remaining_s": None if boundary_remaining_s is None else round(float(boundary_remaining_s), 3),
                "segment_duration_s": None
                if not isinstance(playback_info, dict)
                else playback_info.get("duration_s"),
                "segment_elapsed_s": None if not isinstance(playback_info, dict) else playback_info.get("elapsed_s"),
                "transcript_playing": bool(audio_state.get("transcript_playing", False)),
                "backchannel_playing": bool(audio_state.get("backchannel_playing", False)),
            }

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
                    status.set_agent_decision(
                        choice_id="NONE",
                        choice_text="",
                        reason=reason,
                        latency_ms=latency_ms,
                        ts=time.time(),
                    )
                if debug_agent:
                    emit(f"選択: NONE ({reason})" if reason else "選択: NONE")
                if require_human_signal:
                    pending = None
                continue

            selected_item = next((item for item in items if item.id == selected_id), None)
            if not selected_item:
                emit("相槌の選択に失敗しました。")
                continue

            audio_path = find_audio_file(audio_dir, selected_item)
            if not audio_path:
                emit("音声ファイルが見つかりません。")
                continue

            if decision_action == "WAIT" and require_human_signal and pending is not None:
                pending["wait_used"] = True
                pending["planned"] = {
                    "call_id": call_id,
                    "selected_id": selected_item.id,
                    "selected_text": selected_item.text,
                    "audio_path": str(audio_path),
                    "reason": reason,
                    "latency_ms": int(latency_ms),
                    "planned_at_ts": round(time.time(), 3),
                }
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
            if not played:
                continue
            if require_human_signal:
                pending = None
        except Exception as exc:
            emit(f"エージェント処理でエラーが起きました: {exc}")
            if status:
                status.clear_backchannel_playback()
                status.set_agent_decision(
                    choice_id="NONE",
                    choice_text="",
                    reason=f"エラーのため今回は返しません: {exc}",
                    latency_ms=0,
                    ts=time.time(),
                )
