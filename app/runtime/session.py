import json
import queue
import threading
import time
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
    tilt_axis: str,
    tick_sec: float = 0.1,
    event_queue: "queue.Queue[Dict[str, object]] | None" = None,
    debug: bool = False,
    status: StatusStore | None = None,
) -> None:
    last_eligible: bool | None = None
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
            tilt_axis=tilt_axis,
        )
        store.update(ts=now, signal=signal)
        present = bool(signal.get("present", False))
        hint = signal.get("gesture_hint")
        eligible = bool(present) and isinstance(hint, str) and hint in ("nod", "shake", "tilt")
        if event_queue is not None:
            if eligible and (last_eligible is False or last_eligible is None):
                event_queue.put({"ts": now, "signal": dict(signal)})
        if status:
            status.set_human_signal(text=str(signal.get("reason", "")))
        if debug and not status:
            if last_eligible is None or eligible != last_eligible:
                print(f"IMUサイン: {signal.get('reason', '')}")
            last_eligible = eligible
        else:
            last_eligible = eligible
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
    imu_tilt_axis: str,
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
            raw_max_points=24,
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
                "tilt_axis": imu_tilt_axis,
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
            elif hint == "tilt":
                directory_allowlist = ["question"]
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
    imu_tilt_axis: str = "gx",
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

    items = load_catalog(catalog_path)
    timeline = None
    if transcript_path and transcript_path.exists():
        timeline = TranscriptTimeline.from_file(transcript_path)

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
    imu_tilt_axis_effective = imu_tilt_axis
    if gesture_calib and auto_imu_axis_map and gesture_calib.axis_suggest:
        imu_nod_axis_effective = gesture_calib.axis_suggest.get("nod_axis", imu_nod_axis_effective)
        imu_shake_axis_effective = gesture_calib.axis_suggest.get("shake_axis", imu_shake_axis_effective)
        imu_tilt_axis_effective = gesture_calib.axis_suggest.get("tilt_axis", imu_tilt_axis_effective)
    if status and gesture_calib is not None:
        suggest = gesture_calib.axis_suggest
        axis_map = (
            f"suggest nod={suggest.get('nod_axis','-')}, shake={suggest.get('shake_axis','-')}, "
            f"tilt={suggest.get('tilt_axis','-')} / "
            f"effective nod={imu_nod_axis_effective}, shake={imu_shake_axis_effective}, tilt={imu_tilt_axis_effective}"
        )
        status.set_gesture_calibration(
            summaries=gesture_calib.summaries(),
            axis_map=axis_map,
            ts=gesture_calib.finished_at,
        )

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
            "tilt_axis": imu_tilt_axis_effective,
            "event_queue": signal_events,
            "debug": debug_signal and (status is None),
            "status": status,
        },
        daemon=True,
    ).start()

    if startup_wait_sec > 0:
        emit(f"{int(round(startup_wait_sec))}秒後に文字起こしの読み上げを始めます。")
        time.sleep(startup_wait_sec)

    player = AudioPlayer()
    last_agent_call = 0.0
    last_backchannel_play = 0.0
    last_backchannel_text = ""

    if not timeline:
        emit("文字起こしが見つからないので、相槌エージェントを待機します。")
        while True:
            time.sleep(0.2)
    transcript_events: queue.Queue = queue.Queue()
    cache_dir = tts_cache_dir or Path("data/tts_cache")
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

    if require_human_signal:
        emit("文字起こしの読み上げを開始しました。相槌はIMU閾値到達（3回以上）で判断します。")
    else:
        emit("文字起こしの読み上げを開始しました。相槌はチャンク境界(読み上げ直後)で判断します。")

    recent_ids: list[str] = []
    recent_texts: list[str] = []

    # IMU閾値到達イベント用のキュー
    imu_trigger_queue: queue.Queue = queue.Queue()
    last_transcript_text: str = ""
    last_transcript_t_sec: int = 0
    last_count_display: float = 0.0

    def on_imu_threshold_reached(gesture: str, counts: Dict[str, int]) -> None:
        """IMU閾値に達したときのコールバック"""
        imu_trigger_queue.put({"gesture": gesture, "counts": counts, "ts": time.time()})

    # 閾値監視を設定（有効期間は長めに30秒）
    gesture_accumulation_sec = 30.0
    if require_human_signal:
        signal_store.set_threshold_callback(
            on_imu_threshold_reached,
            min_count=min_gesture_count,
            max_age_s=gesture_accumulation_sec,
        )

    transcript_done = False

    while not transcript_done:
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
                    if status:
                        status.set_transcript_boundary(t_sec=seg.t_sec, text=seg.text, ts=time.time())
                    if debug_transcript and not status:
                        print(f"文字起こし(直後): [{seg.t_sec:04d}s] {seg.text}")
                # segment_end で止まっている再生を即座に再開（IMUトリガー待ちで止まらないように）
                if resume is not None:
                    resume.set()
                    # require_human_signal: false の場合はここでLLM呼び出し
                    if not require_human_signal:
                        imu_trigger_queue.put({
                            "gesture": "segment_end",
                            "counts": {},
                            "ts": time.time(),
                            "force": True,
                        })
        except queue.Empty:
            pass

        # 定期的にカウントを表示（1秒ごと）
        now_for_count = time.time()
        if require_human_signal and (debug_signal or debug_agent) and (now_for_count - last_count_display) >= 1.0:
            gesture_counts = signal_store.count_by_gesture(
                max_age_s=gesture_accumulation_sec,
                now=now_for_count,
            )
            total = sum(gesture_counts.values())
            if total > 0:
                emit(f"現在のカウント: {gesture_counts} (閾値: {min_gesture_count})")
            last_count_display = now_for_count

        # IMUトリガーを処理
        try:
            imu_event = imu_trigger_queue.get_nowait()
        except queue.Empty:
            continue

        # IMU閾値に達した！LLMを呼び出す
        dominant_gesture = imu_event.get("gesture", "")
        gesture_counts = imu_event.get("counts", {})
        is_forced = imu_event.get("force", False)

        now = time.time()
        transcript_context = speaker.get_spoken_context()

        if debug_signal or debug_agent:
            if is_forced:
                emit(f"トリガー: segment_end")
            else:
                emit(f"トリガー: IMU閾値到達 {dominant_gesture} (カウント: {gesture_counts})")

        if debug_agent:
            emit(f"IMU: {imu_buffer.format_status_line(now=now)}")

        # エピソードを消費してリセット
        episode_summary: Dict[str, object] = {}
        if require_human_signal and not is_forced:
            episodes = signal_store.consume_episodes(
                max_age_s=gesture_accumulation_sec,
                now=now,
            )
            episode_summary = signal_store.summarize_episodes(episodes)
            episode_summary["dominant_gesture"] = dominant_gesture
            if debug_signal or debug_agent:
                ep_count = episode_summary.get("count", 0)
                best_score = episode_summary.get("best_nod_score", 0)
                emit(f"エピソード: {ep_count}件, best_nod_score={best_score}, 判定: {dominant_gesture}")
            # 閾値フラグをリセット（次の発火を許可）
            signal_store.reset_threshold()

        if now - last_backchannel_play < backchannel_cooldown_sec:
            if debug_signal or debug_agent:
                emit("スキップ: クールダウン中")
            if status:
                status.clear_backchannel_playback()
                status.set_agent_decision(
                    choice_id="NONE",
                    choice_text="",
                    reason="直近に相槌を出したのでクールダウンします。",
                    latency_ms=0,
                    ts=time.time(),
                )
            # 閾値リセット（再度発火可能にする）
            if require_human_signal:
                signal_store.reset_threshold()
            continue

        if now - last_agent_call < agent_interval_sec:
            if debug_signal or debug_agent:
                emit("スキップ: 間隔制限")
            if status:
                status.clear_backchannel_playback()
                status.set_agent_decision(
                    choice_id="NONE",
                    choice_text="",
                    reason="エージェント呼び出し間隔を守るため今回は返しません。",
                    latency_ms=0,
                    ts=time.time(),
                )
            # 閾値リセット（再度発火可能にする）
            if require_human_signal:
                signal_store.reset_threshold()
            continue

        try:

            imu_bundle = imu_buffer.build_bundle(
                now=now,
                raw_window_sec=2.0,
                raw_max_points=24,
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
                    "tilt_axis": imu_tilt_axis_effective,
                }

            sig = signal_store.snapshot()
            present_recent = sig.is_recent(now=now, hold_sec=human_signal_hold_sec)
            base_signal: Dict[str, object] = sig.latest
            if present_recent and sig.last_present_signal:
                base_signal = sig.last_present_signal

            human_signal = dict(base_signal) if base_signal else {}
            raw_present = bool(sig.latest.get("present", False)) if sig.latest else False
            human_signal["present_raw"] = raw_present
            human_signal["present"] = bool(present_recent)
            human_signal["present_latched"] = bool(present_recent)
            human_signal["hold_sec"] = float(human_signal_hold_sec)
            if sig.last_present_at is not None:
                human_signal["age_since_present_s"] = round(max(0.0, now - sig.last_present_at), 3)
            imu_bundle["human_signal"] = human_signal
            if status:
                reason = str(human_signal.get("reason", ""))
                age_s = human_signal.get("age_since_present_s")
                if isinstance(age_s, (int, float)):
                    status.set_human_signal_used(text=f"{reason} (age={float(age_s):.2f}s)")
                else:
                    status.set_human_signal_used(text=reason)
            if debug_signal or debug_agent:
                reason = str(human_signal.get("reason", ""))
                age_s = human_signal.get("age_since_present_s")
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
            # require_human_signal: true の場合はエピソードで既にチェック済み
            # require_human_signal: false の場合でも、present なら返す（従来互換）
            # エピソードの情報を imu_bundle に追加
            if require_human_signal and episode_summary:
                imu_bundle["episode_summary"] = episode_summary
                # エピソードのベストシグナルを human_signal として使う
                best_signal = episode_summary.get("best_signal")
                if isinstance(best_signal, dict) and best_signal:
                    human_signal = dict(best_signal)
                    human_signal["present"] = True
                    human_signal["from_episode"] = True
                    imu_bundle["human_signal"] = human_signal
            audio_state: Dict[str, object] = {
                "transcript_playing": player.is_music_playing(),
                "backchannel_playing": player.is_effect_playing(),
                "decision_point": "on_boundary" if require_human_signal else "after_transcript_chunk",
            }
            recent_backchannel: Dict[str, object] = {
                "seconds_ago": None
                if last_backchannel_play == 0.0
                else round(now - last_backchannel_play, 3),
                "text": last_backchannel_text,
                "history_ids": recent_ids[-6:],
                "history_texts": recent_texts[-6:],
            }

            # directory_allowlist をエピソードのヒントに基づいて設定
            directory_allowlist: list[str] = []
            if require_human_signal and episode_summary:
                if episode_summary.get("has_nod"):
                    directory_allowlist = ["understanding", "agreement"]
                elif episode_summary.get("has_shake"):
                    directory_allowlist = ["question", "disagreement"]
            avoid_ids = recent_ids[-2:] if recent_ids else []

            t0 = time.time()
            result = graph.invoke(
                {
                    "utterance": last_transcript_text,
                    "imu": imu_bundle,
                    "imu_text": json.dumps(imu_bundle, ensure_ascii=False),
                    "audio_state": audio_state,
                    "recent_backchannel": recent_backchannel,
                    "utterance_t_sec": last_transcript_t_sec,
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
                continue

            selected_item = next((item for item in items if item.id == selected_id), None)
            if not selected_item:
                emit("相槌の選択に失敗しました。")
                continue

            audio_path = find_audio_file(audio_dir, selected_item)
            if not audio_path:
                emit("音声ファイルが見つかりません。")
                continue

            if status:
                status.set_agent_decision(
                    choice_id=selected_item.id,
                    choice_text=selected_item.text,
                    reason=reason,
                    latency_ms=latency_ms,
                    ts=time.time(),
                )

            if (not status) or debug_agent:
                emit(
                    f"選択: {selected_item.directory} s{selected_item.strength} "
                    f"n{selected_item.nod} {selected_item.text}"
                )
            if debug_agent and reason:
                emit(f"理由: {reason}")

            played = player.play_effect(audio_path)
            if status:
                status.set_backchannel_playback(path=audio_path, played=played)
            if not played:
                emit("再生中の音があるので、相槌の再生をスキップします。")
                continue

            if (not status) or debug_agent:
                emit(f"再生: {audio_path}")
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
