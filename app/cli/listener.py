#!/usr/bin/env python3
import argparse
import sys
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.runtime.listener_session import run_listener_session
from app.runtime.status import StatusStore
from app.runtime.trace import TraceWriter


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Listener app (IMU + backchannel)")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--catalog", default="data/catalog.tsv")
    parser.add_argument("--audio-dir", default="data/backchannel")
    parser.add_argument("--thread-id", default="listener-session")

    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8765)

    parser.add_argument("--port", default="/dev/cu.usbserial-310")
    parser.add_argument("--baud", type=int, default=115200)

    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--ui-mode", choices=["participant", "debug"], default="participant")
    parser.add_argument("--trace-jsonl", default="")
    parser.add_argument("--experiment-id", default="", help="省略すると自動生成します")

    parser.add_argument("--mode", choices=["llm", "human", "none"], default="llm")
    parser.add_argument("--human-choice-count", type=int, default=9)
    parser.add_argument("--human-choice-ids", default="")

    parser.add_argument("--debug-imu", action="store_true")
    parser.add_argument("--debug-agent", action="store_true")
    parser.add_argument("--debug-signal", action="store_true")

    parser.add_argument("--agent-interval-sec", type=float, default=1.0)
    parser.add_argument("--backchannel-cooldown-sec", type=float, default=2.0)

    parser.add_argument("--no-speaker-playback", action="store_true")
    parser.add_argument("--speaker-playback-bin", default="ffplay")

    parser.add_argument("--stt-model", default="gpt-4o-transcribe")
    parser.add_argument("--stt-language", default="ja")
    parser.add_argument("--stt-prompt", default="")
    parser.add_argument("--stt-segments-dir", default="data/stt_segments_listener")

    parser.add_argument("--vad-frame-ms", type=int, default=20)
    parser.add_argument("--vad-pre-roll-ms", type=int, default=200)
    parser.add_argument("--vad-silence-end-ms", type=int, default=500)
    parser.add_argument("--vad-min-speech-ms", type=int, default=300)
    parser.add_argument("--vad-start-voice-frames", type=int, default=2)
    parser.add_argument("--vad-min-voice-ms", type=int, default=80)
    parser.add_argument("--vad-calib-sec", type=float, default=1.0)
    parser.add_argument("--vad-threshold-rms", type=int, default=0)
    parser.add_argument("--vad-threshold-mult", type=float, default=3.0)
    parser.add_argument(
        "--vad-max-segment-ms",
        type=int,
        default=20000,
        help="無音が来なくてもこの長さで強制的に区切ります（0以下で無効）",
    )

    parser.add_argument("--no-send-backchannel-to-talker", action="store_true")
    parser.add_argument("--local-backchannel-play", action="store_true")

    parser.add_argument("--calibration-still-sec", type=float, default=10.0)
    parser.add_argument("--calibration-start-delay-sec", type=float, default=3.0)
    parser.add_argument("--calibration-active-sec", type=float, default=20.0)
    parser.add_argument("--calibration-between-sec", type=float, default=3.0)
    parser.add_argument("--calibration-wait-for-imu-sec", type=float, default=15.0)

    parser.add_argument("--gesture-calibration", action="store_true")
    parser.add_argument("--gesture-weak-sec", type=float, default=2.0)
    parser.add_argument("--gesture-strong-sec", type=float, default=2.0)
    parser.add_argument("--gesture-start-delay-sec", type=float, default=2.0)
    parser.add_argument("--gesture-rest-sec", type=float, default=2.0)
    parser.add_argument("--no-auto-imu-axis-map", action="store_true")

    parser.add_argument("--human-signal-gyro-sigma", type=float, default=3.0)
    parser.add_argument("--human-signal-abs-threshold", type=float, default=8.0)
    parser.add_argument("--human-signal-max-age-s", type=float, default=1.5)
    parser.add_argument("--human-signal-min-consecutive", type=int, default=3)
    parser.add_argument("--human-signal-hold-sec", type=float, default=3.0)

    parser.add_argument("--imu-nod-axis", choices=["gx", "gy", "gz"], default="gy")
    parser.add_argument("--imu-shake-axis", choices=["gx", "gy", "gz"], default="gz")

    parser.add_argument("--boundary-silence-ms", type=int, default=350)
    parser.add_argument("--context-max-lines", type=int, default=10)
    parser.add_argument("--early-call-delay-sec", type=float, default=0.2)

    args = parser.parse_args()

    exp_id = str(args.experiment_id or "").strip()
    if not exp_id:
        exp_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

    trace = TraceWriter(Path(args.trace_jsonl)) if args.trace_jsonl else None
    if trace:
        trace.set_meta(role="listener", experiment_id=exp_id)
    status = StatusStore(trace=trace) if args.ui else None
    if status:
        from app.cli.dashboard import run_dashboard

        status.set_experiment(experiment_id=exp_id, mode=args.mode)
        threading.Thread(target=run_dashboard, args=(status,), kwargs={"ui_mode": args.ui_mode}, daemon=True).start()

    run_listener_session(
        catalog_path=Path(args.catalog),
        audio_dir=Path(args.audio_dir),
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        port=args.port,
        baud=args.baud,
        model=args.model,
        thread_id=args.thread_id,
        experiment_id=exp_id,
        status=status,
        trace=trace,
        mode=args.mode,
        human_choice_count=args.human_choice_count,
        human_choice_ids=args.human_choice_ids,
        debug_imu=args.debug_imu,
        debug_agent=args.debug_agent,
        debug_signal=args.debug_signal,
        agent_interval_sec=args.agent_interval_sec,
        backchannel_cooldown_sec=args.backchannel_cooldown_sec,
        speaker_playback=(not args.no_speaker_playback),
        speaker_playback_bin=args.speaker_playback_bin,
        stt_model=args.stt_model,
        stt_language=args.stt_language,
        stt_prompt=args.stt_prompt,
        stt_segments_dir=Path(args.stt_segments_dir),
        vad_frame_ms=args.vad_frame_ms,
        vad_pre_roll_ms=args.vad_pre_roll_ms,
        vad_silence_end_ms=args.vad_silence_end_ms,
        vad_min_speech_ms=args.vad_min_speech_ms,
        vad_start_voice_frames=args.vad_start_voice_frames,
        vad_min_voice_ms=args.vad_min_voice_ms,
        vad_calib_sec=args.vad_calib_sec,
        vad_threshold_rms=args.vad_threshold_rms,
        vad_threshold_mult=args.vad_threshold_mult,
        vad_max_segment_ms=args.vad_max_segment_ms,
        send_backchannel_to_talker=(not args.no_send_backchannel_to_talker),
        local_backchannel_play=args.local_backchannel_play,
        calibration_still_sec=args.calibration_still_sec,
        calibration_active_sec=args.calibration_active_sec,
        calibration_start_delay_sec=args.calibration_start_delay_sec,
        calibration_between_sec=args.calibration_between_sec,
        calibration_wait_for_imu_sec=args.calibration_wait_for_imu_sec,
        human_signal_gyro_sigma=args.human_signal_gyro_sigma,
        human_signal_abs_threshold=args.human_signal_abs_threshold,
        human_signal_max_age_s=args.human_signal_max_age_s,
        human_signal_min_consecutive=args.human_signal_min_consecutive,
        human_signal_hold_sec=args.human_signal_hold_sec,
        imu_nod_axis=args.imu_nod_axis,
        imu_shake_axis=args.imu_shake_axis,
        gesture_calibration=args.gesture_calibration,
        gesture_weak_sec=args.gesture_weak_sec,
        gesture_strong_sec=args.gesture_strong_sec,
        gesture_start_delay_sec=args.gesture_start_delay_sec,
        gesture_rest_sec=args.gesture_rest_sec,
        auto_imu_axis_map=not args.no_auto_imu_axis_map,
        boundary_silence_ms=args.boundary_silence_ms,
        context_max_lines=args.context_max_lines,
        early_call_delay_sec=args.early_call_delay_sec,
    )


if __name__ == "__main__":
    main()
