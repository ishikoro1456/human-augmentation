#!/usr/bin/env python3
import argparse
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.runtime.session import run_session
from app.runtime.status import StatusStore


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Local backchannel agent")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument(
        "--catalog",
        default="data/catalog.tsv",
        help="Path to catalog.tsv",
    )
    parser.add_argument(
        "--audio-dir",
        default="data/backchannel",
        help="Directory that holds audio files",
    )
    parser.add_argument("--port", default="/dev/cu.usbserial-140")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--thread-id", default="local-session")
    parser.add_argument(
        "--transcript",
        default="transcribe.txt",
        help="Path to transcript with [mm:ss] lines",
    )
    parser.add_argument(
        "--transcript-start-sec",
        type=int,
        default=0,
        help="Start position for transcript time in seconds",
    )
    parser.add_argument(
        "--tts-model",
        default="gpt-4o-mini-tts",
        help="OpenAI TTS model for transcript playback",
    )
    parser.add_argument(
        "--tts-voice",
        default="alloy",
        help="OpenAI TTS voice",
    )
    parser.add_argument(
        "--tts-format",
        default="mp3",
        help="Audio format for transcript playback",
    )
    parser.add_argument(
        "--tts-cache-dir",
        default="data/tts_cache",
        help="Directory to cache generated TTS audio",
    )
    parser.add_argument(
        "--debug-imu",
        action="store_true",
        help="Print IMU values once per second",
    )
    parser.add_argument(
        "--debug-transcript",
        action="store_true",
        help="Print transcript context used by the agent",
    )
    parser.add_argument(
        "--debug-agent",
        action="store_true",
        help="Print IMU and agent reason",
    )
    parser.add_argument(
        "--debug-signal",
        action="store_true",
        help="Print IMU backchannel signal decision at each transcript boundary",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Show a live dashboard in the terminal",
    )
    parser.add_argument(
        "--agent-interval-sec",
        type=float,
        default=1.0,
        help="Minimum seconds between agent calls",
    )
    parser.add_argument(
        "--backchannel-cooldown-sec",
        type=float,
        default=2.0,
        help="Minimum seconds between played backchannels",
    )
    parser.add_argument(
        "--calibration-still-sec",
        type=float,
        default=10.0,
        help="Seconds to stay still for IMU calibration (0 to skip)",
    )
    parser.add_argument(
        "--calibration-start-delay-sec",
        type=float,
        default=3.0,
        help="Seconds to wait before calibration starts",
    )
    parser.add_argument(
        "--calibration-active-sec",
        type=float,
        default=20.0,
        help="Seconds to move naturally for IMU calibration (0 to skip)",
    )
    parser.add_argument(
        "--calibration-between-sec",
        type=float,
        default=3.0,
        help="Seconds to rest between still and active calibration",
    )
    parser.add_argument(
        "--calibration-wait-for-imu-sec",
        type=float,
        default=15.0,
        help="Max seconds to wait for IMU before skipping calibration",
    )
    parser.add_argument(
        "--gesture-calibration",
        action="store_true",
        help="Run a short guided gesture calibration (weak/strong nod and shake) at startup",
    )
    parser.add_argument(
        "--gesture-weak-sec",
        type=float,
        default=2.0,
        help="Seconds for weak nod/shake during gesture calibration",
    )
    parser.add_argument(
        "--gesture-strong-sec",
        type=float,
        default=2.0,
        help="Seconds for strong nod/shake during gesture calibration",
    )
    parser.add_argument(
        "--gesture-start-delay-sec",
        type=float,
        default=2.0,
        help="Seconds to wait before each gesture capture starts",
    )
    parser.add_argument(
        "--gesture-rest-sec",
        type=float,
        default=2.0,
        help="Seconds to stay still between gesture captures",
    )
    parser.add_argument(
        "--startup-wait-sec",
        type=float,
        default=2.0,
        help="Seconds to wait after calibration before transcript playback starts",
    )
    parser.add_argument(
        "--no-auto-imu-axis-map",
        action="store_true",
        help="Do not overwrite imu axis mapping based on gesture calibration results",
    )
    parser.add_argument(
        "--no-require-human-signal",
        action="store_true",
        help="Allow the agent to choose backchannels even when no clear IMU signal is detected",
    )
    parser.add_argument(
        "--human-signal-gyro-sigma",
        type=float,
        default=3.0,
        help="Threshold in sigma above still baseline for detecting a human backchannel signal",
    )
    parser.add_argument(
        "--human-signal-abs-threshold",
        type=float,
        default=8.0,
        help="Fallback absolute gyro magnitude threshold when calibration is unavailable",
    )
    parser.add_argument(
        "--human-signal-max-age-s",
        type=float,
        default=1.5,
        help="Max age of latest IMU sample to consider for signal detection",
    )
    parser.add_argument(
        "--human-signal-min-consecutive",
        type=int,
        default=3,
        help="Minimum consecutive IMU samples above threshold to count as a human signal",
    )
    parser.add_argument(
        "--human-signal-hold-sec",
        type=float,
        default=3.0,
        help="Max seconds after a detected human signal to still react (for timing jitter and cooldown)",
    )
    parser.add_argument(
        "--imu-nod-axis",
        choices=["gx", "gy", "gz"],
        default="gy",
        help="Which gyro axis represents a nodding-like motion for this sensor mounting",
    )
    parser.add_argument(
        "--imu-shake-axis",
        choices=["gx", "gy", "gz"],
        default="gz",
        help="Which gyro axis represents a shaking-like motion for this sensor mounting",
    )
    parser.add_argument(
        "--imu-tilt-axis",
        choices=["gx", "gy", "gz"],
        default="gx",
        help="Which gyro axis represents a head-tilt-like motion for this sensor mounting",
    )
    args = parser.parse_args()

    status = StatusStore() if args.ui else None
    if status:
        from app.cli.dashboard import run_dashboard

        threading.Thread(target=run_dashboard, args=(status,), daemon=True).start()

    run_session(
        catalog_path=Path(args.catalog),
        audio_dir=Path(args.audio_dir),
        port=args.port,
        baud=args.baud,
        model=args.model,
        thread_id=args.thread_id,
        status=status,
        debug_imu=args.debug_imu,
        transcript_path=Path(args.transcript),
        transcript_start_sec=args.transcript_start_sec,
        debug_transcript=args.debug_transcript,
        debug_agent=args.debug_agent,
        debug_signal=args.debug_signal,
        tts_model=args.tts_model,
        tts_voice=args.tts_voice,
        tts_format=args.tts_format,
        tts_cache_dir=Path(args.tts_cache_dir),
        agent_interval_sec=args.agent_interval_sec,
        backchannel_cooldown_sec=args.backchannel_cooldown_sec,
        calibration_still_sec=args.calibration_still_sec,
        calibration_active_sec=args.calibration_active_sec,
        calibration_start_delay_sec=args.calibration_start_delay_sec,
        calibration_between_sec=args.calibration_between_sec,
        calibration_wait_for_imu_sec=args.calibration_wait_for_imu_sec,
        require_human_signal=not args.no_require_human_signal,
        human_signal_gyro_sigma=args.human_signal_gyro_sigma,
        human_signal_abs_threshold=args.human_signal_abs_threshold,
        human_signal_max_age_s=args.human_signal_max_age_s,
        human_signal_min_consecutive=args.human_signal_min_consecutive,
        human_signal_hold_sec=args.human_signal_hold_sec,
        imu_nod_axis=args.imu_nod_axis,
        imu_shake_axis=args.imu_shake_axis,
        imu_tilt_axis=args.imu_tilt_axis,
        gesture_calibration=args.gesture_calibration,
        gesture_weak_sec=args.gesture_weak_sec,
        gesture_strong_sec=args.gesture_strong_sec,
        gesture_start_delay_sec=args.gesture_start_delay_sec,
        gesture_rest_sec=args.gesture_rest_sec,
        auto_imu_axis_map=not args.no_auto_imu_axis_map,
        startup_wait_sec=args.startup_wait_sec,
    )


if __name__ == "__main__":
    main()
