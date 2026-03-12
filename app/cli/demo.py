#!/usr/bin/env python3
import argparse
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.demo import load_demo_script, run_demo_session
from app.imu.device import get_device_profile
from app.runtime.status import StatusStore
from app.runtime.trace import TraceWriter


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Conference demo app (sensor-only by default)")
    parser.add_argument("--device-id", default="demo-usbserial-six-axis")
    parser.add_argument("--device-config", default="data/demo/devices.json")
    parser.add_argument("--script", default="")
    parser.add_argument("--catalog", default="data/demo/catalog_en.tsv")
    parser.add_argument("--audio-dir", default="data/demo/backchannel_en")
    parser.add_argument("--port", default="")
    parser.add_argument("--baud", type=int, default=0)
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--thread-id", default="sensor-only-demo")
    parser.add_argument("--trace-jsonl", default="")
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--ui-mode", choices=["participant", "debug"], default="participant")
    parser.add_argument("--interactive-cues", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cue-pause-sec", type=float, default=0.2)
    parser.add_argument("--debug-imu", action="store_true")
    parser.add_argument("--debug-agent", action="store_true")
    parser.add_argument("--debug-signal", action="store_true")
    parser.add_argument("--agent-interval-sec", type=float, default=0.2)
    parser.add_argument("--backchannel-cooldown-sec", type=float, default=0.6)
    parser.add_argument("--gesture-calibration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gesture-weak-sec", type=float, default=2.0)
    parser.add_argument("--gesture-strong-sec", type=float, default=2.0)
    parser.add_argument("--gesture-start-delay-sec", type=float, default=1.0)
    parser.add_argument("--gesture-rest-sec", type=float, default=1.0)
    parser.add_argument("--human-signal-abs-threshold", type=float, default=8.0)
    parser.add_argument("--human-signal-max-age-s", type=float, default=1.5)
    parser.add_argument("--human-signal-min-consecutive", type=int, default=3)
    parser.add_argument("--imu-nod-axis", choices=["gx", "gy", "gz"], default="gy")
    parser.add_argument("--imu-shake-axis", choices=["gx", "gy", "gz"], default="gz")
    args = parser.parse_args()

    trace = TraceWriter(Path(args.trace_jsonl)) if args.trace_jsonl else None
    status = StatusStore(trace=trace) if args.ui else None
    if status:
        from app.cli.dashboard import run_dashboard

        threading.Thread(
            target=run_dashboard,
            args=(status,),
            kwargs={"ui_mode": args.ui_mode},
            daemon=True,
        ).start()

    profile = get_device_profile(Path(args.device_config), args.device_id)
    script = None
    script_path = None
    if str(args.script or "").strip():
        script_path = Path(args.script)
        script = load_demo_script(script_path)
    if trace:
        trace.set_meta(
            role="demo",
            experiment_id=(script.script_id if script is not None else "sensor_only_en"),
            device_id=profile.device_id,
        )

    try:
        run_demo_session(
            script=script,
            script_path=script_path,
            catalog_path=Path(args.catalog),
            audio_dir=Path(args.audio_dir),
            device_profile=profile,
            port=str(args.port or "").strip(),
            baud=(int(args.baud) if int(args.baud) > 0 else None),
            model=args.model,
            thread_id=args.thread_id,
            status=status,
            trace=trace,
            debug_imu=args.debug_imu,
            debug_agent=args.debug_agent,
            debug_signal=args.debug_signal,
            agent_interval_sec=args.agent_interval_sec,
            backchannel_cooldown_sec=args.backchannel_cooldown_sec,
            gesture_calibration=args.gesture_calibration,
            gesture_weak_sec=args.gesture_weak_sec,
            gesture_strong_sec=args.gesture_strong_sec,
            gesture_start_delay_sec=args.gesture_start_delay_sec,
            gesture_rest_sec=args.gesture_rest_sec,
            human_signal_abs_threshold=args.human_signal_abs_threshold,
            human_signal_max_age_s=args.human_signal_max_age_s,
            human_signal_min_consecutive=args.human_signal_min_consecutive,
            imu_nod_axis=args.imu_nod_axis,
            imu_shake_axis=args.imu_shake_axis,
            interactive_cues=args.interactive_cues,
            cue_pause_sec=args.cue_pause_sec,
        )
    except KeyboardInterrupt:
        print("停止しました。")


if __name__ == "__main__":
    main()
