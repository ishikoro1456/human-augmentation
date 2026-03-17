#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.imu.device import get_device_profile, probe_serial_format, resolve_serial_port


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Probe IMU serial format")
    parser.add_argument("--port", default="")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--device-id", default="")
    parser.add_argument("--device-config", default="data/demo/devices.json")
    args = parser.parse_args()

    port = str(args.port or "").strip()
    if args.device_id:
        profile = get_device_profile(Path(args.device_config), args.device_id)
        port = resolve_serial_port(profile, override_port=port)
        baud = int(args.baud or profile.baud)
    else:
        if not port:
            parser.error("--port か --device-id のどちらかが必要です")
        baud = int(args.baud)

    print(f"probe start: port={port} baud={baud} seconds={args.seconds:g}")
    report = probe_serial_format(port=port, baud=baud, seconds=args.seconds)
    print(f"detected_format: {report.detected_format}")
    print(f"sample_count: {report.sample_count}")
    print(f"mean_norm: {report.mean_norm}")
    print(f"stdev_norm: {report.stdev_norm}")
    print(f"reason: {report.reason}")
    if report.preview:
        print("numeric_preview:")
        for sample in report.preview:
            print(f"  {sample}")


if __name__ == "__main__":
    main()
