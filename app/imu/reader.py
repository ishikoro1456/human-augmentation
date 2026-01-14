import time
from typing import Callable, Iterator

import serial
from serial import SerialException

from .parser import parse_imu_line, ImuTuple


def read_imu_lines(
    port: str,
    baud: int,
    *,
    on_log: Callable[[str], None] | None = None,
) -> Iterator[ImuTuple]:
    def _log(message: str) -> None:
        if on_log:
            on_log(message)
        else:
            print(message)

    while True:
        try:
            ser = serial.Serial(port, baud, timeout=1)
        except (FileNotFoundError, SerialException):
            _log(f"IMUポートが見つかりません: {port}")
            time.sleep(2)
            continue

        try:
            while True:
                line = ser.readline().decode(errors="ignore").strip()
                parsed = parse_imu_line(line)
                if parsed is None:
                    continue
                yield parsed
                time.sleep(0)
        except SerialException:
            _log("IMUの接続が切れました。再接続を待ちます。")
            time.sleep(2)
        finally:
            try:
                ser.close()
            except Exception:
                pass
