import time
from typing import Iterator

import serial
from serial import SerialException

from .parser import parse_imu_line, ImuTuple


def read_imu_lines(port: str, baud: int) -> Iterator[ImuTuple]:
    while True:
        try:
            ser = serial.Serial(port, baud, timeout=1)
        except (FileNotFoundError, SerialException):
            print(f"IMUポートが見つかりません: {port}")
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
            print("IMUの接続が切れました。再接続を待ちます。")
            time.sleep(2)
        finally:
            try:
                ser.close()
            except Exception:
                pass
