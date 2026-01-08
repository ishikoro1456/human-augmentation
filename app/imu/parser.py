import re
from typing import Optional, Tuple


ImuTuple = Tuple[float, float, float, float, float, float]


def parse_imu_line(line: str) -> Optional[ImuTuple]:
    values = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    if len(values) < 6:
        return None
    ax, ay, az, gx, gy, gz = map(float, values[:6])
    return ax, ay, az, gx, gy, gz
