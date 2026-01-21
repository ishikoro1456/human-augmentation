import math
import re
from typing import Optional, Tuple


ImuTuple = Tuple[float, float, float, float, float, float]


def parse_imu_line(line: str) -> Optional[ImuTuple]:
    # まずはラベル付き（ax=..., gx=... など）を優先して読む
    float_pat = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
    labeled: dict[str, float] = {}
    for k in ("ax", "ay", "az", "gx", "gy", "gz"):
        m = re.search(rf"\b{k}\b\s*[:=]\s*({float_pat})", line, flags=re.IGNORECASE)
        if m:
            try:
                labeled[k] = float(m.group(1))
            except Exception:
                pass
    if len(labeled) == 6:
        return labeled["ax"], labeled["ay"], labeled["az"], labeled["gx"], labeled["gy"], labeled["gz"]

    # ラベルが無い場合は数値だけ抜き出す（先頭にタイムスタンプ等があっても耐える）
    values: list[float] = []
    for m in re.finditer(float_pat, line):
        try:
            values.append(float(m.group(0)))
        except Exception:
            continue
    if len(values) < 6:
        return None
    if len(values) == 6:
        ax, ay, az, gx, gy, gz = values
        return ax, ay, az, gx, gy, gz

    # 6つずつの候補から、「加速度の大きさが 9.81（m/s^2）か 1.0（g）に近い」ものを選ぶ
    best: Optional[ImuTuple] = None
    best_score: float = float("inf")
    for i in range(len(values) - 5):
        ax, ay, az, gx, gy, gz = values[i : i + 6]
        acc_mag = math.sqrt(float(ax) ** 2 + float(ay) ** 2 + float(az) ** 2)
        score = min(abs(acc_mag - 9.81), abs(acc_mag - 1.0))
        if score < best_score:
            best_score = score
            best = (float(ax), float(ay), float(az), float(gx), float(gy), float(gz))

    # どう見ても加速度っぽい並びが無いなら、末尾6つを使う（先頭に余計な数値が乗るケースを吸収）
    if best is None or best_score > 5.0:
        ax, ay, az, gx, gy, gz = values[-6:]
        return float(ax), float(ay), float(az), float(gx), float(gy), float(gz)
    return best
