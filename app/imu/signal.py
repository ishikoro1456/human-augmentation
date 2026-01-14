from __future__ import annotations

import math
from typing import Dict, Optional

from app.imu.calibration import ImuCalibration


def _sign(v: float, deadband: float) -> int:
    if v > deadband:
        return 1
    if v < -deadband:
        return -1
    return 0


def _sign_changes(values: list[float], *, deadband: float = 0.3) -> int:
    prev = 0
    changes = 0
    for v in values:
        s = _sign(float(v), deadband)
        if s == 0:
            continue
        if prev == 0:
            prev = s
            continue
        if s != prev:
            changes += 1
            prev = s
    return int(changes)


def detect_backchannel_signal(
    imu_bundle: Dict[str, object],
    *,
    calibration: Optional[ImuCalibration],
    gyro_sigma: float = 3.0,
    abs_threshold: float = 8.0,
    max_age_s: float = 1.5,
    min_consecutive_above: int = 3,
    nod_axis: str = "gy",
    shake_axis: str = "gz",
    tilt_axis: str = "gx",
) -> Dict[str, object]:
    age = imu_bundle.get("last_sample_age_s")
    if not isinstance(age, (int, float)):
        return {"present": False, "reason": "サインなし: IMUが未受信です。"}
    age_f = float(age)
    if age_f > float(max_age_s):
        return {
            "present": False,
            "reason": f"サインなし: IMUが古いです(age={age_f:.2f}s)。",
            "last_sample_age_s": round(age_f, 3),
        }

    activity = imu_bundle.get("activity_1s", {})
    if not isinstance(activity, dict):
        return {"present": False, "reason": "サインなし: IMUの要約がありません。"}

    gyro_max = activity.get("gyro_mag_max")
    if not isinstance(gyro_max, (int, float)):
        return {"present": False, "reason": "サインなし: gyro_mag_maxがありません。"}
    gyro_max_f = float(gyro_max)

    threshold = float(abs_threshold)
    threshold_base = "abs"
    if calibration and calibration.still:
        mean = calibration.still.gyro_mag.get("mean")
        stdev = calibration.still.gyro_mag.get("stdev")
        if isinstance(mean, (int, float)) and isinstance(stdev, (int, float)) and float(stdev) > 0:
            threshold = float(mean) + float(gyro_sigma) * float(stdev)
            threshold_base = f"still_mean+{float(gyro_sigma):g}σ"
        elif isinstance(mean, (int, float)):
            threshold = float(mean) * 2.0
            threshold_base = "still_mean*2"

    raw_samples = imu_bundle.get("raw_samples", [])
    mags_1s = []
    axis_abs_sum = {"gx": 0.0, "gy": 0.0, "gz": 0.0}
    axis_count = {"gx": 0, "gy": 0, "gz": 0}
    axis_values_1s: Dict[str, list[float]] = {"gx": [], "gy": [], "gz": []}
    if isinstance(raw_samples, list):
        for s in raw_samples:
            if not isinstance(s, dict):
                continue
            t_rel = s.get("t_rel_s")
            if not isinstance(t_rel, (int, float)):
                continue
            if float(t_rel) < -1.0:
                continue
            gx = s.get("gx")
            gy = s.get("gy")
            gz = s.get("gz")
            if not isinstance(gx, (int, float)) or not isinstance(gy, (int, float)) or not isinstance(
                gz, (int, float)
            ):
                continue
            gx_f = float(gx)
            gy_f = float(gy)
            gz_f = float(gz)
            mags_1s.append(math.sqrt(gx_f * gx_f + gy_f * gy_f + gz_f * gz_f))
            axis_abs_sum["gx"] += abs(gx_f)
            axis_abs_sum["gy"] += abs(gy_f)
            axis_abs_sum["gz"] += abs(gz_f)
            axis_count["gx"] += 1
            axis_count["gy"] += 1
            axis_count["gz"] += 1
            axis_values_1s["gx"].append(gx_f)
            axis_values_1s["gy"].append(gy_f)
            axis_values_1s["gz"].append(gz_f)

    above = [m >= threshold for m in mags_1s] if mags_1s else []
    run = 0
    run_max = 0
    for a in above:
        if a:
            run += 1
            if run > run_max:
                run_max = run
        else:
            run = 0
    above_count = sum(1 for a in above if a)
    present = bool(run_max >= int(min_consecutive_above)) if above else (gyro_max_f >= threshold)
    label = "サインあり" if present else "サインなし"
    dominant_axis = "-"
    axis_abs_mean: Dict[str, float] = {}
    if axis_count["gx"] > 0:
        axis_abs_mean = {
            "gx": axis_abs_sum["gx"] / axis_count["gx"],
            "gy": axis_abs_sum["gy"] / axis_count["gy"],
            "gz": axis_abs_sum["gz"] / axis_count["gz"],
        }
        dominant_axis = max(axis_abs_mean, key=lambda k: axis_abs_mean[k])

    axis_map = {"nod": nod_axis, "shake": shake_axis, "tilt": tilt_axis}
    axis_sign_changes = {
        "gx": _sign_changes(axis_values_1s["gx"]),
        "gy": _sign_changes(axis_values_1s["gy"]),
        "gz": _sign_changes(axis_values_1s["gz"]),
    }
    dominant_sc = axis_sign_changes.get(dominant_axis, 0) if dominant_axis in axis_sign_changes else 0

    gesture_hint = "other"
    if dominant_axis == nod_axis and dominant_sc >= 1:
        gesture_hint = "nod"
    elif dominant_axis == shake_axis and dominant_sc >= 1:
        gesture_hint = "shake"
    elif dominant_axis == tilt_axis and dominant_sc >= 1:
        gesture_hint = "tilt"

    cmp = ">=" if present else "<"
    reason = (
        f"{label}: run={run_max}/{max(1, int(min_consecutive_above))}, "
        f"gyro_mag_max_1s={gyro_max_f:.3f} {cmp} threshold={threshold:.3f} ({threshold_base}), "
        f"axis={dominant_axis}, sc={dominant_sc}, hint={gesture_hint}"
    )
    return {
        "present": present,
        "reason": reason,
        "last_sample_age_s": round(age_f, 3),
        "gyro_mag_max_1s": round(gyro_max_f, 3),
        "threshold": round(threshold, 3),
        "threshold_base": threshold_base,
        "run_max": int(run_max),
        "above_count": int(above_count),
        "min_consecutive_above": int(min_consecutive_above),
        "dominant_axis": dominant_axis,
        "axis_abs_mean_1s": {k: round(v, 3) for k, v in axis_abs_mean.items()} if axis_abs_mean else {},
        "axis_sign_changes_1s": dict(axis_sign_changes),
        "axis_map": axis_map,
        "gesture_hint": gesture_hint,
    }
