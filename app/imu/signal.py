from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional, Tuple

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


def _compute_motion_duration(
    raw_samples: List[Dict[str, object]],
    threshold: float,
) -> Tuple[float, float, float]:
    """閾値を超えた動きの持続時間を計算する。"""
    above_times: List[float] = []
    for s in raw_samples:
        if not isinstance(s, dict):
            continue
        t_rel = s.get("t_rel_s")
        gx = s.get("gx")
        gy = s.get("gy")
        gz = s.get("gz")
        if not isinstance(t_rel, (int, float)):
            continue
        if not all(isinstance(v, (int, float)) for v in [gx, gy, gz]):
            continue
        mag = math.sqrt(float(gx) ** 2 + float(gy) ** 2 + float(gz) ** 2)
        if mag >= threshold:
            above_times.append(float(t_rel))

    if not above_times:
        return 0.0, 0.0, 0.0

    first_t = min(above_times)
    last_t = max(above_times)
    duration = last_t - first_t
    return duration, first_t, last_t


def _compute_posture_change(
    raw_samples: List[Dict[str, object]],
    axis: str,
    window_sec: float = 1.0,
) -> Dict[str, object]:
    """姿勢の変化を計算する（ジャイロ積分）。"""
    samples_in_window = []
    for s in raw_samples:
        if not isinstance(s, dict):
            continue
        t_rel = s.get("t_rel_s")
        val = s.get(axis)
        if not isinstance(t_rel, (int, float)) or not isinstance(val, (int, float)):
            continue
        if float(t_rel) >= -window_sec:
            samples_in_window.append((float(t_rel), float(val)))

    if len(samples_in_window) < 4:
        return {
            "integrated_change": 0.0,
            "posture_returned": True,
            "start_avg": 0.0,
            "end_avg": 0.0,
        }

    samples_in_window.sort(key=lambda x: x[0])

    integrated = 0.0
    prev_t = samples_in_window[0][0]
    for t, val in samples_in_window[1:]:
        dt = t - prev_t
        if dt > 0:
            integrated += val * dt
        prev_t = t

    n = len(samples_in_window)
    quarter = max(1, n // 4)
    start_vals = [v for _, v in samples_in_window[:quarter]]
    end_vals = [v for _, v in samples_in_window[-quarter:]]
    start_avg = sum(start_vals) / len(start_vals) if start_vals else 0.0
    end_avg = sum(end_vals) / len(end_vals) if end_vals else 0.0

    posture_returned = abs(integrated) < 5.0 and abs(start_avg - end_avg) < 2.0

    return {
        "integrated_change": round(integrated, 3),
        "posture_returned": posture_returned,
        "start_avg": round(start_avg, 3),
        "end_avg": round(end_avg, 3),
    }


def _compute_oscillation_symmetry(values: List[float]) -> Dict[str, object]:
    """振幅の対称性を計算する。"""
    if not values:
        return {
            "symmetry_ratio": 1.0,
            "is_symmetric": True,
            "positive_sum": 0.0,
            "negative_sum": 0.0,
        }

    positive_sum = sum(v for v in values if v > 0)
    negative_sum = sum(abs(v) for v in values if v < 0)

    total = positive_sum + negative_sum
    if total < 0.1:
        return {
            "symmetry_ratio": 1.0,
            "is_symmetric": True,
            "positive_sum": 0.0,
            "negative_sum": 0.0,
        }

    if positive_sum > negative_sum:
        ratio = negative_sum / positive_sum if positive_sum > 0 else 0.0
    else:
        ratio = positive_sum / negative_sum if negative_sum > 0 else 0.0

    is_symmetric = ratio >= 0.5

    return {
        "symmetry_ratio": round(ratio, 3),
        "is_symmetric": is_symmetric,
        "positive_sum": round(positive_sum, 3),
        "negative_sum": round(negative_sum, 3),
    }


def _compute_baseline_comparison(imu_bundle: Dict[str, object]) -> Dict[str, object]:
    """長期ベースライン（5秒、30秒）と比較する。"""
    stats = imu_bundle.get("stats", {})
    activity_1s = imu_bundle.get("activity_1s", {})

    if not isinstance(stats, dict) or not isinstance(activity_1s, dict):
        return {"ratio_vs_5s": None, "ratio_vs_30s": None}

    gyro_mag_max_1s = activity_1s.get("gyro_mag_max")
    if not isinstance(gyro_mag_max_1s, (int, float)):
        return {"ratio_vs_5s": None, "ratio_vs_30s": None}

    result: Dict[str, object] = {}

    stats_5 = stats.get("5", {})
    if isinstance(stats_5, dict):
        gx_5 = stats_5.get("gx", {})
        gy_5 = stats_5.get("gy", {})
        gz_5 = stats_5.get("gz", {})
        if all(isinstance(x, dict) for x in [gx_5, gy_5, gz_5]):
            max_5s = max(
                abs(gx_5.get("max", 0) or 0),
                abs(gy_5.get("max", 0) or 0),
                abs(gz_5.get("max", 0) or 0),
                abs(gx_5.get("min", 0) or 0),
                abs(gy_5.get("min", 0) or 0),
                abs(gz_5.get("min", 0) or 0),
            )
            if max_5s > 0.1:
                result["ratio_vs_5s"] = round(float(gyro_mag_max_1s) / max_5s, 3)

    stats_30 = stats.get("30", {})
    if isinstance(stats_30, dict):
        gx_30 = stats_30.get("gx", {})
        gy_30 = stats_30.get("gy", {})
        gz_30 = stats_30.get("gz", {})
        if all(isinstance(x, dict) for x in [gx_30, gy_30, gz_30]):
            max_30s = max(
                abs(gx_30.get("max", 0) or 0),
                abs(gy_30.get("max", 0) or 0),
                abs(gz_30.get("max", 0) or 0),
                abs(gx_30.get("min", 0) or 0),
                abs(gy_30.get("min", 0) or 0),
                abs(gz_30.get("min", 0) or 0),
            )
            if max_30s > 0.1:
                result["ratio_vs_30s"] = round(float(gyro_mag_max_1s) / max_30s, 3)

    return result


def _vec_norm(v: Tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _cosine(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    na = _vec_norm(a)
    nb = _vec_norm(b)
    if na <= 1e-6 or nb <= 1e-6:
        return 0.0
    return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb)


def _compute_acc_features(raw_samples: List[Dict[str, object]]) -> Dict[str, float]:
    return _compute_acc_features_with_flag(raw_samples, acc_available=True)


def _compute_acc_features_with_flag(
    raw_samples: List[Dict[str, object]],
    *,
    acc_available: bool,
) -> Dict[str, float]:
    if not acc_available:
        return {
            "acc_delta_mag_1s": 0.0,
            "acc_axis_stability": 0.0,
            "tilt_return_score": 0.0,
            "acc_available": False,
        }

    acc_1s: List[Tuple[float, float, float]] = []
    acc_2s: List[Tuple[float, float, float]] = []

    for s in raw_samples:
        if not isinstance(s, dict):
            continue
        t_rel = s.get("t_rel_s")
        ax = s.get("ax")
        ay = s.get("ay")
        az = s.get("az")
        if not isinstance(t_rel, (int, float)):
            continue
        if not all(isinstance(v, (int, float)) for v in (ax, ay, az)):
            continue
        vec = (float(ax), float(ay), float(az))
        if float(t_rel) >= -2.0:
            acc_2s.append(vec)
        if float(t_rel) >= -1.0:
            acc_1s.append(vec)

    if len(acc_1s) < 2:
        return {
            "acc_delta_mag_1s": 0.0,
            "acc_axis_stability": 0.0,
            "tilt_return_score": 0.0,
            "acc_available": True,
        }

    delta_mags: List[float] = []
    for prev, cur in zip(acc_1s, acc_1s[1:]):
        dx = cur[0] - prev[0]
        dy = cur[1] - prev[1]
        dz = cur[2] - prev[2]
        delta_mags.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    acc_delta_mag = float(statistics.mean(delta_mags)) if delta_mags else 0.0

    ax_vals = [v[0] for v in acc_1s]
    ay_vals = [v[1] for v in acc_1s]
    az_vals = [v[2] for v in acc_1s]
    stdev_mean = (
        (statistics.pstdev(ax_vals) if len(ax_vals) > 1 else 0.0)
        + (statistics.pstdev(ay_vals) if len(ay_vals) > 1 else 0.0)
        + (statistics.pstdev(az_vals) if len(az_vals) > 1 else 0.0)
    ) / 3.0
    # 0=不安定, 1=安定
    axis_stability = 1.0 - min(1.0, stdev_mean / 1.2)

    tilt_return_score = 0.0
    if len(acc_2s) >= 6:
        n = len(acc_2s)
        k = max(2, n // 3)
        start = acc_2s[:k]
        end = acc_2s[-k:]
        start_mean = (
            float(statistics.mean([v[0] for v in start])),
            float(statistics.mean([v[1] for v in start])),
            float(statistics.mean([v[2] for v in start])),
        )
        end_mean = (
            float(statistics.mean([v[0] for v in end])),
            float(statistics.mean([v[1] for v in end])),
            float(statistics.mean([v[2] for v in end])),
        )
        # cosine -1..1 を 0..1 へ
        tilt_return_score = max(0.0, min(1.0, (_cosine(start_mean, end_mean) + 1.0) / 2.0))

    return {
        "acc_delta_mag_1s": round(acc_delta_mag, 3),
        "acc_axis_stability": round(axis_stability, 3),
        "tilt_return_score": round(tilt_return_score, 3),
        "acc_available": True,
    }


def _estimate_signal_confidence(
    *,
    present: bool,
    run_max: int,
    min_consecutive_above: int,
    nod_likelihood_score: int,
    acc_axis_stability: float,
    tilt_return_score: float,
    acc_available: bool,
) -> float:
    if min_consecutive_above <= 0:
        min_consecutive_above = 1
    run_score = max(0.0, min(1.0, float(run_max) / float(min_consecutive_above)))
    nod_score = max(0.0, min(1.0, float(nod_likelihood_score) / 6.0))
    st_score = max(0.0, min(1.0, float(acc_axis_stability)))
    tilt_score = max(0.0, min(1.0, float(tilt_return_score)))

    if acc_available:
        confidence = (run_score * 0.4) + (nod_score * 0.35) + (st_score * 0.15) + (tilt_score * 0.10)
    else:
        confidence = (run_score * 0.55) + (nod_score * 0.45)
    if not present:
        confidence *= 0.4
    return round(max(0.0, min(1.0, confidence)), 3)


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
    if not isinstance(raw_samples, list):
        raw_samples = []
    sensor_flags = imu_bundle.get("sensor_flags", {})
    acc_available = True
    if isinstance(sensor_flags, dict) and "acc_available" in sensor_flags:
        acc_available = bool(sensor_flags.get("acc_available"))

    mags_1s = []
    axis_abs_sum = {"gx": 0.0, "gy": 0.0, "gz": 0.0}
    axis_count = {"gx": 0, "gy": 0, "gz": 0}
    axis_values_1s: Dict[str, list[float]] = {"gx": [], "gy": [], "gz": []}
    axis_values_2s: Dict[str, list[float]] = {"gx": [], "gy": [], "gz": []}

    for s in raw_samples:
        if not isinstance(s, dict):
            continue
        t_rel = s.get("t_rel_s")
        if not isinstance(t_rel, (int, float)):
            continue
        gx = s.get("gx")
        gy = s.get("gy")
        gz = s.get("gz")
        if not isinstance(gx, (int, float)) or not isinstance(gy, (int, float)) or not isinstance(gz, (int, float)):
            continue
        gx_f = float(gx)
        gy_f = float(gy)
        gz_f = float(gz)
        mag = math.sqrt(gx_f * gx_f + gy_f * gy_f + gz_f * gz_f)

        if float(t_rel) >= -2.0:
            axis_values_2s["gx"].append(gx_f)
            axis_values_2s["gy"].append(gy_f)
            axis_values_2s["gz"].append(gz_f)

        if float(t_rel) >= -1.0:
            mags_1s.append(mag)
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

    dominant_axis = "-"
    axis_abs_mean: Dict[str, float] = {}
    if axis_count["gx"] > 0:
        axis_abs_mean = {
            "gx": axis_abs_sum["gx"] / axis_count["gx"],
            "gy": axis_abs_sum["gy"] / axis_count["gy"],
            "gz": axis_abs_sum["gz"] / axis_count["gz"],
        }
        dominant_axis = max(axis_abs_mean, key=lambda k: axis_abs_mean[k])

    axis_map = {"nod": nod_axis, "shake": shake_axis}
    axis_sign_changes = {
        "gx": _sign_changes(axis_values_1s["gx"]),
        "gy": _sign_changes(axis_values_1s["gy"]),
        "gz": _sign_changes(axis_values_1s["gz"]),
    }
    axis_sign_changes_2s = {
        "gx": _sign_changes(axis_values_2s["gx"]),
        "gy": _sign_changes(axis_values_2s["gy"]),
        "gz": _sign_changes(axis_values_2s["gz"]),
    }
    dominant_sc = axis_sign_changes.get(dominant_axis, 0) if dominant_axis in axis_sign_changes else 0
    dominant_sc_2s = axis_sign_changes_2s.get(dominant_axis, 0) if dominant_axis in axis_sign_changes_2s else 0

    motion_duration, _, _ = _compute_motion_duration(raw_samples, threshold)
    posture_change = _compute_posture_change(raw_samples, dominant_axis, window_sec=2.0)
    oscillation_symmetry = _compute_oscillation_symmetry(axis_values_1s.get(dominant_axis, []))
    baseline_comparison = _compute_baseline_comparison(imu_bundle)
    acc_features = _compute_acc_features_with_flag(raw_samples, acc_available=acc_available)

    motion_features: Dict[str, object] = {
        "duration_s": round(motion_duration, 3),
        "sign_changes_1s": dominant_sc,
        "sign_changes_2s": dominant_sc_2s,
        "has_oscillation": dominant_sc >= 1,
        "posture_returned": posture_change.get("posture_returned", True),
        "integrated_change": posture_change.get("integrated_change", 0.0),
        "symmetry_ratio": oscillation_symmetry.get("symmetry_ratio", 1.0),
        "is_symmetric": oscillation_symmetry.get("is_symmetric", True),
        "ratio_vs_5s": baseline_comparison.get("ratio_vs_5s"),
        "ratio_vs_30s": baseline_comparison.get("ratio_vs_30s"),
        "acc_delta_mag_1s": acc_features["acc_delta_mag_1s"],
        "acc_axis_stability": acc_features["acc_axis_stability"],
        "tilt_return_score": acc_features["tilt_return_score"],
        "acc_available": bool(acc_features.get("acc_available", acc_available)),
    }

    nod_likelihood_score = 0
    if motion_features["has_oscillation"]:
        nod_likelihood_score += 2
    if motion_features["posture_returned"]:
        nod_likelihood_score += 2
    if motion_features["is_symmetric"]:
        nod_likelihood_score += 1
    if 0.3 <= motion_duration <= 2.0:
        nod_likelihood_score += 1
    motion_features["nod_likelihood_score"] = nod_likelihood_score

    enough_sc = (dominant_sc >= 2) or (dominant_sc >= 1 and dominant_sc_2s >= 2)
    gesture_hint = "other"
    if present and nod_likelihood_score >= 4 and enough_sc:
        if dominant_axis == nod_axis:
            gesture_hint = "nod"
        elif dominant_axis == shake_axis:
            gesture_hint = "shake"

    signal_confidence = _estimate_signal_confidence(
        present=present,
        run_max=run_max,
        min_consecutive_above=min_consecutive_above,
        nod_likelihood_score=nod_likelihood_score,
        acc_axis_stability=float(acc_features["acc_axis_stability"]),
        tilt_return_score=float(acc_features["tilt_return_score"]),
        acc_available=bool(acc_features.get("acc_available", acc_available)),
    )

    # 生成段階向け強度（1..5）
    intensity_level_1to5 = int(round(1 + max(0.0, min(1.0, signal_confidence)) * 4))
    intensity_level_1to5 = max(1, min(5, intensity_level_1to5))
    motion_features["intensity_level_1to5"] = int(intensity_level_1to5)

    label = "サインあり" if present else "サインなし"
    cmp = ">=" if present else "<"
    reason = (
        f"{label}: run={run_max}/{max(1, int(min_consecutive_above))}, "
        f"gyro_mag_max_1s={gyro_max_f:.3f} {cmp} threshold={threshold:.3f} ({threshold_base}), "
        f"axis={dominant_axis}, sc={dominant_sc}, hint={gesture_hint}, "
        f"nod_score={nod_likelihood_score}, conf={signal_confidence:.2f}"
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
        "axis_sign_changes_2s": dict(axis_sign_changes_2s),
        "axis_map": axis_map,
        "gesture_hint": gesture_hint,
        "motion_features": motion_features,
        "acc_delta_mag_1s": float(acc_features["acc_delta_mag_1s"]),
        "acc_axis_stability": float(acc_features["acc_axis_stability"]),
        "tilt_return_score": float(acc_features["tilt_return_score"]),
        "signal_confidence_0to1": float(signal_confidence),
    }
