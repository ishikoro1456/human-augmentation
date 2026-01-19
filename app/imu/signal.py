from __future__ import annotations

import math
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
    """
    閾値を超えた動きの持続時間を計算する
    Returns: (duration_s, first_above_t_rel, last_above_t_rel)
    """
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
    """
    姿勢の変化を計算する（ジャイロの積分）
    Returns: {
        "integrated_change": 動きの間の積分値,
        "posture_returned": 姿勢が元に戻ったか（積分値が小さい）,
        "start_avg": 開始時の平均値,
        "end_avg": 終了時の平均値,
    }
    """
    # window_sec 以内のサンプルを取得
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
    
    # 積分（累積和）
    integrated = 0.0
    prev_t = samples_in_window[0][0]
    for t, val in samples_in_window[1:]:
        dt = t - prev_t
        if dt > 0:
            integrated += val * dt
        prev_t = t
    
    # 開始時と終了時の平均（最初と最後の1/4ずつ）
    n = len(samples_in_window)
    quarter = max(1, n // 4)
    start_vals = [v for _, v in samples_in_window[:quarter]]
    end_vals = [v for _, v in samples_in_window[-quarter:]]
    start_avg = sum(start_vals) / len(start_vals) if start_vals else 0.0
    end_avg = sum(end_vals) / len(end_vals) if end_vals else 0.0
    
    # 姿勢が戻ったか（積分値が小さい、かつ開始と終了の平均が近い）
    posture_returned = abs(integrated) < 5.0 and abs(start_avg - end_avg) < 2.0
    
    return {
        "integrated_change": round(integrated, 3),
        "posture_returned": posture_returned,
        "start_avg": round(start_avg, 3),
        "end_avg": round(end_avg, 3),
    }


def _compute_oscillation_symmetry(values: List[float]) -> Dict[str, object]:
    """
    振幅の対称性を計算する
    頷きは正負が対称、メモ取りは一方向に偏る
    """
    if not values:
        return {"symmetry_ratio": 1.0, "is_symmetric": True, "positive_sum": 0.0, "negative_sum": 0.0}
    
    positive_sum = sum(v for v in values if v > 0)
    negative_sum = sum(abs(v) for v in values if v < 0)
    
    total = positive_sum + negative_sum
    if total < 0.1:
        return {"symmetry_ratio": 1.0, "is_symmetric": True, "positive_sum": 0.0, "negative_sum": 0.0}
    
    # 対称なら ratio は 1.0 に近い
    if positive_sum > negative_sum:
        ratio = negative_sum / positive_sum if positive_sum > 0 else 0.0
    else:
        ratio = positive_sum / negative_sum if negative_sum > 0 else 0.0
    
    # 0.5 以上なら対称と判断
    is_symmetric = ratio >= 0.5
    
    return {
        "symmetry_ratio": round(ratio, 3),
        "is_symmetric": is_symmetric,
        "positive_sum": round(positive_sum, 3),
        "negative_sum": round(negative_sum, 3),
    }


def _compute_baseline_comparison(
    imu_bundle: Dict[str, object],
) -> Dict[str, object]:
    """
    長期ベースライン（5秒、30秒）と比較する
    """
    stats = imu_bundle.get("stats", {})
    activity_1s = imu_bundle.get("activity_1s", {})
    
    if not isinstance(stats, dict) or not isinstance(activity_1s, dict):
        return {"ratio_vs_5s": None, "ratio_vs_30s": None}
    
    gyro_mag_max_1s = activity_1s.get("gyro_mag_max")
    if not isinstance(gyro_mag_max_1s, (int, float)):
        return {"ratio_vs_5s": None, "ratio_vs_30s": None}
    
    result: Dict[str, object] = {}
    
    # 5秒との比較
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
    
    # 30秒との比較
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
    if not isinstance(raw_samples, list):
        raw_samples = []
    
    mags_1s = []
    mags_2s = []
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
        if not isinstance(gx, (int, float)) or not isinstance(gy, (int, float)) or not isinstance(
            gz, (int, float)
        ):
            continue
        gx_f = float(gx)
        gy_f = float(gy)
        gz_f = float(gz)
        mag = math.sqrt(gx_f * gx_f + gy_f * gy_f + gz_f * gz_f)
        
        # 2秒以内のデータ
        if float(t_rel) >= -2.0:
            mags_2s.append(mag)
            axis_values_2s["gx"].append(gx_f)
            axis_values_2s["gy"].append(gy_f)
            axis_values_2s["gz"].append(gz_f)
        
        # 1秒以内のデータ
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
    # 2秒ウィンドウでの符号変化
    axis_sign_changes_2s = {
        "gx": _sign_changes(axis_values_2s["gx"]),
        "gy": _sign_changes(axis_values_2s["gy"]),
        "gz": _sign_changes(axis_values_2s["gz"]),
    }
    dominant_sc = axis_sign_changes.get(dominant_axis, 0) if dominant_axis in axis_sign_changes else 0
    dominant_sc_2s = axis_sign_changes_2s.get(dominant_axis, 0) if dominant_axis in axis_sign_changes_2s else 0

    # 追加の特徴量を計算
    motion_duration, first_above_t, last_above_t = _compute_motion_duration(raw_samples, threshold)
    posture_change = _compute_posture_change(raw_samples, dominant_axis, window_sec=2.0)
    oscillation_symmetry = _compute_oscillation_symmetry(axis_values_1s.get(dominant_axis, []))
    baseline_comparison = _compute_baseline_comparison(imu_bundle)

    # 動きの特徴をまとめる
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
    }

    # 頷きらしさスコア（LLMの参考用）
    nod_likelihood_score = 0
    if motion_features["has_oscillation"]:
        nod_likelihood_score += 2
    if motion_features["posture_returned"]:
        nod_likelihood_score += 2
    if motion_features["is_symmetric"]:
        nod_likelihood_score += 1
    if 0.3 <= motion_duration <= 2.0:
        nod_likelihood_score += 1
    motion_features["nod_likelihood_score"] = nod_likelihood_score  # 0-6

    # ジェスチャーの判定は少し厳しめにして誤検知を減らす
    gesture_hint = "other"
    enough_sc = (dominant_sc >= 2) or (dominant_sc >= 1 and dominant_sc_2s >= 2)
    if present and nod_likelihood_score >= 4 and enough_sc:
        if dominant_axis == nod_axis:
            gesture_hint = "nod"
        elif dominant_axis == shake_axis:
            gesture_hint = "shake"

    cmp = ">=" if present else "<"
    reason = (
        f"{label}: run={run_max}/{max(1, int(min_consecutive_above))}, "
        f"gyro_mag_max_1s={gyro_max_f:.3f} {cmp} threshold={threshold:.3f} ({threshold_base}), "
        f"axis={dominant_axis}, sc={dominant_sc}, hint={gesture_hint}, "
        f"nod_score={nod_likelihood_score}"
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
    }
