from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from app.imu.buffer import ImuBuffer, ImuSample


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    if len(values) == 1:
        v = float(values[0])
        return {"mean": v, "stdev": 0.0, "min": v, "max": v}
    return {
        "mean": float(statistics.mean(values)),
        "stdev": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _sample_rate_hz(samples: List[ImuSample]) -> Optional[float]:
    if len(samples) < 2:
        return None
    dt = samples[-1].ts - samples[0].ts
    if dt <= 0:
        return None
    return (len(samples) - 1) / dt


def _count_sign_changes(values: List[float]) -> int:
    last_sign = 0
    changes = 0
    for v in values:
        if v > 0:
            sign = 1
        elif v < 0:
            sign = -1
        else:
            continue
        if last_sign != 0 and sign != last_sign:
            changes += 1
        last_sign = sign
    return changes


@dataclass(frozen=True)
class CalibrationPhase:
    name: str
    start_ts: float
    end_ts: float
    duration_s: float
    count: int
    sample_rate_hz: Optional[float]
    gyro_mag: Dict[str, float]
    acc_mag: Dict[str, float]
    axis_sign_changes: Dict[str, int]
    axis_cycles_hz: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "start_ts": round(self.start_ts, 3),
            "end_ts": round(self.end_ts, 3),
            "duration_s": round(self.duration_s, 3),
            "count": self.count,
            "sample_rate_hz": None if self.sample_rate_hz is None else round(self.sample_rate_hz, 2),
            "gyro_mag": {k: round(v, 4) for k, v in self.gyro_mag.items()},
            "acc_mag": {k: round(v, 4) for k, v in self.acc_mag.items()},
            "axis_sign_changes": {k: int(v) for k, v in self.axis_sign_changes.items()},
            "axis_cycles_hz": {k: round(float(v), 3) for k, v in self.axis_cycles_hz.items()},
        }

    def summary(self) -> str:
        gm = self.gyro_mag.get("mean")
        gs = self.gyro_mag.get("stdev")
        am = self.acc_mag.get("mean")
        a_s = self.acc_mag.get("stdev")
        gm_s = "-" if gm is None else f"{gm:.3f}"
        gs_s = "-" if gs is None else f"{gs:.3f}"
        am_s = "-" if am is None else f"{am:.3f}"
        a_s_s = "-" if a_s is None else f"{a_s:.3f}"
        rate_s = "-" if self.sample_rate_hz is None else f"{self.sample_rate_hz:.1f}Hz"
        return f"rate={rate_s}, gyro_mean={gm_s}±{gs_s}, acc_mean={am_s}±{a_s_s}"


@dataclass(frozen=True)
class ImuCalibration:
    started_at: float
    finished_at: float
    still: Optional[CalibrationPhase]
    active: Optional[CalibrationPhase]
    warnings: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "started_at": round(self.started_at, 3),
            "finished_at": round(self.finished_at, 3),
            "still": None if self.still is None else self.still.to_dict(),
            "active": None if self.active is None else self.active.to_dict(),
            "warnings": list(self.warnings),
        }


def normalize_activity(
    activity_1s: Dict[str, float],
    calibration: ImuCalibration,
) -> Dict[str, object]:
    base = calibration.active or calibration.still
    if base is None:
        return {}

    def _z(value: Optional[float], mean: Optional[float], stdev: Optional[float]) -> Optional[float]:
        if value is None or mean is None or stdev is None:
            return None
        if stdev <= 0:
            return None
        return (float(value) - float(mean)) / float(stdev)

    gyro_mean = base.gyro_mag.get("mean")
    gyro_stdev = base.gyro_mag.get("stdev")
    acc_mean = base.acc_mag.get("mean")
    acc_stdev = base.acc_mag.get("stdev")

    gm = activity_1s.get("gyro_mag_mean")
    gmx = activity_1s.get("gyro_mag_max")
    am = activity_1s.get("acc_mag_mean")
    amx = activity_1s.get("acc_mag_max")

    out: Dict[str, object] = {
        "base_phase": base.name,
        "gyro_mag_mean_z": _z(gm, gyro_mean, gyro_stdev),
        "gyro_mag_max_z": _z(gmx, gyro_mean, gyro_stdev),
        "acc_mag_mean_z": _z(am, acc_mean, acc_stdev),
        "acc_mag_max_z": _z(amx, acc_mean, acc_stdev),
    }
    for k, v in list(out.items()):
        if isinstance(v, float):
            out[k] = round(v, 3)
    return out


def _phase_from_samples(
    *,
    name: str,
    samples: List[ImuSample],
    start_ts: float,
    end_ts: float,
) -> CalibrationPhase:
    gyro_mag = [math.sqrt(s.gx * s.gx + s.gy * s.gy + s.gz * s.gz) for s in samples]
    acc_mag = [math.sqrt(s.ax * s.ax + s.ay * s.ay + s.az * s.az) for s in samples]
    duration_s = max(0.0, end_ts - start_ts)
    gx = [s.gx for s in samples]
    gy = [s.gy for s in samples]
    gz = [s.gz for s in samples]
    axis_sc = {"gx": _count_sign_changes(gx), "gy": _count_sign_changes(gy), "gz": _count_sign_changes(gz)}
    axis_cycles_hz: Dict[str, float] = {}
    for axis, sc in axis_sc.items():
        if duration_s <= 0:
            axis_cycles_hz[axis] = 0.0
        else:
            axis_cycles_hz[axis] = float(sc) / 2.0 / float(duration_s)
    return CalibrationPhase(
        name=name,
        start_ts=start_ts,
        end_ts=end_ts,
        duration_s=duration_s,
        count=len(samples),
        sample_rate_hz=_sample_rate_hz(samples),
        gyro_mag=_stats(gyro_mag),
        acc_mag=_stats(acc_mag),
        axis_sign_changes=axis_sc,
        axis_cycles_hz=axis_cycles_hz,
    )


def run_calibration(
    buffer: ImuBuffer,
    *,
    still_sec: float,
    active_sec: float,
    start_delay_sec: float = 0.0,
    between_phases_sec: float = 0.0,
    wait_for_imu_sec: float = 15.0,
    tick_sec: float = 0.2,
    log: Callable[[str], None] | None = None,
) -> Optional[ImuCalibration]:
    def _log(message: str) -> None:
        if log:
            log(message)
        else:
            print(message)

    still_sec = float(still_sec)
    active_sec = float(active_sec)
    if still_sec <= 0 and active_sec <= 0:
        return None

    wait_start = time.time()
    while True:
        latest = buffer.latest()
        if latest is not None and time.time() - latest.ts < 2.0:
            break
        if time.time() - wait_start > wait_for_imu_sec:
            _log("IMUが見つからないので、計測をスキップします。")
            return None
        time.sleep(0.2)

    started_at = time.time()
    warnings: List[str] = []
    still_phase: Optional[CalibrationPhase] = None
    active_phase: Optional[CalibrationPhase] = None

    first_phase = "静止" if still_sec > 0 else "活動"
    if start_delay_sec > 0:
        _log(f"計測を始めます。{int(math.ceil(start_delay_sec))}秒後に{first_phase}計測に入ります。")
        time.sleep(start_delay_sec)

    if still_sec > 0:
        _log(f"計測(1/2): 静止してください。{int(round(still_sec))}秒ほど待ちます。")
        t0 = time.time()
        last_note = 0.0
        while True:
            now = time.time()
            remaining = (t0 + still_sec) - now
            if remaining <= 0:
                break
            if now - last_note >= 1.0 and remaining <= 5.0:
                _log(f"静止の残り {int(math.ceil(remaining))} 秒")
                last_note = now
            time.sleep(tick_sec)
        t1 = time.time()
        samples = buffer.between(start_ts=t0, end_ts=t1)
        still_phase = _phase_from_samples(name="still", samples=samples, start_ts=t0, end_ts=t1)
        if still_phase.count < 20:
            warnings.append("still_samples_low")

    if active_sec > 0:
        if still_sec > 0 and between_phases_sec > 0:
            _log(f"次の計測まで{int(math.ceil(between_phases_sec))}秒休憩します。")
            time.sleep(between_phases_sec)
        _log(
            "計測(2/2): 普段どおりに聞いているつもりで、自然に動いてください。"
            f"{int(round(active_sec))}秒ほど待ちます。"
        )
        t0 = time.time()
        last_note = 0.0
        while True:
            now = time.time()
            remaining = (t0 + active_sec) - now
            if remaining <= 0:
                break
            if now - last_note >= 1.0 and remaining <= 5.0:
                _log(f"計測の残り {int(math.ceil(remaining))} 秒")
                last_note = now
            time.sleep(tick_sec)
        t1 = time.time()
        samples = buffer.between(start_ts=t0, end_ts=t1)
        active_phase = _phase_from_samples(name="active", samples=samples, start_ts=t0, end_ts=t1)
        if active_phase.count < 40:
            warnings.append("active_samples_low")

    finished_at = time.time()
    calib = ImuCalibration(
        started_at=started_at,
        finished_at=finished_at,
        still=still_phase,
        active=active_phase,
        warnings=warnings,
    )
    _log("計測が終わりました。")
    if calib.still:
        _log(f"計測結果(still): {calib.still.summary()}")
    if calib.active:
        _log(f"計測結果(active): {calib.active.summary()}")
    if calib.warnings:
        _log(f"注意: {', '.join(calib.warnings)}")
    return calib
