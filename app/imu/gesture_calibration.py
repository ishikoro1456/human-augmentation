from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from app.imu.buffer import ImuBuffer, ImuSample


def _sample_rate_hz(samples: List[ImuSample]) -> Optional[float]:
    if len(samples) < 2:
        return None
    dt = samples[-1].ts - samples[0].ts
    if dt <= 0:
        return None
    return (len(samples) - 1) / dt


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.mean(values))


def _max(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(max(values))


@dataclass(frozen=True)
class GestureExample:
    name: str
    instruction: str
    start_ts: float
    end_ts: float
    duration_s: float
    count: int
    sample_rate_hz: Optional[float]
    axis_abs_mean: Dict[str, float]
    axis_abs_max: Dict[str, float]
    gyro_mag_mean: Optional[float]
    gyro_mag_max: Optional[float]
    dominant_axis: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "instruction": self.instruction,
            "start_ts": round(self.start_ts, 3),
            "end_ts": round(self.end_ts, 3),
            "duration_s": round(self.duration_s, 3),
            "count": self.count,
            "sample_rate_hz": None if self.sample_rate_hz is None else round(self.sample_rate_hz, 2),
            "axis_abs_mean": {k: round(v, 3) for k, v in self.axis_abs_mean.items()},
            "axis_abs_max": {k: round(v, 3) for k, v in self.axis_abs_max.items()},
            "gyro_mag_mean": None if self.gyro_mag_mean is None else round(self.gyro_mag_mean, 3),
            "gyro_mag_max": None if self.gyro_mag_max is None else round(self.gyro_mag_max, 3),
            "dominant_axis": self.dominant_axis,
        }

    def summary(self) -> str:
        rate_s = "-" if self.sample_rate_hz is None else f"{self.sample_rate_hz:.1f}Hz"
        gm_s = "-" if self.gyro_mag_mean is None else f"{self.gyro_mag_mean:.2f}"
        gmx_s = "-" if self.gyro_mag_max is None else f"{self.gyro_mag_max:.2f}"
        return f"axis={self.dominant_axis}, rate={rate_s}, gyro_mean={gm_s}, gyro_max={gmx_s}"


@dataclass(frozen=True)
class GestureCalibration:
    started_at: float
    finished_at: float
    examples: Dict[str, GestureExample]
    axis_suggest: Dict[str, str]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "started_at": round(self.started_at, 3),
            "finished_at": round(self.finished_at, 3),
            "examples": {k: v.to_dict() for k, v in self.examples.items()},
            "axis_suggest": dict(self.axis_suggest),
            "warnings": list(self.warnings),
        }

    def summaries(self) -> List[str]:
        keys = ["nod_weak", "nod_strong", "shake_weak", "shake_strong"]
        out: List[str] = []
        for k in keys:
            ex = self.examples.get(k)
            if ex is None:
                continue
            out.append(f"{k}: {ex.summary()}")
        return out


def _example_from_samples(
    *,
    name: str,
    instruction: str,
    samples: List[ImuSample],
    start_ts: float,
    end_ts: float,
) -> GestureExample:
    gx_abs = [abs(s.gx) for s in samples]
    gy_abs = [abs(s.gy) for s in samples]
    gz_abs = [abs(s.gz) for s in samples]
    axis_abs_mean = {
        "gx": float(statistics.mean(gx_abs)) if gx_abs else 0.0,
        "gy": float(statistics.mean(gy_abs)) if gy_abs else 0.0,
        "gz": float(statistics.mean(gz_abs)) if gz_abs else 0.0,
    }
    axis_abs_max = {
        "gx": float(max(gx_abs)) if gx_abs else 0.0,
        "gy": float(max(gy_abs)) if gy_abs else 0.0,
        "gz": float(max(gz_abs)) if gz_abs else 0.0,
    }
    dominant_axis = max(axis_abs_mean, key=lambda k: axis_abs_mean[k]) if axis_abs_mean else "-"
    gyro_mag = [math.sqrt(s.gx * s.gx + s.gy * s.gy + s.gz * s.gz) for s in samples]
    return GestureExample(
        name=name,
        instruction=instruction,
        start_ts=start_ts,
        end_ts=end_ts,
        duration_s=max(0.0, end_ts - start_ts),
        count=len(samples),
        sample_rate_hz=_sample_rate_hz(samples),
        axis_abs_mean=axis_abs_mean,
        axis_abs_max=axis_abs_max,
        gyro_mag_mean=_mean(gyro_mag),
        gyro_mag_max=_max(gyro_mag),
        dominant_axis=dominant_axis,
    )


def _log_default(message: str) -> None:
    print(message)


def run_gesture_calibration(
    buffer: ImuBuffer,
    *,
    weak_sec: float = 2.0,
    strong_sec: float = 2.0,
    start_delay_sec: float = 1.0,
    rest_sec: float = 1.0,
    tick_sec: float = 0.2,
    log: Callable[[str], None] | None = None,
) -> Optional[GestureCalibration]:
    weak_sec = float(weak_sec)
    strong_sec = float(strong_sec)
    if weak_sec <= 0 and strong_sec <= 0:
        return None

    _log = log or _log_default
    started_at = time.time()
    warnings: List[str] = []
    examples: Dict[str, GestureExample] = {}

    steps: List[Tuple[str, str, float]] = []
    if weak_sec > 0:
        steps.append(("nod_weak", "弱く頷いてください。", weak_sec))
    if strong_sec > 0:
        steps.append(("nod_strong", "強く頷いてください。", strong_sec))
    if weak_sec > 0:
        steps.append(("shake_weak", "弱く首を横に振ってください。", weak_sec))
    if strong_sec > 0:
        steps.append(("shake_strong", "強く首を横に振ってください。", strong_sec))

    _log("追加の計測: 頷き/首振りの強さを覚えるための短い計測をします。")
    _log("無理のない範囲で大丈夫です。途中でやめたいときは計測秒数を0にしてください。")

    for name, instruction, duration_s in steps:
        if rest_sec > 0:
            _log(f"準備: 一度止めてください（{int(round(rest_sec))}秒）")
            time.sleep(rest_sec)

        if start_delay_sec > 0:
            _log(f"次: {instruction} {int(round(start_delay_sec))}秒後に計測します。")
            time.sleep(start_delay_sec)
        else:
            _log(f"次: {instruction} いまから計測します。")

        _log(f"計測開始: {name}（{int(round(duration_s))}秒）")
        t0 = time.time()
        last_note = 0.0
        while True:
            now = time.time()
            remaining = (t0 + duration_s) - now
            if remaining <= 0:
                break
            if now - last_note >= 1.0 and remaining <= 3.0:
                _log(f"残り {int(math.ceil(remaining))} 秒")
                last_note = now
            time.sleep(tick_sec)
        t1 = time.time()

        samples = buffer.between(start_ts=t0, end_ts=t1)
        ex = _example_from_samples(
            name=name,
            instruction=instruction,
            samples=samples,
            start_ts=t0,
            end_ts=t1,
        )
        examples[name] = ex
        if ex.count < 10:
            warnings.append(f"{name}_samples_low")
        _log(f"計測結果({name}): {ex.summary()}")

    axis_suggest: Dict[str, str] = {}
    nod = examples.get("nod_weak")
    shake = examples.get("shake_weak")
    if nod:
        axis_suggest["nod_axis"] = nod.dominant_axis
    if shake:
        axis_suggest["shake_axis"] = shake.dominant_axis

    finished_at = time.time()
    calib = GestureCalibration(
        started_at=started_at,
        finished_at=finished_at,
        examples=examples,
        axis_suggest=axis_suggest,
        warnings=warnings,
    )
    if axis_suggest:
        _log(
            "軸の推定: "
            f"nod={axis_suggest.get('nod_axis','-')}, "
            f"shake={axis_suggest.get('shake_axis','-')}"
        )
    if warnings:
        _log(f"注意: {', '.join(warnings)}")
    return calib
