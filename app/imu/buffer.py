from __future__ import annotations

import math
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional


@dataclass(frozen=True)
class ImuSample:
    ts: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    has_acc: bool = True

    def to_dict(self) -> Dict[str, float]:
        return {
            "ts": self.ts,
            "ax": self.ax,
            "ay": self.ay,
            "az": self.az,
            "gx": self.gx,
            "gy": self.gy,
            "gz": self.gz,
        }


def _downsample(items: List[ImuSample], max_points: int) -> List[ImuSample]:
    if max_points <= 0 or len(items) <= max_points:
        return items
    if max_points == 1:
        return [items[-1]]
    step = (len(items) - 1) / float(max_points - 1)
    out: List[ImuSample] = []
    for i in range(max_points):
        idx = int(round(i * step))
        out.append(items[idx])
    return out


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    if len(values) == 1:
        v = values[0]
        return {"mean": v, "stdev": 0.0, "min": v, "max": v}
    return {
        "mean": float(statistics.mean(values)),
        "stdev": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _delta_summary(samples: List[ImuSample]) -> Dict[str, object]:
    if len(samples) < 2:
        return {}
    dgx: List[float] = []
    dgy: List[float] = []
    dgz: List[float] = []
    dmag: List[float] = []
    for prev, cur in zip(samples, samples[1:]):
        dx = float(cur.gx - prev.gx)
        dy = float(cur.gy - prev.gy)
        dz = float(cur.gz - prev.gz)
        dgx.append(dx)
        dgy.append(dy)
        dgz.append(dz)
        dmag.append(math.sqrt(dx * dx + dy * dy + dz * dz))

    def _summarize(values: List[float]) -> Dict[str, float]:
        if not values:
            return {}
        mean = float(statistics.mean(values))
        mean_abs = float(statistics.mean([abs(v) for v in values]))
        max_abs = float(max(abs(v) for v in values))
        return {
            "mean": mean,
            "mean_abs": mean_abs,
            "max_abs": max_abs,
        }

    out: Dict[str, object] = {
        "count": len(dmag),
        "gx": _summarize(dgx),
        "gy": _summarize(dgy),
        "gz": _summarize(dgz),
        "mag": _summarize(dmag),
    }
    for axis in ("gx", "gy", "gz", "mag"):
        axis_stats = out.get(axis)
        if isinstance(axis_stats, dict):
            for k, v in list(axis_stats.items()):
                if isinstance(v, float):
                    axis_stats[k] = round(v, 3)
    return out


class ImuBuffer:
    def __init__(self, *, max_seconds: float = 120.0, max_samples: int = 20_000) -> None:
        self._lock = threading.Lock()
        self._samples: Deque[ImuSample] = deque()
        self._max_seconds = float(max_seconds)
        self._max_samples = int(max_samples)

    def add(self, sample: ImuSample) -> None:
        with self._lock:
            self._samples.append(sample)
            cutoff = sample.ts - self._max_seconds
            while self._samples and self._samples[0].ts < cutoff:
                self._samples.popleft()
            while len(self._samples) > self._max_samples:
                self._samples.popleft()

    def latest(self) -> Optional[ImuSample]:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def between(self, *, start_ts: float, end_ts: float) -> List[ImuSample]:
        if end_ts < start_ts:
            return []
        with self._lock:
            return [s for s in self._samples if start_ts <= s.ts <= end_ts]

    def window(self, *, seconds: float, now: Optional[float] = None) -> List[ImuSample]:
        now_ts = time.time() if now is None else float(now)
        cutoff = now_ts - float(seconds)
        with self._lock:
            return [s for s in self._samples if s.ts >= cutoff]

    def sample_rate_hz(self, *, seconds: float = 2.0, now: Optional[float] = None) -> Optional[float]:
        w = self.window(seconds=seconds, now=now)
        if len(w) < 2:
            return None
        dt = w[-1].ts - w[0].ts
        if dt <= 0:
            return None
        return (len(w) - 1) / dt

    def activity(self, *, seconds: float = 1.0, now: Optional[float] = None) -> Dict[str, float]:
        w = self.window(seconds=seconds, now=now)
        if not w:
            return {}
        gyro_mag = [math.sqrt(s.gx * s.gx + s.gy * s.gy + s.gz * s.gz) for s in w]
        acc_mag = [math.sqrt(s.ax * s.ax + s.ay * s.ay + s.az * s.az) for s in w]
        return {
            "gyro_mag_mean": float(statistics.mean(gyro_mag)),
            "gyro_mag_max": float(max(gyro_mag)),
            "acc_mag_mean": float(statistics.mean(acc_mag)),
            "acc_mag_max": float(max(acc_mag)),
        }

    def build_bundle(
        self,
        *,
        now: Optional[float] = None,
        raw_window_sec: float = 2.0,
        raw_max_points: int = 24,
        stats_windows_sec: List[float] | None = None,
    ) -> Dict[str, object]:
        now_ts = time.time() if now is None else float(now)
        latest = self.latest()
        last_ts = latest.ts if latest else None
        age_s = None if last_ts is None else max(0.0, now_ts - last_ts)

        raw = self.window(seconds=raw_window_sec, now=now_ts)
        raw_ds = _downsample(raw, raw_max_points)
        raw_dicts: List[Dict[str, float]] = []
        for s in raw_ds:
            raw_dicts.append(
                {
                    "t_rel_s": round(s.ts - now_ts, 3),
                    "ax": round(s.ax, 3),
                    "ay": round(s.ay, 3),
                    "az": round(s.az, 3),
                    "gx": round(s.gx, 3),
                    "gy": round(s.gy, 3),
                    "gz": round(s.gz, 3),
                }
            )

        windows = stats_windows_sec or [1.0, 5.0, 30.0, 120.0]
        stats: Dict[str, Dict[str, object]] = {}
        for sec in windows:
            w = self.window(seconds=sec, now=now_ts)
            stats[str(int(sec))] = {
                "seconds": sec,
                "count": len(w),
                "ax": _stats([s.ax for s in w]),
                "ay": _stats([s.ay for s in w]),
                "az": _stats([s.az for s in w]),
                "gx": _stats([s.gx for s in w]),
                "gy": _stats([s.gy for s in w]),
                "gz": _stats([s.gz for s in w]),
            }

        rate = self.sample_rate_hz(seconds=raw_window_sec, now=now_ts)
        activity_1s = self.activity(seconds=1.0, now=now_ts)

        out: Dict[str, object] = {
            "now_ts": round(now_ts, 3),
            "last_sample_age_s": None if age_s is None else round(age_s, 3),
            "sample_rate_hz": None if rate is None else round(rate, 2),
            "raw_window_sec": raw_window_sec,
            "raw_samples": raw_dicts,
            "gyro_delta": _delta_summary(raw_ds),
            "activity_1s": {k: round(v, 3) for k, v in activity_1s.items()},
            "stats": stats,
            "sensor_flags": {
                "acc_available": bool(latest.has_acc) if latest is not None else True,
            },
        }
        if latest:
            out["latest"] = {
                "ax": round(latest.ax, 3),
                "ay": round(latest.ay, 3),
                "az": round(latest.az, 3),
                "gx": round(latest.gx, 3),
                "gy": round(latest.gy, 3),
                "gz": round(latest.gz, 3),
            }
        return out

    def format_status_line(self, *, now: Optional[float] = None) -> str:
        now_ts = time.time() if now is None else float(now)
        latest = self.latest()
        if not latest:
            return "IMU: （未受信）"
        rate = self.sample_rate_hz(seconds=2.0, now=now_ts)
        activity_1s = self.activity(seconds=1.0, now=now_ts)
        rate_s = "-" if rate is None else f"{rate:.1f}Hz"
        gm = activity_1s.get("gyro_mag_mean")
        gmx = activity_1s.get("gyro_mag_max")
        am = activity_1s.get("acc_mag_mean")
        amx = activity_1s.get("acc_mag_max")
        return (
            "IMU: "
            f"ax={latest.ax:.3f}, ay={latest.ay:.3f}, az={latest.az:.3f}, "
            f"gx={latest.gx:.3f}, gy={latest.gy:.3f}, gz={latest.gz:.3f}, "
            f"rate={rate_s}, "
            f"gyro_mag_mean_1s={gm:.3f} max_1s={gmx:.3f}, "
            f"acc_mag_mean_1s={am:.3f} max_1s={amx:.3f}"
        )
