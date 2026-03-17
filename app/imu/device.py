from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional

import serial
from serial import SerialException

from .parser import parse_imu_line


_FLOAT_PAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
_OUTPUT_AXES = ("ax", "ay", "az", "gx", "gy", "gz")


@dataclass(frozen=True)
class DeviceProfile:
    device_id: str
    input_kind: str
    baud: int = 115200
    port: str = ""
    port_globs: tuple[str, ...] = ()
    axis_map: Dict[str, str] = field(default_factory=dict)
    sign_flip: Dict[str, int] = field(default_factory=dict)
    nod_axis: str = ""
    shake_axis: str = ""


@dataclass(frozen=True)
class NormalizedImuReading:
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    has_acc: bool = True


@dataclass(frozen=True)
class ProbeReport:
    detected_format: str
    sample_count: int
    mean_norm: float | None
    stdev_norm: float | None
    preview: tuple[tuple[float, ...], ...]
    reason: str


def _identity_axis_map(input_kind: str) -> Dict[str, str]:
    if input_kind == "six_axis":
        return {axis: axis for axis in _OUTPUT_AXES}
    if input_kind == "gyro_xyz":
        return {"gx": "x", "gy": "y", "gz": "z"}
    return {}


def load_device_profiles(path: Path) -> Dict[str, DeviceProfile]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("profiles", [])
    if not isinstance(rows, list):
        raise ValueError("profiles must be a list")

    out: Dict[str, DeviceProfile] = {}
    for idx, item in enumerate(rows):
        if not isinstance(item, dict):
            raise ValueError(f"profiles[{idx}] must be an object")
        device_id = str(item.get("device_id", "")).strip()
        if not device_id:
            raise ValueError(f"profiles[{idx}] device_id is required")
        input_kind = str(item.get("input_kind", "")).strip()
        if input_kind not in ("six_axis", "gyro_xyz"):
            raise ValueError(f"profiles[{idx}] input_kind must be six_axis or gyro_xyz")

        axis_map_raw = item.get("axis_map", {})
        sign_flip_raw = item.get("sign_flip", {})
        axis_map = _identity_axis_map(input_kind)
        if isinstance(axis_map_raw, dict):
            for key, value in axis_map_raw.items():
                axis_map[str(key)] = str(value)
        sign_flip = {axis: 1 for axis in _OUTPUT_AXES}
        if isinstance(sign_flip_raw, dict):
            for key, value in sign_flip_raw.items():
                try:
                    sign_flip[str(key)] = -1 if int(value) < 0 else 1
                except Exception:
                    sign_flip[str(key)] = 1

        port_globs_raw = item.get("port_globs", [])
        port_globs: tuple[str, ...] = ()
        if isinstance(port_globs_raw, list):
            port_globs = tuple(str(v) for v in port_globs_raw if str(v).strip())

        nod_axis = str(item.get("nod_axis", "")).strip()
        if nod_axis not in ("gx", "gy", "gz"):
            nod_axis = ""
        shake_axis = str(item.get("shake_axis", "")).strip()
        if shake_axis not in ("gx", "gy", "gz"):
            shake_axis = ""

        out[device_id] = DeviceProfile(
            device_id=device_id,
            input_kind=input_kind,
            baud=int(item.get("baud", 115200)),
            port=str(item.get("port", "")).strip(),
            port_globs=port_globs,
            axis_map=axis_map,
            sign_flip=sign_flip,
            nod_axis=nod_axis,
            shake_axis=shake_axis,
        )
    return out


def get_device_profile(path: Path, device_id: str) -> DeviceProfile:
    profiles = load_device_profiles(path)
    profile = profiles.get(str(device_id))
    if profile is None:
        raise ValueError(f"unknown device_id: {device_id}")
    return profile


def resolve_serial_port(profile: DeviceProfile, *, override_port: str = "") -> str:
    if override_port.strip():
        return override_port.strip()
    if profile.port:
        return profile.port
    candidates: list[str] = []
    for pattern in profile.port_globs:
        candidates.extend(sorted(str(path) for path in Path("/").glob(pattern.lstrip("/"))))
    uniq = sorted(dict.fromkeys(candidates))
    if len(uniq) == 1:
        return uniq[0]
    if not uniq:
        patterns = ", ".join(profile.port_globs) if profile.port_globs else "(none)"
        raise ValueError(
            f"no serial port matched for device_id={profile.device_id}. "
            f"patterns={patterns}. "
            "別の名前で見えている場合は --port か DEMO_PORT=/dev/cu.... を使ってください。"
        )
    joined = ", ".join(uniq)
    raise ValueError(
        f"multiple serial ports matched for device_id={profile.device_id}: {joined}. "
        "使うポートを --port か DEMO_PORT=/dev/cu.... で指定してください。"
    )


def parse_xyz_line(line: str) -> Optional[tuple[float, float, float]]:
    labeled: dict[str, float] = {}
    for key in ("x", "y", "z"):
        match = re.search(rf"\b{key}\b\s*[:=]\s*({_FLOAT_PAT})", line, flags=re.IGNORECASE)
        if match:
            try:
                labeled[key] = float(match.group(1))
            except Exception:
                pass
    if len(labeled) == 3:
        return labeled["x"], labeled["y"], labeled["z"]

    values: list[float] = []
    for match in re.finditer(_FLOAT_PAT, line):
        try:
            values.append(float(match.group(0)))
        except Exception:
            continue
    if len(values) == 3:
        return values[0], values[1], values[2]
    return None


def normalize_reading(
    *,
    profile: DeviceProfile,
    six_axis: tuple[float, float, float, float, float, float] | None = None,
    xyz: tuple[float, float, float] | None = None,
) -> NormalizedImuReading:
    source: Dict[str, float] = {}
    if profile.input_kind == "six_axis":
        if six_axis is None:
            raise ValueError("six_axis data is required")
        source = dict(zip(_OUTPUT_AXES, six_axis))
    elif profile.input_kind == "gyro_xyz":
        if xyz is None:
            raise ValueError("xyz data is required")
        source = {"x": xyz[0], "y": xyz[1], "z": xyz[2]}
    else:
        raise ValueError(f"unsupported input_kind: {profile.input_kind}")

    normalized: Dict[str, float] = {}
    has_acc = False
    for axis in _OUTPUT_AXES:
        src = profile.axis_map.get(axis, "")
        if src and src in source:
            normalized[axis] = float(source[src]) * int(profile.sign_flip.get(axis, 1))
            if axis in ("ax", "ay", "az"):
                has_acc = True
        else:
            normalized[axis] = 0.0
    return NormalizedImuReading(
        ax=normalized["ax"],
        ay=normalized["ay"],
        az=normalized["az"],
        gx=normalized["gx"],
        gy=normalized["gy"],
        gz=normalized["gz"],
        has_acc=has_acc,
    )


def read_device_imu_lines(
    profile: DeviceProfile,
    *,
    override_port: str = "",
    override_baud: int | None = None,
    on_log: Callable[[str], None] | None = None,
) -> Iterator[NormalizedImuReading]:
    port = resolve_serial_port(profile, override_port=override_port)
    baud = int(profile.baud if override_baud is None else override_baud)

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
                if not line:
                    continue
                reading: NormalizedImuReading | None = None
                if profile.input_kind == "six_axis":
                    parsed = parse_imu_line(line)
                    if parsed is not None:
                        reading = normalize_reading(profile=profile, six_axis=parsed)
                elif profile.input_kind == "gyro_xyz":
                    parsed_xyz = parse_xyz_line(line)
                    if parsed_xyz is not None:
                        reading = normalize_reading(profile=profile, xyz=parsed_xyz)
                if reading is None:
                    continue
                yield reading
                time.sleep(0)
        except SerialException:
            _log("IMUの接続が切れました。再接続を待ちます。")
            time.sleep(2)
        finally:
            try:
                ser.close()
            except Exception:
                pass


def probe_serial_format(*, port: str, baud: int, seconds: float = 2.0) -> ProbeReport:
    xyz_samples: list[tuple[float, float, float]] = []
    six_axis_count = 0
    raw_preview: list[tuple[float, ...]] = []
    started = time.time()

    try:
        ser = serial.Serial(port, baud, timeout=1)
    except (FileNotFoundError, SerialException) as exc:
        return ProbeReport(
            detected_format="unknown",
            sample_count=0,
            mean_norm=None,
            stdev_norm=None,
            preview=(),
            reason=f"シリアルを開けません: {exc}",
        )

    try:
        while (time.time() - started) < float(max(0.2, seconds)):
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            parsed6 = parse_imu_line(line)
            if parsed6 is not None:
                six_axis_count += 1
                if len(raw_preview) < 5:
                    raw_preview.append(tuple(round(v, 3) for v in parsed6))
                continue
            parsed3 = parse_xyz_line(line)
            if parsed3 is not None:
                xyz_samples.append(parsed3)
                if len(raw_preview) < 5:
                    raw_preview.append(tuple(round(v, 3) for v in parsed3))
    finally:
        ser.close()

    if six_axis_count > 0:
        return ProbeReport(
            detected_format="six_axis",
            sample_count=int(six_axis_count),
            mean_norm=None,
            stdev_norm=None,
            preview=tuple(raw_preview),
            reason="6 軸フォーマットを読めました。",
        )

    report = classify_xyz_semantics(xyz_samples)
    if raw_preview and not report.preview:
        return ProbeReport(
            detected_format=report.detected_format,
            sample_count=report.sample_count,
            mean_norm=report.mean_norm,
            stdev_norm=report.stdev_norm,
            preview=tuple(raw_preview),
            reason=report.reason,
        )
    return report


def classify_xyz_semantics(samples: list[tuple[float, float, float]]) -> ProbeReport:
    if not samples:
        return ProbeReport(
            detected_format="unknown",
            sample_count=0,
            mean_norm=None,
            stdev_norm=None,
            preview=(),
            reason="sample がありません。",
        )

    norms = [
        (x * x + y * y + z * z) ** 0.5
        for x, y, z in samples
    ]
    mean_norm = float(statistics.mean(norms))
    stdev_norm = float(statistics.pstdev(norms)) if len(norms) > 1 else 0.0
    preview = tuple(tuple(round(v, 3) for v in sample) for sample in samples[:5])

    if 0.7 <= mean_norm <= 1.3 or 8.0 <= mean_norm <= 11.5:
        return ProbeReport(
            detected_format="accel_xyz",
            sample_count=len(samples),
            mean_norm=round(mean_norm, 3),
            stdev_norm=round(stdev_norm, 3),
            preview=preview,
            reason="ベクトル長の平均が重力加速度付近なので、加速度 3 軸の可能性が高いです。",
        )

    if mean_norm < 5.0:
        return ProbeReport(
            detected_format="gyro_xyz",
            sample_count=len(samples),
            mean_norm=round(mean_norm, 3),
            stdev_norm=round(stdev_norm, 3),
            preview=preview,
            reason="ベクトル長の平均が小さく重力加速度らしさがないので、ジャイロ 3 軸の可能性が高いです。",
        )

    return ProbeReport(
        detected_format="unknown",
        sample_count=len(samples),
        mean_norm=round(mean_norm, 3),
        stdev_norm=round(stdev_norm, 3),
        preview=preview,
        reason="値域だけでは判定できません。生データとファームウェア仕様を確認してください。",
    )
