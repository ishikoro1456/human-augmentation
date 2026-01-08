from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MotionSnapshot:
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    pitch_angle: float
    yaw_angle: float
    nod_strength: float
    shake_strength: float
    event: str
    nod_level: int
    shake_level: int
    ts: float

    def to_dict(self) -> dict:
        return {
            "ax": self.ax,
            "ay": self.ay,
            "az": self.az,
            "gx": self.gx,
            "gy": self.gy,
            "gz": self.gz,
            "pitch_angle": self.pitch_angle,
            "yaw_angle": self.yaw_angle,
            "nod_strength": self.nod_strength,
            "shake_strength": self.shake_strength,
            "nod_level": self.nod_level,
            "shake_level": self.shake_level,
            "event": self.event,
            "ts": self.ts,
        }


class MotionDetector:
    def __init__(
        self,
        dt: float = 0.3,
        thresh_pitch: float = 20.0,
        thresh_yaw: float = 25.0,
        cooldown: float = 10.0,
    ) -> None:
        self.dt = dt
        self.thresh_pitch = thresh_pitch
        self.thresh_yaw = thresh_yaw
        self.cooldown = cooldown
        self.pitch_angle = 0.0
        self.yaw_angle = 0.0
        self.last_detect = 0.0

    def update(
        self,
        data: Tuple[float, float, float, float, float, float],
        now: float,
    ) -> MotionSnapshot:
        ax, ay, az, gx, gy, gz = data

        # reference のロジック: gy を pitch, gz を yaw に積分
        self.pitch_angle += gy * self.dt
        self.yaw_angle += gz * self.dt
        self.pitch_angle *= 0.98
        self.yaw_angle *= 0.98

        pitch_abs = abs(self.pitch_angle)
        yaw_abs = abs(self.yaw_angle)

        nod_strength = min(yaw_abs / self.thresh_yaw, 1.0)
        shake_strength = min(pitch_abs / self.thresh_pitch, 1.0)
        nod_level = max(1, int(round(nod_strength * 4)) + 1) if nod_strength > 0 else 0
        shake_level = max(1, int(round(shake_strength * 4)) + 1) if shake_strength > 0 else 0

        event = "none"
        if pitch_abs > self.thresh_pitch and (now - self.last_detect > self.cooldown):
            event = "shake"
            self.last_detect = now
            self.pitch_angle = 0.0
            self.yaw_angle = 0.0
        elif yaw_abs > self.thresh_yaw and (now - self.last_detect > self.cooldown):
            event = "nod"
            self.last_detect = now
            self.pitch_angle = 0.0
            self.yaw_angle = 0.0

        return MotionSnapshot(
            ax=ax,
            ay=ay,
            az=az,
            gx=gx,
            gy=gy,
            gz=gz,
            pitch_angle=self.pitch_angle,
            yaw_angle=self.yaw_angle,
            nod_strength=nod_strength,
            shake_strength=shake_strength,
            event=event,
            nod_level=nod_level,
            shake_level=shake_level,
            ts=now,
        )


def format_motion(snapshot: Optional[MotionSnapshot]) -> str:
    if snapshot is None:
        return "IMU情報なし"
    return (
        "IMU: "
        f"ax={snapshot.ax:.3f}, ay={snapshot.ay:.3f}, az={snapshot.az:.3f}, "
        f"gx={snapshot.gx:.3f}, gy={snapshot.gy:.3f}, gz={snapshot.gz:.3f}, "
        f"pitch={snapshot.pitch_angle:.2f}, yaw={snapshot.yaw_angle:.2f}, "
        f"nod_strength={snapshot.nod_strength:.2f}, nod_level={snapshot.nod_level}, "
        f"shake_strength={snapshot.shake_strength:.2f}, shake_level={snapshot.shake_level}, "
        f"event={snapshot.event}"
    )
