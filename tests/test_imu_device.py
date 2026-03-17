import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.imu.device import DeviceProfile, classify_xyz_semantics, get_device_profile, normalize_reading
from app.imu.signal import detect_backchannel_signal


class DeviceNormalizationTests(unittest.TestCase):
    def test_classify_accel_xyz(self) -> None:
        samples = [
            (0.0, 0.1, 9.8),
            (0.2, 0.0, 9.7),
            (0.1, -0.1, 9.9),
        ]
        report = classify_xyz_semantics(samples)
        self.assertEqual(report.detected_format, "accel_xyz")

    def test_classify_gyro_xyz(self) -> None:
        samples = [
            (0.1, -0.2, 0.3),
            (0.5, -0.4, 0.2),
            (0.0, 0.1, -0.1),
        ]
        report = classify_xyz_semantics(samples)
        self.assertEqual(report.detected_format, "gyro_xyz")

    def test_normalize_gyro_xyz_marks_acc_unavailable(self) -> None:
        profile = DeviceProfile(
            device_id="demo-gyro-xyz",
            input_kind="gyro_xyz",
            axis_map={"gx": "x", "gy": "y", "gz": "z"},
            sign_flip={"gx": 1, "gy": 1, "gz": 1},
        )
        reading = normalize_reading(profile=profile, xyz=(1.0, 2.0, 3.0))
        self.assertFalse(reading.has_acc)
        self.assertEqual((reading.gx, reading.gy, reading.gz), (1.0, 2.0, 3.0))

    def test_detect_signal_with_gyro_only_bundle(self) -> None:
        raw_samples = []
        for i in range(12):
            t_rel = -1.2 + (i * 0.1)
            raw_samples.append(
                {
                    "t_rel_s": round(t_rel, 3),
                    "ax": 0.0,
                    "ay": 0.0,
                    "az": 0.0,
                    "gx": 2.0 if i % 3 == 0 else -2.0,
                    "gy": 1.0 if i % 2 == 0 else -1.0,
                    "gz": 10.0 if i % 2 == 0 else -9.5,
                }
            )

        imu_bundle = {
            "last_sample_age_s": 0.05,
            "activity_1s": {
                "gyro_mag_max": 12.0,
                "gyro_mag_mean": 6.0,
                "acc_mag_mean": 0.0,
                "acc_mag_max": 0.0,
            },
            "raw_samples": raw_samples,
            "stats": {},
            "sensor_flags": {"acc_available": False},
        }

        out = detect_backchannel_signal(
            imu_bundle,
            calibration=None,
            abs_threshold=5.0,
            min_consecutive_above=2,
            nod_axis="gz",
            shake_axis="gx",
        )

        self.assertEqual(out["gesture_hint"], "nod")
        self.assertFalse(out["motion_features"]["acc_available"])

    def test_load_device_profile_with_gesture_axes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "devices.json"
            path.write_text(
                """
                {
                  "profiles": [
                    {
                      "device_id": "demo-usbserial-six-axis",
                      "input_kind": "six_axis",
                      "nod_axis": "gz",
                      "shake_axis": "gx"
                    }
                  ]
                }
                """.strip(),
                encoding="utf-8",
            )
            profile = get_device_profile(path, "demo-usbserial-six-axis")

        self.assertEqual(profile.nod_axis, "gz")
        self.assertEqual(profile.shake_axis, "gx")


if __name__ == "__main__":
    unittest.main()
