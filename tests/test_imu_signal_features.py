import unittest

from app.imu.signal import detect_backchannel_signal


class ImuSignalFeatureTests(unittest.TestCase):
    def test_detect_includes_6axis_features(self) -> None:
        raw_samples = []
        # 約1秒分、往復動作っぽい波形
        for i in range(12):
            t_rel = -1.2 + (i * 0.1)
            gx = 4.0 if i % 2 == 0 else -4.0
            gy = 2.0 if i % 3 == 0 else -2.0
            gz = 10.0 if i % 2 == 0 else -9.0
            ax = 0.1 * i
            ay = 9.6 + (0.05 * (-1) ** i)
            az = 0.2 * (-1) ** i
            raw_samples.append(
                {
                    "t_rel_s": round(t_rel, 3),
                    "ax": ax,
                    "ay": ay,
                    "az": az,
                    "gx": gx,
                    "gy": gy,
                    "gz": gz,
                }
            )

        imu_bundle = {
            "last_sample_age_s": 0.05,
            "activity_1s": {
                "gyro_mag_max": 12.0,
                "gyro_mag_mean": 6.0,
                "acc_mag_mean": 9.7,
                "acc_mag_max": 9.9,
            },
            "raw_samples": raw_samples,
            "stats": {},
        }

        out = detect_backchannel_signal(
            imu_bundle,
            calibration=None,
            abs_threshold=5.0,
            min_consecutive_above=2,
            nod_axis="gz",
            shake_axis="gx",
        )

        self.assertIn("acc_delta_mag_1s", out)
        self.assertIn("acc_axis_stability", out)
        self.assertIn("tilt_return_score", out)
        self.assertIn("signal_confidence_0to1", out)
        self.assertIn("motion_features", out)
        self.assertIn("intensity_level_1to5", out["motion_features"])


if __name__ == "__main__":
    unittest.main()
