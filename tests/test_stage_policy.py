import unittest

from app.eval.stage_policy import (
    StageConfig,
    estimate_intensity_level,
    generate_stage_response,
    get_stage_config,
)


class _FakeResp:
    def __init__(self, text: str) -> None:
        self.output_text = text


class _FakeResponses:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)

    def create(self, **kwargs):  # noqa: ANN003
        if self._outputs:
            return _FakeResp(self._outputs.pop(0))
        return _FakeResp("はい")


class _FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.responses = _FakeResponses(outputs)


class StagePolicyTests(unittest.TestCase):
    def test_stage1_fixed_yes(self) -> None:
        stage = get_stage_config(0)
        res = generate_stage_response(
            client=_FakeClient([]),
            model="dummy",
            stage=stage,
            gesture_hint="nod",
            utterance="",
            context="",
            intensity_1to5=3,
            imu_features={},
        )
        self.assertEqual(res.text, "はい")
        self.assertEqual(res.generation_mode, "fixed")

    def test_stage3_retry_on_constraint_violation(self) -> None:
        stage = get_stage_config(2)
        # 1回目: 制約違反（句点あり長文）
        # 2回目: 制約内
        client = _FakeClient(["これは長すぎる文章です。", "うん"])
        res = generate_stage_response(
            client=client,
            model="dummy",
            stage=stage,
            gesture_hint="nod",
            utterance="テスト",
            context="テスト文脈",
            intensity_1to5=3,
            imu_features={"signal_confidence_0to1": 0.8},
        )
        self.assertEqual(res.text, "うん")
        self.assertTrue(res.constraints_ok)
        self.assertEqual(res.generation_mode, "ai_generated")

    def test_intensity_estimation_bounds(self) -> None:
        self.assertEqual(estimate_intensity_level(nod_score=0, signal_confidence=0.0), 1)
        self.assertGreaterEqual(estimate_intensity_level(nod_score=6, signal_confidence=1.0), 4)


if __name__ == "__main__":
    unittest.main()
