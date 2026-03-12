import tempfile
import types
import sys
import unittest
from pathlib import Path

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from fastapi.templating import Jinja2Templates
except Exception:  # pragma: no cover
    FastAPI = None
    TestClient = None
    Jinja2Templates = None


@unittest.skipIf(TestClient is None, "fastapi test client is unavailable")
class ExperimentRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        from app.eval.db import init_db
        from app.eval.routes import experiment as exp_router

        self._tmp = tempfile.TemporaryDirectory()
        db_path = Path(self._tmp.name) / "route_test.sqlite"
        self._conn = init_db(db_path)
        self._orig_main_mod = sys.modules.get("app.eval.main")
        fake_main = types.ModuleType("app.eval.main")
        fake_main.conn = self._conn
        sys.modules["app.eval.main"] = fake_main

        class _FakeDecision:
            def __init__(self, call_id: str) -> None:
                self.call_id = call_id
                self.stage_index = 0
                self.stage_name = "Stage1 YES/NO"
                self.sentence_idx = 0
                self.sentence_text = "テスト"
                self.gesture_hint = "nod"
                self.intensity_1to5 = 3
                self.selected_id = "ST1_YES"
                self.selected_text = "はい"
                self.generated_text = "はい"
                self.generation_mode = "fixed"
                self.constraints_ok = True
                self.reason = "test"
                self.latency_ms = 10
                self.signal_confidence = 0.8
                self.imu_features = {}
                self.audio_path = ""

        class _FakeSession:
            def __init__(self) -> None:
                self.session_id = "sess_test"
                self.sentences = [{"id": 1, "text": "テスト文"}]
                self._state = "stage_review"
                self._stage_index = 0
                self._decisions = [_FakeDecision("c1")]

            def snapshot(self):
                return {
                    "session_id": self.session_id,
                    "state": self._state,
                    "stage_index": self._stage_index,
                    "stage_total": 4,
                    "stage_name": "Stage1 YES/NO",
                    "stage_description": "desc",
                    "stage_runs": [
                        {
                            "stage_index": 0,
                            "stage_name": "Stage1 YES/NO",
                            "decision_count": 1,
                        }
                    ]
                    + [
                        {
                            "stage_index": 1,
                            "stage_name": "Stage2 YES/NO+5強度",
                            "decision_count": 0,
                        },
                        {
                            "stage_index": 2,
                            "stage_name": "Stage3 超短フレーズ",
                            "decision_count": 0,
                        },
                        {
                            "stage_index": 3,
                            "stage_name": "Stage4 文脈あり1文",
                            "decision_count": 0,
                        },
                    ],
                    "current_idx": 0,
                    "total": 1,
                    "decisions": [
                        {
                            "call_id": "c1",
                            "sentence_text": "テスト",
                            "gesture_hint": "nod",
                            "selected_text": "はい",
                            "intensity_1to5": 3,
                            "generation_mode": "fixed",
                            "constraints_ok": True,
                            "signal_confidence": 0.8,
                            "latency_ms": 10,
                        }
                    ],
                    "script_title": "test",
                    "tts_ready": True,
                }

            def get_stage_decisions(self, stage_index: int):
                return list(self._decisions)

            def mark_stage_review_submitted(self, *, stage_index: int, evaluator_id: str) -> None:
                _ = (stage_index, evaluator_id)

            def advance_stage(self) -> bool:
                self._state = "done"
                return False

            def advance(self):
                self._state = "stage_review"
                return False

            def inject_gesture(self, hint: str) -> None:
                _ = hint

        self.fake_session = _FakeSession()

        try:
            import app.eval.experiment as exp_mod
        except ModuleNotFoundError as exc:
            self.skipTest(f"missing optional dependency: {exc}")

        self._orig_get_session = exp_mod.get_session
        exp_mod.get_session = lambda session_id: self.fake_session if session_id == "sess_test" else None

        app = FastAPI()
        templates = Jinja2Templates(
            directory=str(Path(__file__).resolve().parents[1] / "app" / "eval" / "templates")
        )
        exp_router.setup(templates)
        app.include_router(exp_router.router)
        self.client = TestClient(app)

        self._exp_mod = exp_mod

    def tearDown(self) -> None:
        self._exp_mod.get_session = self._orig_get_session
        if self._orig_main_mod is not None:
            sys.modules["app.eval.main"] = self._orig_main_mod
        else:
            sys.modules.pop("app.eval.main", None)
        self._tmp.cleanup()

    def test_stage_review_get_and_post(self) -> None:
        r = self.client.get("/experiment/run/sess_test/stage/0/review")
        self.assertEqual(r.status_code, 200)
        self.assertIn("事後評価", r.text)

        payload = {
            "evaluator_id": "P01",
            "common_q1": "5",
            "common_q2": "5",
            "common_q3": "5",
            "common_q4": "5",
            "specific_q1": "4",
            "specific_q2": "4",
            "specific_q3": "4",
            "overall_comment": "ok",
            "ann_c1_issues": "bad_timing",
            "ann_c1_comment": "気になる",
        }
        r2 = self.client.post(
            "/experiment/run/sess_test/stage/0/review",
            data=payload,
            follow_redirects=False,
        )
        self.assertEqual(r2.status_code, 303)
        self.assertIn("/experiment/run/sess_test/finish", r2.headers.get("location", ""))


if __name__ == "__main__":
    unittest.main()
