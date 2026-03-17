import tempfile
import unittest
from pathlib import Path

from app.eval.db import (
    get_response_annotations,
    get_stage_evaluation,
    init_db,
    replace_response_annotations,
    save_stage_evaluation,
)
from app.eval.models import ResponseAnnotation, StageEvaluation


class EvalDbStageTests(unittest.TestCase):
    def test_stage_eval_and_annotations_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.sqlite"
            conn = init_db(db_path)
            ev = StageEvaluation(
                session_id="sess1",
                stage_index=1,
                evaluator_id="P01",
                common_q1=5,
                common_q2=6,
                common_q3=5,
                common_q4=4,
                specific_q1=6,
                specific_q2=5,
                specific_q3=4,
                overall_comment="ok",
            )
            save_stage_evaluation(conn, ev)
            loaded = get_stage_evaluation(conn, "sess1", 1, "P01")
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.common_q2, 6)

            anns = [
                ResponseAnnotation(
                    session_id="sess1",
                    stage_index=1,
                    call_id="c1",
                    evaluator_id="P01",
                    issue_tags=["bad_timing"],
                    comment="ずれた",
                ),
                ResponseAnnotation(
                    session_id="sess1",
                    stage_index=1,
                    call_id="c2",
                    evaluator_id="P01",
                    issue_tags=["too_long", "hallucinated_content"],
                    comment="長い",
                ),
            ]
            replace_response_annotations(
                conn,
                session_id="sess1",
                stage_index=1,
                evaluator_id="P01",
                annotations=anns,
            )
            loaded_anns = get_response_annotations(
                conn,
                session_id="sess1",
                stage_index=1,
                evaluator_id="P01",
            )
            self.assertEqual(len(loaded_anns), 2)
            self.assertEqual(loaded_anns[0].call_id, "c1")


if __name__ == "__main__":
    unittest.main()
