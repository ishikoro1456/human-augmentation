import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set

from .models import Evaluation, ResponseAnnotation, StageEvaluation

SCHEMA_EVALUATIONS = """
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    call_id TEXT NOT NULL,
    evaluator_id TEXT NOT NULL,
    appropriateness INTEGER NOT NULL CHECK(appropriateness BETWEEN 1 AND 7),
    would_have_sent TEXT NOT NULL DEFAULT '',
    issues TEXT NOT NULL DEFAULT '[]',
    comment TEXT NOT NULL DEFAULT '',
    time_spent_ms INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(experiment_id, call_id, evaluator_id)
)
"""

SCHEMA_STAGE_EVALUATIONS = """
CREATE TABLE IF NOT EXISTS stage_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    stage_index INTEGER NOT NULL,
    evaluator_id TEXT NOT NULL,
    common_q1 INTEGER NOT NULL CHECK(common_q1 BETWEEN 1 AND 7),
    common_q2 INTEGER NOT NULL CHECK(common_q2 BETWEEN 1 AND 7),
    common_q3 INTEGER NOT NULL CHECK(common_q3 BETWEEN 1 AND 7),
    common_q4 INTEGER NOT NULL CHECK(common_q4 BETWEEN 1 AND 7),
    specific_q1 INTEGER NOT NULL CHECK(specific_q1 BETWEEN 1 AND 7),
    specific_q2 INTEGER NOT NULL CHECK(specific_q2 BETWEEN 1 AND 7),
    specific_q3 INTEGER NOT NULL CHECK(specific_q3 BETWEEN 1 AND 7),
    overall_comment TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(session_id, stage_index, evaluator_id)
)
"""

SCHEMA_RESPONSE_ANNOTATIONS = """
CREATE TABLE IF NOT EXISTS response_annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    stage_index INTEGER NOT NULL,
    call_id TEXT NOT NULL,
    evaluator_id TEXT NOT NULL,
    issue_tags_json TEXT NOT NULL DEFAULT '[]',
    comment TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(session_id, stage_index, call_id, evaluator_id)
)
"""


def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA_EVALUATIONS)
    conn.execute(SCHEMA_STAGE_EVALUATIONS)
    conn.execute(SCHEMA_RESPONSE_ANNOTATIONS)
    conn.commit()
    return conn


# ── Existing per-decision evaluation APIs ───────────────────────────────
def save_evaluation(conn: sqlite3.Connection, ev: Evaluation) -> None:
    conn.execute(
        """
        INSERT INTO evaluations
            (experiment_id, call_id, evaluator_id, appropriateness,
             would_have_sent, issues, comment, time_spent_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(experiment_id, call_id, evaluator_id) DO UPDATE SET
            appropriateness = excluded.appropriateness,
            would_have_sent = excluded.would_have_sent,
            issues = excluded.issues,
            comment = excluded.comment,
            time_spent_ms = excluded.time_spent_ms,
            created_at = datetime('now')
        """,
        (
            ev.experiment_id,
            ev.call_id,
            ev.evaluator_id,
            ev.appropriateness,
            ev.would_have_sent,
            json.dumps(ev.issues, ensure_ascii=False),
            ev.comment,
            ev.time_spent_ms,
        ),
    )
    conn.commit()


def get_evaluation(
    conn: sqlite3.Connection,
    experiment_id: str,
    call_id: str,
    evaluator_id: str,
) -> Optional[Evaluation]:
    row = conn.execute(
        "SELECT * FROM evaluations WHERE experiment_id=? AND call_id=? AND evaluator_id=?",
        (experiment_id, call_id, evaluator_id),
    ).fetchone()
    if not row:
        return None
    return _row_to_eval(row)


def get_evaluated_call_ids(
    conn: sqlite3.Connection,
    experiment_id: str,
    evaluator_id: str,
) -> Set[str]:
    rows = conn.execute(
        "SELECT call_id FROM evaluations WHERE experiment_id=? AND evaluator_id=?",
        (experiment_id, evaluator_id),
    ).fetchall()
    return {r["call_id"] for r in rows}


def get_session_stats(
    conn: sqlite3.Connection,
    experiment_id: str,
    evaluator_id: str,
) -> Dict:
    rows = conn.execute(
        "SELECT appropriateness, time_spent_ms FROM evaluations WHERE experiment_id=? AND evaluator_id=?",
        (experiment_id, evaluator_id),
    ).fetchall()
    if not rows:
        return {"count": 0, "distribution": {}, "avg_time_ms": 0}
    dist = {}
    total_time = 0
    for r in rows:
        dist[r["appropriateness"]] = dist.get(r["appropriateness"], 0) + 1
        total_time += r["time_spent_ms"]
    return {
        "count": len(rows),
        "distribution": dist,
        "avg_time_ms": total_time // len(rows) if rows else 0,
    }


# ── Stage-level evaluation APIs ──────────────────────────────────────────
def save_stage_evaluation(conn: sqlite3.Connection, ev: StageEvaluation) -> None:
    conn.execute(
        """
        INSERT INTO stage_evaluations
            (session_id, stage_index, evaluator_id,
             common_q1, common_q2, common_q3, common_q4,
             specific_q1, specific_q2, specific_q3,
             overall_comment)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id, stage_index, evaluator_id) DO UPDATE SET
            common_q1 = excluded.common_q1,
            common_q2 = excluded.common_q2,
            common_q3 = excluded.common_q3,
            common_q4 = excluded.common_q4,
            specific_q1 = excluded.specific_q1,
            specific_q2 = excluded.specific_q2,
            specific_q3 = excluded.specific_q3,
            overall_comment = excluded.overall_comment,
            created_at = datetime('now')
        """,
        (
            ev.session_id,
            ev.stage_index,
            ev.evaluator_id,
            ev.common_q1,
            ev.common_q2,
            ev.common_q3,
            ev.common_q4,
            ev.specific_q1,
            ev.specific_q2,
            ev.specific_q3,
            ev.overall_comment,
        ),
    )
    conn.commit()


def get_stage_evaluation(
    conn: sqlite3.Connection,
    session_id: str,
    stage_index: int,
    evaluator_id: str,
) -> Optional[StageEvaluation]:
    row = conn.execute(
        "SELECT * FROM stage_evaluations WHERE session_id=? AND stage_index=? AND evaluator_id=?",
        (session_id, int(stage_index), evaluator_id),
    ).fetchone()
    if not row:
        return None
    return _row_to_stage_eval(row)


def get_stage_evaluations_by_session(
    conn: sqlite3.Connection,
    session_id: str,
    evaluator_id: str,
) -> List[StageEvaluation]:
    rows = conn.execute(
        "SELECT * FROM stage_evaluations WHERE session_id=? AND evaluator_id=? ORDER BY stage_index",
        (session_id, evaluator_id),
    ).fetchall()
    return [_row_to_stage_eval(r) for r in rows]


def save_response_annotation(conn: sqlite3.Connection, ann: ResponseAnnotation) -> None:
    conn.execute(
        """
        INSERT INTO response_annotations
            (session_id, stage_index, call_id, evaluator_id, issue_tags_json, comment)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id, stage_index, call_id, evaluator_id) DO UPDATE SET
            issue_tags_json = excluded.issue_tags_json,
            comment = excluded.comment,
            created_at = datetime('now')
        """,
        (
            ann.session_id,
            int(ann.stage_index),
            ann.call_id,
            ann.evaluator_id,
            json.dumps(list(ann.issue_tags), ensure_ascii=False),
            ann.comment,
        ),
    )


def replace_response_annotations(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    stage_index: int,
    evaluator_id: str,
    annotations: List[ResponseAnnotation],
) -> None:
    conn.execute(
        "DELETE FROM response_annotations WHERE session_id=? AND stage_index=? AND evaluator_id=?",
        (session_id, int(stage_index), evaluator_id),
    )
    for ann in annotations:
        save_response_annotation(conn, ann)
    conn.commit()


def get_response_annotations(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    stage_index: int,
    evaluator_id: str,
) -> List[ResponseAnnotation]:
    rows = conn.execute(
        """
        SELECT * FROM response_annotations
        WHERE session_id=? AND stage_index=? AND evaluator_id=?
        ORDER BY created_at, call_id
        """,
        (session_id, int(stage_index), evaluator_id),
    ).fetchall()
    return [_row_to_response_annotation(r) for r in rows]


# ── Export helpers ───────────────────────────────────────────────────────
def get_all_evaluations(conn: sqlite3.Connection) -> List[Dict]:
    rows = conn.execute("SELECT * FROM evaluations ORDER BY created_at").fetchall()
    return [dict(r) for r in rows]


def get_all_stage_evaluations(conn: sqlite3.Connection) -> List[Dict]:
    rows = conn.execute("SELECT * FROM stage_evaluations ORDER BY created_at").fetchall()
    return [dict(r) for r in rows]


def get_all_response_annotations(conn: sqlite3.Connection) -> List[Dict]:
    rows = conn.execute("SELECT * FROM response_annotations ORDER BY created_at").fetchall()
    return [dict(r) for r in rows]


# ── Row converters ───────────────────────────────────────────────────────
def _row_to_eval(row: sqlite3.Row) -> Evaluation:
    return Evaluation(
        id=row["id"],
        experiment_id=row["experiment_id"],
        call_id=row["call_id"],
        evaluator_id=row["evaluator_id"],
        appropriateness=row["appropriateness"],
        would_have_sent=row["would_have_sent"],
        issues=json.loads(row["issues"]) if row["issues"] else [],
        comment=row["comment"],
        time_spent_ms=row["time_spent_ms"],
        created_at=row["created_at"],
    )


def _row_to_stage_eval(row: sqlite3.Row) -> StageEvaluation:
    return StageEvaluation(
        id=row["id"],
        session_id=row["session_id"],
        stage_index=int(row["stage_index"]),
        evaluator_id=row["evaluator_id"],
        common_q1=int(row["common_q1"]),
        common_q2=int(row["common_q2"]),
        common_q3=int(row["common_q3"]),
        common_q4=int(row["common_q4"]),
        specific_q1=int(row["specific_q1"]),
        specific_q2=int(row["specific_q2"]),
        specific_q3=int(row["specific_q3"]),
        overall_comment=row["overall_comment"],
        created_at=row["created_at"],
    )


def _row_to_response_annotation(row: sqlite3.Row) -> ResponseAnnotation:
    issue_tags_raw = row["issue_tags_json"]
    issue_tags = []
    if isinstance(issue_tags_raw, str) and issue_tags_raw:
        try:
            parsed = json.loads(issue_tags_raw)
            if isinstance(parsed, list):
                issue_tags = [str(x) for x in parsed if str(x).strip()]
        except Exception:
            issue_tags = []

    return ResponseAnnotation(
        id=row["id"],
        session_id=row["session_id"],
        stage_index=int(row["stage_index"]),
        call_id=row["call_id"],
        evaluator_id=row["evaluator_id"],
        issue_tags=issue_tags,
        comment=row["comment"],
        created_at=row["created_at"],
    )
