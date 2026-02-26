import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set

from .models import Evaluation

SCHEMA = """
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


def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    conn.commit()
    return conn


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
    conn: sqlite3.Connection, experiment_id: str, call_id: str, evaluator_id: str
) -> Optional[Evaluation]:
    row = conn.execute(
        "SELECT * FROM evaluations WHERE experiment_id=? AND call_id=? AND evaluator_id=?",
        (experiment_id, call_id, evaluator_id),
    ).fetchone()
    if not row:
        return None
    return _row_to_eval(row)


def get_evaluated_call_ids(
    conn: sqlite3.Connection, experiment_id: str, evaluator_id: str
) -> Set[str]:
    rows = conn.execute(
        "SELECT call_id FROM evaluations WHERE experiment_id=? AND evaluator_id=?",
        (experiment_id, evaluator_id),
    ).fetchall()
    return {r["call_id"] for r in rows}


def get_session_stats(
    conn: sqlite3.Connection, experiment_id: str, evaluator_id: str
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


def get_all_evaluations(conn: sqlite3.Connection) -> List[Dict]:
    rows = conn.execute("SELECT * FROM evaluations ORDER BY created_at").fetchall()
    return [dict(r) for r in rows]


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
