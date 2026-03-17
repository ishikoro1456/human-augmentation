import csv
import io
import json

from fastapi import APIRouter
from fastapi.responses import Response

router = APIRouter()


def _to_csv(rows: list[dict], *, filename: str) -> Response:
    if not rows:
        return Response("No data", media_type="text/plain")

    all_keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=all_keys)
    writer.writeheader()
    writer.writerows(rows)

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/export/csv")
async def export_csv() -> Response:
    from app.eval.main import conn
    from app.eval.db import get_all_evaluations

    rows = get_all_evaluations(conn)
    return _to_csv(rows, filename="evaluations.csv")


@router.get("/export/stage-csv")
async def export_stage_csv() -> Response:
    from app.eval.main import conn
    from app.eval.db import get_all_stage_evaluations

    rows = get_all_stage_evaluations(conn)
    return _to_csv(rows, filename="stage_evaluations.csv")


@router.get("/export/annotations-csv")
async def export_annotations_csv() -> Response:
    from app.eval.main import conn
    from app.eval.db import get_all_response_annotations

    rows = get_all_response_annotations(conn)
    return _to_csv(rows, filename="response_annotations.csv")


@router.get("/export/json")
async def export_json() -> Response:
    from app.eval.main import conn
    from app.eval.db import (
        get_all_evaluations,
        get_all_response_annotations,
        get_all_stage_evaluations,
    )

    payload = {
        "evaluations": get_all_evaluations(conn),
        "stage_evaluations": get_all_stage_evaluations(conn),
        "response_annotations": get_all_response_annotations(conn),
    }
    return Response(
        content=json.dumps(payload, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=evaluations_all.json"},
    )
