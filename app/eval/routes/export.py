import csv
import io
import json
from fastapi import APIRouter
from fastapi.responses import Response, StreamingResponse

router = APIRouter()


@router.get("/export/csv")
async def export_csv():
    from app.eval.main import conn
    from app.eval.db import get_all_evaluations

    rows = get_all_evaluations(conn)
    if not rows:
        return Response("No data", media_type="text/plain")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=evaluations.csv"},
    )


@router.get("/export/json")
async def export_json():
    from app.eval.main import conn
    from app.eval.db import get_all_evaluations

    rows = get_all_evaluations(conn)
    return Response(
        content=json.dumps(rows, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=evaluations.json"},
    )
