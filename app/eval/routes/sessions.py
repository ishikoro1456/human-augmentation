import datetime
from fastapi import APIRouter, Cookie, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates: Jinja2Templates = None  # injected from main.py


def _fmt_ts(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%Y/%m/%d %H:%M")


def setup(t: Jinja2Templates) -> None:
    global templates
    templates = t


@router.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse("/sessions")


@router.get("/sessions", response_class=HTMLResponse)
async def session_list(
    request: Request,
    evaluator_id: str = Cookie(default=""),
):
    from app.eval.main import loader, conn
    from app.eval.db import get_evaluated_call_ids

    summaries = loader.get_sessions()

    # Attach completion info
    session_infos = []
    for s in summaries:
        evaluated = (
            len(get_evaluated_call_ids(conn, s.experiment_id, evaluator_id))
            if evaluator_id
            else 0
        )
        session_infos.append(
            {
                "experiment_id": s.experiment_id,
                "date_str": _fmt_ts(s.start_ts),
                "mode": s.mode,
                "model": s.model,
                "total": s.decision_count,
                "evaluated": evaluated,
                "pct": int(100 * evaluated / s.decision_count) if s.decision_count else 0,
            }
        )

    return templates.TemplateResponse(
        "session_list.html",
        {
            "request": request,
            "sessions": session_infos,
            "evaluator_id": evaluator_id,
        },
    )


@router.post("/sessions", response_class=HTMLResponse)
async def set_evaluator(
    request: Request,
    response: Response,
):
    form = await request.form()
    evaluator_id = (form.get("evaluator_id") or "").strip()
    resp = RedirectResponse("/sessions", status_code=303)
    if evaluator_id:
        resp.set_cookie(key="evaluator_id", value=evaluator_id, max_age=86400 * 30)
    return resp
