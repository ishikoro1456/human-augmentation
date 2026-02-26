import datetime
from fastapi import APIRouter, Cookie, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates: Jinja2Templates = None


def setup(t: Jinja2Templates) -> None:
    global templates
    templates = t


def _fmt_ts(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")


GESTURE_LABELS = {
    "nod": ("うなずき", "↓↑"),
    "shake": ("首振り", "←→"),
    "other": ("その他", "〜"),
}

STRENGTH_LABELS = {
    0: "",
    1: "弱",
    3: "中",
    5: "強",
}

ISSUE_OPTIONS = [
    ("wrong_direction", "肯定/否定が逆"),
    ("too_strong", "強すぎる"),
    ("too_weak", "弱すぎる"),
    ("bad_timing", "タイミングが悪い"),
    ("should_have_stayed_silent", "無言が良かった"),
    ("should_have_responded", "何か返すべきだった"),
]


@router.get("/evaluate/{experiment_id}/{index}", response_class=HTMLResponse)
async def evaluate_page(
    request: Request,
    experiment_id: str,
    index: int,
    evaluator_id: str = Cookie(default=""),
):
    from app.eval.main import loader, conn
    from app.eval.db import get_evaluation, get_evaluated_call_ids

    session = loader.get_session(experiment_id)
    if not session:
        return HTMLResponse("Session not found", status_code=404)

    if index < 0 or index >= len(session.decisions):
        return RedirectResponse(f"/evaluate/{experiment_id}/complete")

    decision = session.decisions[index]
    timeline = loader.build_timeline(session)
    evaluated_ids = get_evaluated_call_ids(conn, experiment_id, evaluator_id)
    existing_eval = get_evaluation(conn, experiment_id, decision.call_id, evaluator_id)
    catalog = loader.get_catalog()

    return templates.TemplateResponse(
        "evaluate.html",
        {
            "request": request,
            "session": session,
            "decision": decision,
            "timeline": timeline,
            "evaluated_ids": evaluated_ids,
            "existing_eval": existing_eval,
            "evaluator_id": evaluator_id,
            "catalog": catalog,
            "progress": {
                "current": index + 1,
                "total": len(session.decisions),
                "evaluated": len(evaluated_ids),
            },
            "prev_index": index - 1 if index > 0 else None,
            "next_index": index + 1 if index + 1 < len(session.decisions) else None,
            "gesture_labels": GESTURE_LABELS,
            "strength_labels": STRENGTH_LABELS,
            "issue_options": ISSUE_OPTIONS,
            "fmt_ts": _fmt_ts,
        },
    )


@router.post("/evaluate/{experiment_id}/{index}", response_class=HTMLResponse)
async def submit_evaluation(
    request: Request,
    experiment_id: str,
    index: int,
    evaluator_id: str = Cookie(default=""),
):
    from app.eval.main import loader, conn
    from app.eval.db import save_evaluation
    from app.eval.models import Evaluation

    session = loader.get_session(experiment_id)
    if not session or index >= len(session.decisions):
        return HTMLResponse("Not found", status_code=404)

    decision = session.decisions[index]
    form = await request.form()

    appropriateness_raw = form.get("appropriateness")
    if not appropriateness_raw:
        # No rating given — bounce back
        return RedirectResponse(
            f"/evaluate/{experiment_id}/{index}", status_code=303
        )

    issues = form.getlist("issues")
    ev = Evaluation(
        experiment_id=experiment_id,
        call_id=decision.call_id,
        evaluator_id=evaluator_id or form.get("evaluator_id", "anonymous"),
        appropriateness=int(appropriateness_raw),
        would_have_sent=form.get("would_have_sent", ""),
        issues=issues,
        comment=form.get("comment", ""),
        time_spent_ms=int(form.get("time_spent_ms", 0) or 0),
    )
    save_evaluation(conn, ev)

    next_index = index + 1
    if next_index >= len(session.decisions):
        return RedirectResponse(
            f"/evaluate/{experiment_id}/complete", status_code=303
        )
    return RedirectResponse(
        f"/evaluate/{experiment_id}/{next_index}", status_code=303
    )


@router.get("/evaluate/{experiment_id}/complete", response_class=HTMLResponse)
async def complete_page(
    request: Request,
    experiment_id: str,
    evaluator_id: str = Cookie(default=""),
):
    from app.eval.main import loader, conn
    from app.eval.db import get_session_stats

    session = loader.get_session(experiment_id)
    if not session:
        return HTMLResponse("Session not found", status_code=404)

    stats = get_session_stats(conn, experiment_id, evaluator_id)

    return templates.TemplateResponse(
        "complete.html",
        {
            "request": request,
            "session": session,
            "stats": stats,
            "evaluator_id": evaluator_id,
        },
    )
