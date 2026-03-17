"""実験モード: 4段階リアルタイム相槌 + 段階レビュー"""

import json
from pathlib import Path

from fastapi import APIRouter, Cookie, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.eval.models import ResponseAnnotation, StageEvaluation
from app.eval.stage_policy import ISSUE_TAG_OPTIONS, STAGE_CONFIGS, get_stage_config

router = APIRouter()
templates: Jinja2Templates = None

SCRIPTS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "scripts"

GESTURE_LABELS = {
    "nod": ("うなずき", "↓↑"),
    "shake": ("首振り", "←→"),
    "other": ("その他", "〜"),
}

COMMON_QUESTION_LABELS = {
    "common_q1": "自分の意図が伝わったと感じましたか",
    "common_q2": "タイミングは自然でしたか",
    "common_q3": "返答は受け入れやすかったですか",
    "common_q4": "この段階の使い勝手は良かったですか",
}

SPECIFIC_LABELS = {
    "yesno_direction_correctness": "YES/NOの向きは正確でしたか",
    "unnecessary_response_rate": "不要な返答は少なかったですか",
    "missed_response_rate": "返すべき場面の取りこぼしは少なかったですか",
    "intensity_match": "強弱は意図に合っていましたか",
    "intensity_stability": "同じ強さで一貫して返せていましたか",
    "intensity_controllability": "強さをコントロールできていると感じましたか",
    "micro_phrase_naturalness": "超短フレーズは自然でしたか",
    "brevity_appropriateness": "短さは場面に合っていましたか",
    "intent_transfer_clarity": "短い返答でも意図は伝わりましたか",
    "contextual_fit": "文脈との整合性はありましたか",
    "sentence_naturalness": "1文の表現は自然でしたか",
    "information_amount_appropriateness": "情報量は適切でしたか",
}


def setup(t: Jinja2Templates) -> None:
    global templates
    templates = t


def _load_scripts() -> list[dict]:
    scripts = []
    for path in sorted(SCRIPTS_DIR.glob("*.json")):
        try:
            scripts.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    return scripts


def _load_script(script_id: str) -> dict | None:
    for path in SCRIPTS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("id") == script_id:
                return data
        except Exception:
            pass
    return None


def _specific_question_labels(stage_index: int) -> dict[str, str]:
    conf = get_stage_config(stage_index)
    out: dict[str, str] = {}
    for i, key in enumerate(conf.specific_metrics, start=1):
        out[f"specific_q{i}"] = SPECIFIC_LABELS.get(key, key)
    return out


# ── Script list ───────────────────────────────────────────────────────────


@router.get("/experiment", response_class=HTMLResponse)
async def experiment_list(request: Request, evaluator_id: str = Cookie(default="")):
    scripts = _load_scripts()
    return templates.TemplateResponse(
        "experiment_list.html",
        {
            "request": request,
            "scripts": scripts,
            "evaluator_id": evaluator_id,
            "stages": STAGE_CONFIGS,
        },
    )


# ── Create session ────────────────────────────────────────────────────────


@router.post("/experiment/{script_id}/create")
async def create_experiment(
    request: Request,
    script_id: str,
    evaluator_id: str = Cookie(default=""),
):
    from app.eval.main import BASE_DIR, catalog, oai_client
    from app.eval.experiment import create_session

    script = _load_script(script_id)
    if not script:
        return HTMLResponse("Script not found", status_code=404)

    form = await request.form()
    imu_port = (form.get("imu_port") or "").strip() or None
    imu_baud = int(form.get("imu_baud") or 115200)
    protocol = str(form.get("protocol") or "imu_4stage_v1").strip() or "imu_4stage_v1"

    sess = create_session(
        script=script,
        oai_client=oai_client,
        catalog=catalog,
        catalog_path=BASE_DIR / "data" / "catalog.tsv",
        backchannel_dir=BASE_DIR / "data" / "backchannel",
        tts_cache_dir=BASE_DIR / "data" / "tts_cache",
        protocol=protocol,
        imu_port=imu_port,
        imu_baud=imu_baud,
    )
    if imu_port:
        sess.start_imu()

    resp = RedirectResponse(f"/experiment/run/{sess.session_id}", status_code=303)
    if evaluator_id:
        resp.set_cookie("evaluator_id", evaluator_id, max_age=86400 * 30)
    return resp


# ── Run page ──────────────────────────────────────────────────────────────


@router.get("/experiment/run/{session_id}", response_class=HTMLResponse)
async def run_page(
    request: Request,
    session_id: str,
    evaluator_id: str = Cookie(default=""),
):
    from app.eval.experiment import get_session

    sess = get_session(session_id)
    if not sess:
        return HTMLResponse("Session not found", status_code=404)

    snap = sess.snapshot()
    if snap["state"] == "stage_review":
        return RedirectResponse(
            f"/experiment/run/{session_id}/stage/{snap['stage_index']}/review",
            status_code=303,
        )
    if snap["state"] == "done":
        return RedirectResponse(f"/experiment/run/{session_id}/finish", status_code=303)

    return templates.TemplateResponse(
        "experiment_run.html",
        {
            "request": request,
            "snap": snap,
            "sentences": sess.sentences,
            "evaluator_id": evaluator_id,
            "gesture_labels": GESTURE_LABELS,
            "stages": STAGE_CONFIGS,
        },
    )


# ── Advance (play next sentence) ──────────────────────────────────────────


@router.post("/experiment/run/{session_id}/advance", response_class=HTMLResponse)
async def advance(session_id: str):
    from app.eval.experiment import get_session

    sess = get_session(session_id)
    if not sess:
        return HTMLResponse("Session not found", status_code=404)
    sess.advance()
    snap = sess.snapshot()
    if snap["state"] == "stage_review":
        return RedirectResponse(
            f"/experiment/run/{session_id}/stage/{snap['stage_index']}/review",
            status_code=303,
        )
    return HTMLResponse("", status_code=204)


# ── Mock gesture trigger ──────────────────────────────────────────────────


@router.post("/experiment/run/{session_id}/gesture/{hint}", response_class=HTMLResponse)
async def gesture(session_id: str, hint: str):
    from app.eval.experiment import get_session

    sess = get_session(session_id)
    if not sess or hint not in ("nod", "shake"):
        return HTMLResponse("", status_code=204)
    sess.inject_gesture(hint)
    return HTMLResponse("", status_code=204)


# ── Poll (htmx polling target) ────────────────────────────────────────────


@router.get("/experiment/run/{session_id}/poll", response_class=HTMLResponse)
async def poll(request: Request, session_id: str):
    from app.eval.experiment import get_session

    sess = get_session(session_id)
    if not sess:
        return HTMLResponse("<div id='rt-panel'></div>")

    snap = sess.snapshot()
    next_url = ""
    if snap["state"] == "stage_review":
        next_url = f"/experiment/run/{session_id}/stage/{snap['stage_index']}/review"
    elif snap["state"] == "done":
        next_url = f"/experiment/run/{session_id}/finish"

    return templates.TemplateResponse(
        "partials/experiment_rt.html",
        {
            "request": request,
            "snap": snap,
            "session_id": session_id,
            "gesture_labels": GESTURE_LABELS,
            "next_url": next_url,
        },
    )


# ── Stage review ──────────────────────────────────────────────────────────


@router.get("/experiment/run/{session_id}/stage/{stage_index}/review", response_class=HTMLResponse)
async def stage_review_page(
    request: Request,
    session_id: str,
    stage_index: int,
    evaluator_id: str = Cookie(default=""),
):
    from app.eval.db import get_response_annotations, get_stage_evaluation
    from app.eval.experiment import get_session
    from app.eval.main import conn

    sess = get_session(session_id)
    if not sess:
        return HTMLResponse("Session not found", status_code=404)

    snap = sess.snapshot()
    stage_idx = int(stage_index)
    if stage_idx < 0 or stage_idx >= snap["stage_total"]:
        return HTMLResponse("Stage not found", status_code=404)

    # 現在レビュー中の段階以外は、既に完了しているときのみ閲覧可
    if stage_idx != snap["stage_index"] and snap["state"] != "done":
        return RedirectResponse(
            f"/experiment/run/{session_id}/stage/{snap['stage_index']}/review",
            status_code=303,
        )

    decisions = sess.get_stage_decisions(stage_idx)
    existing_eval = None
    existing_eval_map: dict[str, int | str] = {}
    annotations_map: dict[str, dict] = {}
    if evaluator_id:
        existing_eval = get_stage_evaluation(conn, session_id, stage_idx, evaluator_id)
        if existing_eval is not None:
            existing_eval_map = {
                "common_q1": existing_eval.common_q1,
                "common_q2": existing_eval.common_q2,
                "common_q3": existing_eval.common_q3,
                "common_q4": existing_eval.common_q4,
                "specific_q1": existing_eval.specific_q1,
                "specific_q2": existing_eval.specific_q2,
                "specific_q3": existing_eval.specific_q3,
                "overall_comment": existing_eval.overall_comment,
            }
        anns = get_response_annotations(
            conn,
            session_id=session_id,
            stage_index=stage_idx,
            evaluator_id=evaluator_id,
        )
        annotations_map = {
            a.call_id: {
                "issue_tags": list(a.issue_tags),
                "comment": a.comment,
            }
            for a in anns
        }

    return templates.TemplateResponse(
        "experiment_stage_review.html",
        {
            "request": request,
            "snap": snap,
            "session_id": session_id,
            "stage": get_stage_config(stage_idx),
            "stages": STAGE_CONFIGS,
            "stage_index": stage_idx,
            "decisions": decisions,
            "evaluator_id": evaluator_id,
            "common_labels": COMMON_QUESTION_LABELS,
            "specific_labels": _specific_question_labels(stage_idx),
            "issue_options": ISSUE_TAG_OPTIONS,
            "existing_eval": existing_eval,
            "existing_eval_map": existing_eval_map,
            "annotations_map": annotations_map,
            "gesture_labels": GESTURE_LABELS,
        },
    )


@router.post("/experiment/run/{session_id}/stage/{stage_index}/review")
async def submit_stage_review(
    request: Request,
    session_id: str,
    stage_index: int,
    evaluator_id: str = Cookie(default=""),
):
    from app.eval.db import replace_response_annotations, save_stage_evaluation
    from app.eval.experiment import get_session
    from app.eval.main import conn

    sess = get_session(session_id)
    if not sess:
        return HTMLResponse("Session not found", status_code=404)

    form = await request.form()
    evaluator = str(evaluator_id or form.get("evaluator_id", "anonymous")).strip() or "anonymous"

    def _q(name: str) -> int:
        v = int(form.get(name) or 1)
        return max(1, min(7, v))

    ev = StageEvaluation(
        session_id=session_id,
        stage_index=int(stage_index),
        evaluator_id=evaluator,
        common_q1=_q("common_q1"),
        common_q2=_q("common_q2"),
        common_q3=_q("common_q3"),
        common_q4=_q("common_q4"),
        specific_q1=_q("specific_q1"),
        specific_q2=_q("specific_q2"),
        specific_q3=_q("specific_q3"),
        overall_comment=str(form.get("overall_comment", "")),
    )
    save_stage_evaluation(conn, ev)

    annotations: list[ResponseAnnotation] = []
    for d in sess.get_stage_decisions(stage_index):
        tags = [str(x) for x in form.getlist(f"ann_{d.call_id}_issues") if str(x).strip()]
        comment = str(form.get(f"ann_{d.call_id}_comment", "")).strip()
        if not tags and not comment:
            continue
        annotations.append(
            ResponseAnnotation(
                session_id=session_id,
                stage_index=int(stage_index),
                call_id=d.call_id,
                evaluator_id=evaluator,
                issue_tags=tags,
                comment=comment,
            )
        )

    replace_response_annotations(
        conn,
        session_id=session_id,
        stage_index=int(stage_index),
        evaluator_id=evaluator,
        annotations=annotations,
    )
    sess.mark_stage_review_submitted(stage_index=int(stage_index), evaluator_id=evaluator)

    moved = sess.advance_stage()
    snap = sess.snapshot()
    if not moved and snap["state"] == "done":
        return RedirectResponse(f"/experiment/run/{session_id}/finish", status_code=303)

    return RedirectResponse(f"/experiment/run/{session_id}", status_code=303)


# ── Finish page ───────────────────────────────────────────────────────────


@router.get("/experiment/run/{session_id}/finish", response_class=HTMLResponse)
async def finish_page(
    request: Request,
    session_id: str,
    evaluator_id: str = Cookie(default=""),
):
    from app.eval.db import get_response_annotations, get_stage_evaluations_by_session
    from app.eval.experiment import get_session
    from app.eval.main import conn

    sess = get_session(session_id)
    if not sess:
        return HTMLResponse("Session not found", status_code=404)

    snap = sess.snapshot()
    evaluator = str(evaluator_id).strip()
    stage_evals = []
    ann_by_stage: dict[int, int] = {}
    if evaluator:
        stage_evals = get_stage_evaluations_by_session(conn, session_id, evaluator)
        for idx in range(len(STAGE_CONFIGS)):
            anns = get_response_annotations(
                conn,
                session_id=session_id,
                stage_index=idx,
                evaluator_id=evaluator,
            )
            ann_by_stage[idx] = len(anns)

    stage_eval_map = {ev.stage_index: ev for ev in stage_evals}

    return templates.TemplateResponse(
        "experiment_finish.html",
        {
            "request": request,
            "snap": snap,
            "session_id": session_id,
            "stages": STAGE_CONFIGS,
            "stage_eval_map": stage_eval_map,
            "annotation_counts": ann_by_stage,
            "common_labels": COMMON_QUESTION_LABELS,
            "evaluator_id": evaluator_id,
        },
    )
