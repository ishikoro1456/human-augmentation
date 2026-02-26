"""実験モード: 台本TTS再生 + リアルタイム相槌判断"""
import json
from pathlib import Path

from fastapi import APIRouter, Cookie, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates: Jinja2Templates = None

SCRIPTS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "scripts"

GESTURE_LABELS = {
    "nod": ("うなずき", "↓↑"),
    "shake": ("首振り", "←→"),
    "other": ("その他", "〜"),
}
STRENGTH_LABELS = {0: "", 1: "弱", 3: "中", 5: "強"}


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


# ── Script list ───────────────────────────────────────────────────────────

@router.get("/experiment", response_class=HTMLResponse)
async def experiment_list(request: Request, evaluator_id: str = Cookie(default="")):
    scripts = _load_scripts()
    return templates.TemplateResponse(
        "experiment_list.html",
        {"request": request, "scripts": scripts, "evaluator_id": evaluator_id},
    )


# ── Create session ────────────────────────────────────────────────────────

@router.post("/experiment/{script_id}/create")
async def create_experiment(
    request: Request,
    script_id: str,
    evaluator_id: str = Cookie(default=""),
):
    from app.eval.main import oai_client, catalog, BASE_DIR
    from app.eval.experiment import create_session

    script = _load_script(script_id)
    if not script:
        return HTMLResponse("Script not found", status_code=404)

    form = await request.form()
    imu_port = (form.get("imu_port") or "").strip() or None
    imu_baud = int(form.get("imu_baud") or 115200)

    sess = create_session(
        script=script,
        oai_client=oai_client,
        catalog=catalog,
        catalog_path=BASE_DIR / "data" / "catalog.tsv",
        backchannel_dir=BASE_DIR / "data" / "backchannel",
        tts_cache_dir=BASE_DIR / "data" / "tts_cache",
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
    return templates.TemplateResponse(
        "experiment_run.html",
        {
            "request": request,
            "snap": snap,
            "sentences": sess.sentences,
            "evaluator_id": evaluator_id,
            "gesture_labels": GESTURE_LABELS,
            "strength_labels": STRENGTH_LABELS,
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
    return templates.TemplateResponse(
        "partials/experiment_rt.html",
        {
            "request": request,
            "snap": snap,
            "session_id": session_id,
            "gesture_labels": GESTURE_LABELS,
            "strength_labels": STRENGTH_LABELS,
        },
    )
