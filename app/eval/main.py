from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import CATALOG_PATH, DB_PATH, TRACE_PATH
from .db import init_db
from .trace_loader import TraceLoader
from .routes import sessions as sessions_router
from .routes import evaluate as evaluate_router
from .routes import export as export_router

loader: TraceLoader = None
conn = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global loader, conn
    loader = TraceLoader(TRACE_PATH, CATALOG_PATH)
    loader.load()
    conn = init_db(DB_PATH)
    yield
    if conn:
        conn.close()


app = FastAPI(title="Backchannel Evaluation", lifespan=lifespan)

_templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))

sessions_router.setup(templates)
evaluate_router.setup(templates)

app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "static")),
    name="static",
)

app.include_router(sessions_router.router)
app.include_router(evaluate_router.router)
app.include_router(export_router.router)
