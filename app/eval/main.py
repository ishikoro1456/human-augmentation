from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI

BASE_DIR = Path(__file__).parent.parent.parent
load_dotenv(BASE_DIR / ".env")

from .config import CATALOG_PATH, DB_PATH, TRACE_PATH  # noqa: E402
from .db import init_db  # noqa: E402
from .trace_loader import TraceLoader  # noqa: E402
from .routes import sessions as sessions_router  # noqa: E402
from .routes import evaluate as evaluate_router  # noqa: E402
from .routes import export as export_router  # noqa: E402
from .routes import experiment as experiment_router  # noqa: E402

loader: TraceLoader = None
conn = None
oai_client: OpenAI = None
catalog = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global loader, conn, oai_client, catalog
    loader = TraceLoader(TRACE_PATH, CATALOG_PATH)
    loader.load()
    conn = init_db(DB_PATH)
    oai_client = OpenAI()
    from app.core.catalog import load_catalog
    catalog = load_catalog(CATALOG_PATH)
    yield
    if conn:
        conn.close()


app = FastAPI(title="Backchannel Evaluation", lifespan=lifespan)

_templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))

sessions_router.setup(templates)
evaluate_router.setup(templates)
experiment_router.setup(templates)

app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "static")),
    name="static",
)

app.include_router(sessions_router.router)
app.include_router(evaluate_router.router)
app.include_router(export_router.router)
app.include_router(experiment_router.router)
