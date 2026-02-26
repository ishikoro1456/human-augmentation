from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
TRACE_PATH = BASE_DIR / "data" / "logs" / "trace_listener.jsonl"
CATALOG_PATH = BASE_DIR / "data" / "catalog.tsv"
DB_PATH = BASE_DIR / "data" / "eval_ratings.sqlite"
