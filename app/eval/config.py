from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
TRACE_PATH = BASE_DIR / "data" / "runtime" / "logs" / "trace_listener.jsonl"
CATALOG_PATH = BASE_DIR / "data" / "catalog.tsv"
DB_PATH = BASE_DIR / "data" / "runtime" / "eval" / "eval_ratings.sqlite"
