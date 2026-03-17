import csv
from pathlib import Path
from typing import List

from .types import BackchannelItem


def load_catalog(path: Path) -> List[BackchannelItem]:
    items: List[BackchannelItem] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            items.append(
                BackchannelItem(
                    id=row["id"].strip(),
                    directory=row["directory"].strip(),
                    strength=int(row["strength"]),
                    nod=int(row["nod"]),
                    text=row["text"].strip(),
                )
            )
    return items


def build_context_text(path: Path) -> str:
    items = load_catalog(path)
    lines = [
        f"{item.id}\t{item.directory}\t{item.strength}\t{item.nod}\t{item.text}"
        for item in items
    ]
    return "\n".join(lines)
