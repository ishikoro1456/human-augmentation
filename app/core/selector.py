from pathlib import Path
from typing import Iterable, Optional

from .types import BackchannelItem


def pick_by_tags(
    items: Iterable[BackchannelItem],
    directory: str,
    strength: int,
    nod: int,
) -> Optional[BackchannelItem]:
    for item in items:
        if (
            item.directory == directory
            and item.strength == strength
            and item.nod == nod
        ):
            return item
    return None


def find_audio_file(
    base_dir: Path,
    item: BackchannelItem,
) -> Optional[Path]:
    target_dir = base_dir / item.directory
    if not target_dir.exists():
        return None
    prefix = f"{item.id}_s{item.strength}_n{item.nod}_"
    for path in target_dir.iterdir():
        if path.is_file() and path.name.startswith(prefix):
            return path
    return None
