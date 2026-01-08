from dataclasses import dataclass


@dataclass(frozen=True)
class BackchannelItem:
    id: str
    directory: str
    strength: int
    nod: int
    text: str
