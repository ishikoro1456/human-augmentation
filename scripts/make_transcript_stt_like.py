#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


TIMESTAMP = re.compile(r"^\[(\d+):(\d{2})\]\s*(.*)$")
# 句点と読点の区切りも使い、短いチャンクを作る
BREAK = re.compile(r"(?<=[。！？!?、])")


@dataclass(frozen=True)
class Entry:
    t_sec: int
    text: str


def _to_sec(mm: int, ss: int) -> int:
    return int(mm) * 60 + int(ss)


def split_units(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = [p.strip() for p in BREAK.split(text) if p.strip()]
    merged: list[str] = []
    for p in parts:
        if not merged:
            merged.append(p)
            continue
        if len(p) <= 2:
            merged[-1] = (merged[-1] + p).strip()
            continue
        merged.append(p)
    return merged


def group_units(units: list[str], *, max_chars: int = 40, min_chars: int = 12) -> list[str]:
    chunks: list[str] = []
    buf = ""
    for u in units:
        candidate = u if not buf else (buf + u)
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            chunks.append(buf.strip())
            buf = u
        else:
            chunks.append(u.strip())
            buf = ""

    if buf.strip():
        chunks.append(buf.strip())

    # 末尾が短すぎる場合は直前と結合
    if len(chunks) >= 2 and len(chunks[-1]) < min_chars:
        chunks[-2] = (chunks[-2] + chunks[-1]).strip()
        chunks.pop()
    return chunks


def load_timestamped(path: Path) -> list[Entry]:
    entries: list[Entry] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = TIMESTAMP.match(line)
        if not m:
            continue
        mm, ss, text = m.groups()
        entries.append(Entry(t_sec=_to_sec(int(mm), int(ss)), text=text.strip()))
    return entries


def main() -> int:
    in_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path("transcribe_2min.txt")
    out_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("transcribe_stt.txt")

    entries = load_timestamped(in_path)
    if not entries:
        print("No timestamped entries found. Expected lines like: [mm:ss] text", file=sys.stderr)
        return 2

    out_lines: list[str] = []
    for e in entries:
        units = split_units(e.text)
        chunks = group_units(units)
        out_lines.extend(chunks)
        out_lines.append("")

    # 末尾の空行を1つだけにする
    while len(out_lines) >= 2 and out_lines[-1] == "" and out_lines[-2] == "":
        out_lines.pop()

    out_path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({sum(1 for x in out_lines if x.strip())} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
