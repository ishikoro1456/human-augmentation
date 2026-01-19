#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


TIMESTAMP = re.compile(r"^\[(\d+):(\d{2})\]\s*(.*)$")
# 日本語は句点の後に空白が無いことが多いので、空白に依存しない
SENT_SPLIT = re.compile(r"(?<=[。！？!?])")


@dataclass(frozen=True)
class Entry:
    t_sec: int
    text: str


def _to_sec(mm: int, ss: int) -> int:
    return int(mm) * 60 + int(ss)


def _fmt_ts(t_sec: int) -> str:
    mm = t_sec // 60
    ss = t_sec % 60
    return f"[{mm:02d}:{ss:02d}]"


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = [p.strip() for p in SENT_SPLIT.split(text) if p.strip()]
    # 末尾に句読点だけが残ることがあるので、短すぎる断片は前に寄せる
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


def group_sentences(sentences: list[str], *, max_chars: int = 90, min_chars: int = 25) -> list[str]:
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        candidate = s if not buf else (buf + s)
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            chunks.append(buf.strip())
            buf = s
        else:
            chunks.append(s.strip())
            buf = ""

    if buf.strip():
        chunks.append(buf.strip())

    # 末尾が短すぎる場合は直前と結合
    if len(chunks) >= 2 and len(chunks[-1]) < min_chars:
        chunks[-2] = (chunks[-2] + chunks[-1]).strip()
        chunks.pop()
    return chunks


def distribute_times(start: int, end: int, n: int) -> list[int]:
    if n <= 1:
        return [start]
    span = max(1, end - start)
    times: list[int] = []
    last = start - 1
    for i in range(n):
        t = start + int((span * i) / n)
        if t <= last:
            t = last + 1
        if t >= end:
            t = max(start, end - 1)
            if t <= last:
                t = last + 1
        times.append(t)
        last = t
    return times


def load_entries(path: Path) -> list[Entry]:
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


def build_natural(entries: list[Entry]) -> list[Entry]:
    out: list[Entry] = []
    for i, e in enumerate(entries):
        next_t = entries[i + 1].t_sec if i < (len(entries) - 1) else (e.t_sec + 120)
        sentences = split_sentences(e.text)
        if not sentences:
            continue
        chunks = group_sentences(sentences)
        times = distribute_times(e.t_sec, next_t, len(chunks))
        for t, chunk in zip(times, chunks):
            out.append(Entry(t_sec=t, text=chunk))
    return out


def main() -> int:
    in_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path("transcribe.txt")
    out_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("transcribe_natural.txt")

    entries = load_entries(in_path)
    if not entries:
        print("No entries found. Expected lines like: [mm:ss] text", file=sys.stderr)
        return 2

    natural = build_natural(entries)
    lines: list[str] = []
    last_mm: int | None = None
    for e in natural:
        mm = e.t_sec // 60
        if last_mm is not None and mm != last_mm:
            lines.append("")
        lines.append(f"{_fmt_ts(e.t_sec)} {e.text}")
        last_mm = mm
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(natural)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
