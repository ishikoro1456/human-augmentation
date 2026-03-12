#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.tts.openai_tts import synthesize_to_file


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "output"


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate OpenAI TTS audio for video script text",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["paper_materials/video/3_propose.txt"],
        help="Text file paths to read (default: paper_materials/video/3_propose.txt)",
    )
    parser.add_argument(
        "--out-dir",
        default="paper_materials/video/tts",
        help="Output directory for generated audio",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Base name for output files (default: derived from input names)",
    )
    parser.add_argument(
        "--voices",
        nargs="+",
        default=["alloy", "coral", "nova", "onyx", "cedar"],
        help="Voice names (space-separated)",
    )
    parser.add_argument("--model", default="gpt-4o-mini-tts", help="OpenAI TTS model")
    parser.add_argument(
        "--format",
        default="mp3",
        help="Audio format (e.g., mp3, wav)",
    )
    parser.add_argument(
        "--instructions",
        default="",
        help="Optional speaking style instruction",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Also create per-input audio files (in addition to the combined one)",
    )
    args = parser.parse_args()

    in_paths = [Path(p) for p in args.inputs]
    missing = [str(p) for p in in_paths if not p.exists()]
    if missing:
        print("Missing input files:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 2

    texts: list[tuple[str, str]] = []
    for p in in_paths:
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            continue
        texts.append((p.stem, text))

    if not texts:
        print("No non-empty text found in inputs.", file=sys.stderr)
        return 2

    combined_text = "\n".join(t for _, t in texts)
    out_dir = Path(args.out_dir)

    base_name = args.name.strip()
    if not base_name:
        base_name = "_".join(_slug(stem) for stem, _ in texts)

    client = OpenAI()
    for voice in args.voices:
        out_path = out_dir / f"{_slug(base_name)}__{_slug(voice)}.{args.format}"
        synthesize_to_file(
            client=client,
            text=combined_text,
            out_path=out_path,
            model=args.model,
            voice=voice,
            response_format=args.format,
            instructions=(args.instructions or None),
        )
        print(f"Wrote {out_path}")

        if args.per_file:
            for stem, text in texts:
                per_file_path = (
                    out_dir / f"{_slug(stem)}__{_slug(voice)}.{args.format}"
                )
                synthesize_to_file(
                    client=client,
                    text=text,
                    out_path=per_file_path,
                    model=args.model,
                    voice=voice,
                    response_format=args.format,
                    instructions=(args.instructions or None),
                )
                print(f"Wrote {per_file_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
