#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.catalog import load_catalog
from app.demo import load_demo_script
from app.tts.openai_tts import synthesize_to_file


SCRIPT_TTS_INSTRUCTIONS = (
    "Speak clear English for a conference demo. "
    "Use a calm, natural pace and a professional but friendly tone."
)

BACKCHANNEL_TTS_INSTRUCTIONS = (
    "Speak as a short, natural English backchannel. "
    "Keep it brief, conversational, and easy to hear in a live demo."
)


def _write_manifest(
    *,
    manifest_path: Path,
    config: dict[str, object],
    files: list[dict[str, str]],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _generate_script_audio(
    *,
    client: OpenAI,
    script_path: Path,
    model: str,
    voice: str,
    response_format: str,
    overwrite: bool,
) -> list[dict[str, str]]:
    script = load_demo_script(script_path)
    audio_dir = script.resolve_audio_dir(script_path=script_path)
    generated: list[dict[str, str]] = []
    for segment in script.speech_segments():
        out_path = audio_dir / segment.audio
        if out_path.exists() and not overwrite:
            generated.append({"kind": "script", "text": segment.text, "path": str(out_path)})
            continue
        synthesize_to_file(
            client=client,
            text=segment.text,
            out_path=out_path,
            model=model,
            voice=voice,
            response_format=response_format,
            instructions=SCRIPT_TTS_INSTRUCTIONS,
        )
        generated.append({"kind": "script", "text": segment.text, "path": str(out_path)})
    return generated


def _resolve_backchannel_path(*, base_dir: Path, item_id: str, directory: str, response_format: str) -> Path | None:
    target_dir = base_dir / directory
    matches = sorted(target_dir.glob(f"{item_id}_*.{response_format}"))
    return matches[0] if matches else None


def _fallback_backchannel_path(*, base_dir: Path, item_id: str, directory: str, strength: int, nod: int, text: str, response_format: str) -> Path:
    slug = (
        text.lower()
        .replace("'", "")
        .replace("?", "")
        .replace(".", "")
        .replace(" ", "_")
    )
    return base_dir / directory / f"{item_id}_s{strength}_n{nod}_{slug}.{response_format}"


def _generate_backchannel_audio(
    *,
    client: OpenAI,
    catalog_path: Path,
    audio_dir: Path,
    model: str,
    voice: str,
    response_format: str,
    overwrite: bool,
) -> list[dict[str, str]]:
    generated: list[dict[str, str]] = []
    for item in load_catalog(catalog_path):
        out_path = _resolve_backchannel_path(
            base_dir=audio_dir,
            item_id=item.id,
            directory=item.directory,
            response_format=response_format,
        )
        if out_path is None:
            out_path = _fallback_backchannel_path(
                base_dir=audio_dir,
                item_id=item.id,
                directory=item.directory,
                strength=item.strength,
                nod=item.nod,
                text=item.text,
                response_format=response_format,
            )
        if out_path.exists() and not overwrite:
            generated.append({"kind": "backchannel", "text": item.text, "path": str(out_path)})
            continue
        synthesize_to_file(
            client=client,
            text=item.text,
            out_path=out_path,
            model=model,
            voice=voice,
            response_format=response_format,
            instructions=BACKCHANNEL_TTS_INSTRUCTIONS,
        )
        generated.append({"kind": "backchannel", "text": item.text, "path": str(out_path)})
    return generated


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate OpenAI TTS audio for the conference demo assets.")
    parser.add_argument("--script", default="data/demo/scripts/conference_demo_en.json")
    parser.add_argument("--catalog", default="data/demo/catalog_en.tsv")
    parser.add_argument("--audio-dir", default="data/demo/backchannel_en")
    parser.add_argument("--script-model", default="gpt-4o-mini-tts")
    parser.add_argument("--script-voice", default="marin")
    parser.add_argument("--backchannel-model", default="gpt-4o-mini-tts")
    parser.add_argument("--backchannel-voice", default="cedar")
    parser.add_argument("--format", default="mp3")
    parser.add_argument("--manifest", default="data/demo/audio_manifest.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    client = OpenAI()
    script_path = Path(args.script)
    catalog_path = Path(args.catalog)
    audio_dir = Path(args.audio_dir)
    manifest_path = Path(args.manifest)

    generated = []
    generated.extend(
        _generate_script_audio(
            client=client,
            script_path=script_path,
            model=args.script_model,
            voice=args.script_voice,
            response_format=args.format,
            overwrite=args.overwrite,
        )
    )
    generated.extend(
        _generate_backchannel_audio(
            client=client,
            catalog_path=catalog_path,
            audio_dir=audio_dir,
            model=args.backchannel_model,
            voice=args.backchannel_voice,
            response_format=args.format,
            overwrite=args.overwrite,
        )
    )

    _write_manifest(
        manifest_path=manifest_path,
        config={
            "script_path": str(script_path),
            "catalog_path": str(catalog_path),
            "audio_dir": str(audio_dir),
            "script_model": args.script_model,
            "script_voice": args.script_voice,
            "script_instructions": SCRIPT_TTS_INSTRUCTIONS,
            "backchannel_model": args.backchannel_model,
            "backchannel_voice": args.backchannel_voice,
            "backchannel_instructions": BACKCHANNEL_TTS_INSTRUCTIONS,
            "format": args.format,
            "overwrite": bool(args.overwrite),
        },
        files=generated,
    )

    print(f"Generated {len(generated)} files")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
