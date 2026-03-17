from pathlib import Path
from typing import Optional

from openai import OpenAI


def synthesize_to_file(
    client: OpenAI,
    text: str,
    out_path: Path,
    model: str = "gpt-4o-mini-tts",
    voice: str = "alloy",
    response_format: str = "mp3",
    instructions: Optional[str] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": response_format,
    }
    if instructions:
        params["instructions"] = instructions
    with client.audio.speech.with_streaming_response.create(**params) as response:
        response.stream_to_file(out_path)
