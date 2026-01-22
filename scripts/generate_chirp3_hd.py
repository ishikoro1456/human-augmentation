#!/usr/bin/env python3
import argparse
import csv
import os
import re
from typing import Iterable, Dict, Optional

from google.cloud import texttospeech
from google.api_core.client_options import ClientOptions


def sanitize_filename(text: str, max_len: int = 60) -> str:
    text = text.strip()
    text = re.sub(r"[\\/:*?\"<>|]", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("_")
    if not text:
        return "text"
    return text[:max_len]


def load_rows(path: str) -> Iterable[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def build_client(endpoint: Optional[str]) -> texttospeech.TextToSpeechClient:
    if endpoint:
        return texttospeech.TextToSpeechClient(
            client_options=ClientOptions(api_endpoint=endpoint)
        )
    return texttospeech.TextToSpeechClient()


def list_chirp3_hd_voices(client: texttospeech.TextToSpeechClient, language: str) -> None:
    voices = client.list_voices()
    for voice in voices.voices:
        if language in voice.language_codes and "Chirp3-HD" in voice.name:
            print(voice.name)


def synthesize_one(
    client: texttospeech.TextToSpeechClient,
    text: str,
    voice_name: str,
    language: str,
    encoding: str,
) -> bytes:
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language,
        name=voice_name,
    )

    if encoding == "mp3":
        audio_encoding = texttospeech.AudioEncoding.MP3
    elif encoding == "wav":
        audio_encoding = texttospeech.AudioEncoding.LINEAR16
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    audio_config = texttospeech.AudioConfig(audio_encoding=audio_encoding)

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    return response.audio_content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate short backchannel audio with Chirp 3 HD voices."
    )
    parser.add_argument("--input", default="aizuchi_set.tsv")
    parser.add_argument("--out", default="backchannel")
    parser.add_argument("--language", default="ja-JP")
    parser.add_argument(
        "--voice",
        default="ja-JP-Chirp3-HD-Leda",
        help="Voice name, for example ja-JP-Chirp3-HD-Leda",
    )
    parser.add_argument("--encoding", choices=["mp3", "wav"], default="mp3")
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Optional endpoint, e.g. us-texttospeech.googleapis.com",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available Chirp3-HD voices for the language and exit",
    )

    args = parser.parse_args()

    client = build_client(args.endpoint)

    if args.list_voices:
        list_chirp3_hd_voices(client, args.language)
        return

    os.makedirs(args.out, exist_ok=True)

    for row in load_rows(args.input):
        directory = row["directory"].strip()
        strength = row["strength"].strip()
        nod = row["nod"].strip()
        text = row["text"].strip()
        row_id = row["id"].strip()

        out_dir = os.path.join(args.out, directory)
        os.makedirs(out_dir, exist_ok=True)

        slug = sanitize_filename(text)
        ext = args.encoding
        filename = f"{row_id}_s{strength}_n{nod}_{slug}.{ext}"
        out_path = os.path.join(out_dir, filename)
        prefix = f"{row_id}_s{strength}_n{nod}_"

        # 同じ prefix の古い音声を掃除（text が変わるとファイル名が変わるので、残ると選択が不安定になる）
        try:
            for old in os.listdir(out_dir):
                if old.startswith(prefix) and old != filename:
                    try:
                        os.remove(os.path.join(out_dir, old))
                    except OSError:
                        pass
        except OSError:
            pass

        audio_content = synthesize_one(
            client=client,
            text=text,
            voice_name=args.voice,
            language=args.language,
            encoding=args.encoding,
        )

        with open(out_path, "wb") as f:
            f.write(audio_content)

        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
