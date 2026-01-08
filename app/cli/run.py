#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.runtime.session import run_session


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Local backchannel agent")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument(
        "--catalog",
        default="data/catalog.tsv",
        help="Path to catalog.tsv",
    )
    parser.add_argument(
        "--audio-dir",
        default="data/backchannel",
        help="Directory that holds audio files",
    )
    parser.add_argument("--port", default="/dev/cu.usbserial-140")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--thread-id", default="local-session")
    parser.add_argument(
        "--transcript",
        default="transcribe.txt",
        help="Path to transcript with [mm:ss] lines",
    )
    parser.add_argument(
        "--transcript-start-sec",
        type=int,
        default=0,
        help="Start position for transcript time in seconds",
    )
    parser.add_argument(
        "--tts-model",
        default="gpt-4o-mini-tts",
        help="OpenAI TTS model for transcript playback",
    )
    parser.add_argument(
        "--tts-voice",
        default="alloy",
        help="OpenAI TTS voice",
    )
    parser.add_argument(
        "--tts-format",
        default="mp3",
        help="Audio format for transcript playback",
    )
    parser.add_argument(
        "--tts-cache-dir",
        default="data/tts_cache",
        help="Directory to cache generated TTS audio",
    )
    parser.add_argument(
        "--debug-imu",
        action="store_true",
        help="Print IMU values once per second",
    )
    parser.add_argument(
        "--debug-transcript",
        action="store_true",
        help="Print transcript context used by the agent",
    )
    parser.add_argument(
        "--debug-agent",
        action="store_true",
        help="Print IMU and agent reason",
    )
    args = parser.parse_args()

    run_session(
        catalog_path=Path(args.catalog),
        audio_dir=Path(args.audio_dir),
        port=args.port,
        baud=args.baud,
        model=args.model,
        thread_id=args.thread_id,
        debug_imu=args.debug_imu,
        transcript_path=Path(args.transcript),
        transcript_start_sec=args.transcript_start_sec,
        debug_transcript=args.debug_transcript,
        debug_agent=args.debug_agent,
        tts_model=args.tts_model,
        tts_voice=args.tts_voice,
        tts_format=args.tts_format,
        tts_cache_dir=Path(args.tts_cache_dir),
    )


if __name__ == "__main__":
    main()
