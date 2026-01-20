#!/usr/bin/env python3
import argparse
import audioop
import base64
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.audio.player import AudioPlayer
from app.core.catalog import load_catalog
from app.core.selector import find_audio_file
from app.net.jsonl import iter_jsonl_messages, resolve_host, send_jsonl


def _now_ms() -> int:
    return int(time.time() * 1000)

def _list_avfoundation_devices(ffmpeg_bin: str) -> None:
    cmd = [ffmpeg_bin, "-f", "avfoundation", "-list_devices", "true", "-i", ""]
    subprocess.run(cmd, check=False)


def _start_ffmpeg_mic(
    *,
    ffmpeg_bin: str,
    device: str,
    sample_rate: int,
) -> subprocess.Popen:
    cmd = [
        ffmpeg_bin,
        "-loglevel",
        "error",
        "-f",
        "avfoundation",
        "-i",
        device,
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-f",
        "s16le",
        "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    frame_ms: int = 20  # 送信する1フレームの長さ（ミリ秒）


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Talker app (mic -> send audio, receive backchannel)")
    parser.add_argument("--connect-host", default="", help="Listener host or IP")
    parser.add_argument("--connect-port", type=int, default=8765)

    parser.add_argument("--ffmpeg-bin", default=os.environ.get("FFMPEG_BIN", "ffmpeg"))
    parser.add_argument(
        "--mic-device",
        default=":0",
        help="ffmpeg avfoundation device string. Example ':0' (audio only). Use --list-devices to see indices.",
    )
    parser.add_argument("--list-devices", action="store_true", help="List avfoundation devices and exit")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frame-ms", type=int, default=20, help="Audio frame size to send (ms)")

    parser.add_argument("--catalog", default="data/catalog.tsv")
    parser.add_argument("--audio-dir", default="data/backchannel")
    parser.add_argument("--no-play-backchannel", action="store_true", help="Do not play received backchannel audio")

    parser.add_argument("--debug-net", action="store_true", help="Print network events")

    args = parser.parse_args()

    if args.list_devices:
        _list_avfoundation_devices(args.ffmpeg_bin)
        return
    if not args.connect_host:
        parser.error("--connect-host は必須です（--list-devices のときは不要です）")

    host = resolve_host(args.connect_host)
    port = int(args.connect_port)

    audio_cfg = AudioConfig(sample_rate=int(args.sample_rate), frame_ms=int(args.frame_ms))
    items = load_catalog(Path(args.catalog))
    audio_dir = Path(args.audio_dir)
    player = AudioPlayer()

    # 接続（落ちても再接続）
    sock: Optional[socket.socket] = None
    sock_lock = threading.Lock()

    def _ensure_connected() -> socket.socket:
        nonlocal sock
        while True:
            with sock_lock:
                if sock is not None:
                    return sock
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((host, port))
                with sock_lock:
                    sock = s
                send_jsonl(
                    s,
                    {
                        "type": "hello",
                        "role": "talker",
                        "version": 2,
                        "audio": {
                            "format": "s16le",
                            "sample_rate": int(audio_cfg.sample_rate),
                            "channels": 1,
                            "frame_ms": int(audio_cfg.frame_ms),
                        },
                        "ts_ms": _now_ms(),
                    },
                )
                print(f"接続しました: {host}:{port}")
                return s
            except Exception as exc:
                print(f"接続に失敗しました: {host}:{port} ({exc})")
                time.sleep(1.0)

    def _drop_connection() -> None:
        nonlocal sock
        with sock_lock:
            s = sock
            sock = None
        if s is not None:
            try:
                s.close()
            except Exception:
                pass

    print(f"接続を待ちます: {host}:{port}")
    _ensure_connected()

    def _recv_loop() -> None:
        while True:
            s = _ensure_connected()
            try:
                for msg in iter_jsonl_messages(s):
                    if not isinstance(msg, dict):
                        continue
                    if str(msg.get("type", "")) != "backchannel":
                        if args.debug_net:
                            print(f"受信: {msg}")
                        continue
                    bid = str(msg.get("id", "") or "")
                    if not bid:
                        continue
                    btext = str(msg.get("text", "") or "")
                    if args.debug_net:
                        print(f"受信(backchannel): {bid} {btext}".strip())
                    if args.no_play_backchannel:
                        continue
                    item = next((it for it in items if it.id == bid), None)
                    if item is None:
                        continue
                    path = find_audio_file(audio_dir, item)
                    if not path:
                        continue
                    player.play_effect(path)
            except Exception as exc:
                if args.debug_net:
                    print(f"受信が切れました（再接続します）: {exc}")
                _drop_connection()
                time.sleep(0.2)
    threading.Thread(target=_recv_loop, daemon=True).start()

    proc = _start_ffmpeg_mic(ffmpeg_bin=args.ffmpeg_bin, device=args.mic_device, sample_rate=audio_cfg.sample_rate)
    if proc.stdout is None:
        raise RuntimeError("ffmpegのstdoutが取れません。")

    print("マイク取り込みを開始しました。止めるには Ctrl+C です。")
    try:
        frame_samples = int(audio_cfg.sample_rate * audio_cfg.frame_ms / 1000)
        frame_bytes = frame_samples * 2
        seq = 0
        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break

            rms = int(audioop.rms(raw, 2))
            payload = {
                "type": "audio_chunk",
                "seq": int(seq),
                "ts_ms": _now_ms(),
                "rms": int(rms),
                "data_b64": base64.b64encode(raw).decode("ascii"),
            }
            seq += 1

            try:
                s = _ensure_connected()
                send_jsonl(s, payload)
            except Exception as exc:
                if args.debug_net:
                    print(f"送信に失敗しました（再接続します）: {exc}")
                _drop_connection()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
        try:
            _drop_connection()
        except Exception:
            pass


if __name__ == "__main__":
    main()
