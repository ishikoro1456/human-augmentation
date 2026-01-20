#!/usr/bin/env python3
import argparse
import audioop
import os
import socket
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.net.jsonl import resolve_host, send_jsonl


def _now_ms() -> int:
    return int(time.time() * 1000)


def _write_wav(path: Path, *, pcm_s16le: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm_s16le)


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
class VadConfig:
    frame_ms: int = 20
    pre_roll_ms: int = 200
    silence_end_ms: int = 500
    min_speech_ms: int = 300
    calib_sec: float = 1.0
    threshold_rms: int = 0  # 0なら自動
    threshold_mult: float = 3.0


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Talker app (mic -> STT -> send transcript)")
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

    parser.add_argument("--stt-model", default="whisper-1")
    parser.add_argument("--stt-language", default="ja")
    parser.add_argument("--stt-prompt", default="")

    parser.add_argument("--segments-dir", default="data/stt_segments")

    parser.add_argument("--vad-frame-ms", type=int, default=20)
    parser.add_argument("--vad-pre-roll-ms", type=int, default=200)
    parser.add_argument("--vad-silence-end-ms", type=int, default=500)
    parser.add_argument("--vad-min-speech-ms", type=int, default=300)
    parser.add_argument("--vad-calib-sec", type=float, default=1.0)
    parser.add_argument("--vad-threshold-rms", type=int, default=0)
    parser.add_argument("--vad-threshold-mult", type=float, default=3.0)

    args = parser.parse_args()

    if args.list_devices:
        _list_avfoundation_devices(args.ffmpeg_bin)
        return
    if not args.connect_host:
        parser.error("--connect-host は必須です（--list-devices のときは不要です）")

    host = resolve_host(args.connect_host)
    port = int(args.connect_port)

    client = OpenAI()

    seg_dir = Path(args.segments_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)

    cfg = VadConfig(
        frame_ms=int(args.vad_frame_ms),
        pre_roll_ms=int(args.vad_pre_roll_ms),
        silence_end_ms=int(args.vad_silence_end_ms),
        min_speech_ms=int(args.vad_min_speech_ms),
        calib_sec=float(args.vad_calib_sec),
        threshold_rms=int(args.vad_threshold_rms),
        threshold_mult=float(args.vad_threshold_mult),
    )

    # 接続（落ちても再接続）
    sock: Optional[socket.socket] = None

    def _ensure_connected() -> socket.socket:
        nonlocal sock
        while True:
            if sock is not None:
                return sock
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((host, port))
                sock = s
                send_jsonl(
                    s,
                    {
                        "type": "hello",
                        "role": "talker",
                        "version": 1,
                        "ts_ms": _now_ms(),
                    },
                )
                print(f"接続しました: {host}:{port}")
                return s
            except Exception as exc:
                print(f"接続に失敗しました: {host}:{port} ({exc})")
                time.sleep(1.0)

    proc = _start_ffmpeg_mic(ffmpeg_bin=args.ffmpeg_bin, device=args.mic_device, sample_rate=args.sample_rate)
    if proc.stdout is None:
        raise RuntimeError("ffmpegのstdoutが取れません。")

    frame_samples = int(args.sample_rate * cfg.frame_ms / 1000)
    frame_bytes = frame_samples * 2
    pre_roll_frames = max(0, int(cfg.pre_roll_ms / cfg.frame_ms))
    silence_end_frames = max(1, int(cfg.silence_end_ms / cfg.frame_ms))
    min_speech_frames = max(1, int(cfg.min_speech_ms / cfg.frame_ms))

    # ノイズ床をざっくり推定（無音のときだけ更新）
    noise_rms = 200.0
    calib_until = time.time() + max(0.0, cfg.calib_sec)

    speaking = False
    silence_frames = 0
    pre_roll = []
    segment_frames = []
    segment_start_ms = 0
    seg_id = 0

    def _send_state(*, speaking_now: bool, silence_ms: int, rms: int) -> None:
        s = _ensure_connected()
        send_jsonl(
            s,
            {
                "type": "speech_state",
                "speaking": bool(speaking_now),
                "silence_ms": int(max(0, silence_ms)),
                "rms": int(rms),
                "ts_ms": _now_ms(),
            },
        )

    print("マイク取り込みを開始しました。止めるには Ctrl+C です。")
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break

            rms = int(audioop.rms(raw, 2))

            # 自動しきい値
            if time.time() < calib_until:
                noise_rms = (noise_rms * 0.95) + (rms * 0.05)
            elif not speaking:
                # 無音中だけゆっくり更新
                noise_rms = (noise_rms * 0.995) + (rms * 0.005)

            thr = int(cfg.threshold_rms) if cfg.threshold_rms > 0 else int(max(300.0, noise_rms * cfg.threshold_mult))
            is_voice = rms >= thr

            pre_roll.append(raw)
            if len(pre_roll) > pre_roll_frames:
                pre_roll = pre_roll[-pre_roll_frames:]

            if not speaking:
                if is_voice:
                    speaking = True
                    silence_frames = 0
                    segment_frames = list(pre_roll)
                    segment_start_ms = _now_ms()
                    _send_state(speaking_now=True, silence_ms=0, rms=rms)
                continue

            # speaking中
            segment_frames.append(raw)
            if is_voice:
                silence_frames = 0
            else:
                silence_frames += 1

            if silence_frames < silence_end_frames:
                continue

            # 区切り
            speaking = False
            seg_end_ms = _now_ms()
            _send_state(speaking_now=False, silence_ms=int(silence_frames * cfg.frame_ms), rms=rms)

            if len(segment_frames) < min_speech_frames:
                segment_frames = []
                pre_roll = []
                continue

            seg_id += 1
            wav_path = seg_dir / f"seg_{seg_id:04d}_{segment_start_ms}_{seg_end_ms}.wav"
            pcm = b"".join(segment_frames)
            _write_wav(wav_path, pcm_s16le=pcm, sample_rate=args.sample_rate)

            try:
                with wav_path.open("rb") as f:
                    params = {
                        "model": args.stt_model,
                        "file": f,
                        "response_format": "text",
                        "language": args.stt_language,
                    }
                    if args.stt_prompt:
                        params["prompt"] = args.stt_prompt
                    text = client.audio.transcriptions.create(**params)
                text = str(text).strip()
            except Exception as exc:
                print(f"文字起こしに失敗しました: {exc}")
                segment_frames = []
                pre_roll = []
                continue

            if text:
                s = _ensure_connected()
                send_jsonl(
                    s,
                    {
                        "type": "segment_final",
                        "segment_id": int(seg_id),
                        "text": text,
                        "start_ts_ms": int(segment_start_ms),
                        "end_ts_ms": int(seg_end_ms),
                        "ts_ms": int(seg_end_ms),
                    },
                )
                print(text)

            segment_frames = []
            pre_roll = []
            silence_frames = 0
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
        try:
            if sock is not None:
                sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
