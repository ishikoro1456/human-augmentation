#!/usr/bin/env python3
import argparse
import base64
import math
import os
import platform
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
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
from app.runtime.trace import TraceWriter


def _now_ms() -> int:
    return int(time.time() * 1000)


def _rms_s16le(raw: bytes) -> int:
    if not raw:
        return 0
    count = int(len(raw) // 2)
    if count <= 0:
        return 0
    total = 0
    for (s,) in struct.iter_unpack("<h", raw[: count * 2]):
        total += int(s) * int(s)
    return int(math.sqrt(total / count))


def _default_mic_backend() -> str:
    sysname = platform.system().lower()
    if sysname == "darwin":
        return "avfoundation"
    if sysname == "windows":
        return "dshow"
    return "pulse"


def _list_avfoundation_devices(ffmpeg_bin: str) -> None:
    cmd = [ffmpeg_bin, "-f", "avfoundation", "-list_devices", "true", "-i", ""]
    subprocess.run(cmd, check=False)


def _dshow_audio_devices(ffmpeg_bin: str) -> list[str]:
    cmd = [ffmpeg_bin, "-hide_banner", "-f", "dshow", "-list_devices", "true", "-i", "dummy"]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    devices: list[str] = []
    in_audio = False
    for line in text.splitlines():
        if "DirectShow audio devices" in line:
            in_audio = True
            continue
        if "DirectShow video devices" in line:
            in_audio = False
        if not in_audio:
            continue
        if "Alternative name" in line:
            continue
        m = re.search(r"\"([^\"]+)\"", line)
        if m:
            devices.append(m.group(1))
    return devices


def _list_dshow_devices(ffmpeg_bin: str) -> None:
    devices = _dshow_audio_devices(ffmpeg_bin)
    if not devices:
        print("DirectShow audio devices: 見つかりませんでした（ffmpeg が見えているか確認してください）")
        return
    print("DirectShow audio devices:")
    for i, name in enumerate(devices):
        print(f"[{i}] {name}")
    print("")
    print("使い方: --mic-backend dshow --mic-device :<index>")


def _list_pulse_devices() -> None:
    cmd = ["pactl", "list", "sources", "short"]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        out = (proc.stdout or "").strip()
        if out:
            print("PulseAudio sources:")
            print(out)
            print("")
            print("使い方: --mic-backend pulse --mic-device default")
            return
    except FileNotFoundError:
        pass
    print("PulseAudio sources: pactl が見つかりませんでした")
    print("確認: pactl list sources short")


def _list_alsa_devices() -> None:
    cmd = ["arecord", "-l"]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        out = (proc.stdout or "").strip()
        if out:
            print("ALSA devices:")
            print(out)
            print("")
            print("使い方: --mic-backend alsa --mic-device default")
            return
    except FileNotFoundError:
        pass
    print("ALSA devices: arecord が見つかりませんでした")
    print("確認: arecord -l")


def _start_ffmpeg_mic(
    *,
    ffmpeg_bin: str,
    backend: str,
    device: str,
    sample_rate: int,
) -> subprocess.Popen:
    if backend == "avfoundation":
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
    elif backend == "dshow":
        cmd = [
            ffmpeg_bin,
            "-loglevel",
            "error",
            "-f",
            "dshow",
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
    elif backend == "pulse":
        cmd = [
            ffmpeg_bin,
            "-loglevel",
            "error",
            "-f",
            "pulse",
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
    elif backend == "alsa":
        cmd = [
            ffmpeg_bin,
            "-loglevel",
            "error",
            "-f",
            "alsa",
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
    else:
        raise ValueError(f"unsupported mic backend: {backend}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    frame_ms: int = 20  # 送信する1フレームの長さ（ミリ秒）


@dataclass
class TalkerUiState:
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    connected: bool = False
    remote: str = ""
    experiment_id: str = ""
    guide: str = ""
    guide_ts: float | None = None

    mic_rms_mean_2s: float | None = None
    mic_rms_max_2s: int | None = None
    mic_ts: float | None = None

    last_backchannel_id: str = ""
    last_backchannel_text: str = ""
    last_backchannel_ts: float | None = None
    last_backchannel_played: bool | None = None

    def set_connected(self, *, connected: bool, remote: str = "") -> None:
        with self.lock:
            self.connected = bool(connected)
            self.remote = str(remote)

    def set_experiment_id(self, exp_id: str) -> None:
        with self.lock:
            self.experiment_id = str(exp_id)

    def set_guide(self, text: str) -> None:
        now = time.time()
        with self.lock:
            self.guide = str(text)
            self.guide_ts = float(now)

    def set_mic_stats(self, *, rms_mean_2s: float, rms_max_2s: int) -> None:
        now = time.time()
        with self.lock:
            self.mic_rms_mean_2s = float(rms_mean_2s)
            self.mic_rms_max_2s = int(rms_max_2s)
            self.mic_ts = float(now)

    def set_backchannel(self, *, bid: str, text: str, played: bool | None) -> None:
        now = time.time()
        with self.lock:
            self.last_backchannel_id = str(bid)
            self.last_backchannel_text = str(text)
            self.last_backchannel_ts = float(now)
            self.last_backchannel_played = played

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "connected": bool(self.connected),
                "remote": str(self.remote),
                "experiment_id": str(self.experiment_id),
                "guide": str(self.guide),
                "guide_ts": self.guide_ts,
                "mic_rms_mean_2s": self.mic_rms_mean_2s,
                "mic_rms_max_2s": self.mic_rms_max_2s,
                "mic_ts": self.mic_ts,
                "last_backchannel_id": str(self.last_backchannel_id),
                "last_backchannel_text": str(self.last_backchannel_text),
                "last_backchannel_ts": self.last_backchannel_ts,
                "last_backchannel_played": self.last_backchannel_played,
            }


def _fmt_age(ts: float | None) -> str:
    if ts is None:
        return "-"
    sec = max(0.0, time.time() - float(ts))
    if sec < 10:
        return f"{sec:0.1f}s"
    if sec < 60:
        return f"{sec:0.0f}s"
    return f"{int(sec)}s"


def _meter(value: float | None, *, max_value: float = 2000.0, width: int = 22) -> str:
    if value is None:
        return "-"
    v = max(0.0, float(value))
    frac = min(1.0, v / float(max_value))
    filled = int(round(frac * int(width)))
    bar = "█" * filled + " " * (int(width) - filled)
    return f"[{bar}] {v:.0f}"


def _run_talker_ui(ui: TalkerUiState, *, ui_mode: str = "participant", refresh_hz: int = 12) -> None:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()
    mode_norm = str(ui_mode or "participant").strip().lower()
    if mode_norm not in ("participant", "debug"):
        mode_norm = "participant"

    def render() -> Panel:
        snap = ui.snapshot()
        guide = (snap.get("guide") or "").strip() or "（待機中）"
        guide_age = _fmt_age(snap.get("guide_ts"))
        guide_panel = Panel(
            Text(guide, overflow="fold", no_wrap=False),
            title=f"案内 (age={guide_age})",
            border_style="yellow",
        )

        info = Table.grid(padding=(0, 1))
        info.add_column(style="bold")
        info.add_column()
        exp = str(snap.get("experiment_id") or "-").strip() or "-"
        info.add_row("experiment", exp)
        if bool(snap.get("connected")):
            info.add_row("listener", f"connected {snap.get('remote','')}".strip())
        else:
            info.add_row("listener", "not connected")

        mic_age = _fmt_age(snap.get("mic_ts"))
        mic_mean = snap.get("mic_rms_mean_2s")
        mic_max = snap.get("mic_rms_max_2s")
        mean_s = _meter(mic_mean, max_value=2000.0, width=18)
        max_s = "-" if mic_max is None else str(int(mic_max))
        info.add_row("mic", f"age={mic_age} mean={mean_s} max={max_s}")
        info_panel = Panel(info, title="状態", border_style="magenta")

        last_id = str(snap.get("last_backchannel_id") or "").strip()
        last_text = str(snap.get("last_backchannel_text") or "").strip()
        last_age = _fmt_age(snap.get("last_backchannel_ts"))
        last_played = snap.get("last_backchannel_played")
        played_s = "-" if last_played is None else ("yes" if bool(last_played) else "no")
        if last_id:
            bc = f"{last_id} {last_text}\nplayed={played_s} age={last_age}"
        else:
            bc = "（まだ届いていません）"
        bc_panel = Panel(Text(bc, overflow="fold", no_wrap=False), title="直近の相槌", border_style="green")

        body = Table.grid(expand=True)
        if mode_norm == "participant":
            body.add_row(guide_panel)
            body.add_row(info_panel)
            return Panel(body, title="話し手", border_style="white")
        body.add_row(guide_panel)
        body.add_row(info_panel)
        body.add_row(bc_panel)
        return Panel(body, title="Talker", border_style="white")

    with Live(render(), console=console, refresh_per_second=int(max(1, refresh_hz)), screen=True):
        while True:
            time.sleep(0.1)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Talker app (mic -> send audio, receive backchannel)")
    parser.add_argument("--connect-host", default="", help="Listener host or IP")
    parser.add_argument("--connect-port", type=int, default=8765)

    parser.add_argument("--ffmpeg-bin", default=os.environ.get("FFMPEG_BIN", "ffmpeg"))
    parser.add_argument(
        "--mic-backend",
        choices=["auto", "avfoundation", "dshow", "pulse", "alsa"],
        default="auto",
        help="ffmpeg mic backend. auto: mac=avfoundation, windows=dshow, linux=pulse",
    )
    parser.add_argument(
        "--mic-device",
        default=":0",
        help="mic device. avfoundation: ':<index>'. dshow: ':<index>' or 'audio=<name>'. pulse/alsa: 'default' など。Use --list-devices.",
    )
    parser.add_argument("--list-devices", action="store_true", help="List mic devices and exit")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frame-ms", type=int, default=20, help="Audio frame size to send (ms)")

    parser.add_argument("--catalog", default="data/catalog.tsv")
    parser.add_argument("--audio-dir", default="data/backchannel")
    parser.add_argument("--no-play-backchannel", action="store_true", help="Do not play received backchannel audio")

    parser.add_argument("--ui", action="store_true", help="Show a simple UI for the talker")
    parser.add_argument("--ui-mode", choices=["participant", "debug"], default="participant")
    parser.add_argument("--ui-refresh-hz", type=int, default=12)
    parser.add_argument("--debug-net", action="store_true", help="Print network events")
    parser.add_argument("--trace-jsonl", default="", help="通信と再生のログをJSONLで残します")

    args = parser.parse_args()

    mic_backend = str(args.mic_backend or "auto").strip().lower()
    if mic_backend == "auto":
        mic_backend = _default_mic_backend()

    if args.list_devices:
        if mic_backend == "avfoundation":
            _list_avfoundation_devices(args.ffmpeg_bin)
        elif mic_backend == "dshow":
            _list_dshow_devices(args.ffmpeg_bin)
        elif mic_backend == "pulse":
            _list_pulse_devices()
        elif mic_backend == "alsa":
            _list_alsa_devices()
        else:
            print(f"未対応の mic backend です: {mic_backend}")
        return
    if not args.connect_host:
        parser.error("--connect-host は必須です（--list-devices のときは不要です）")

    trace = TraceWriter(Path(args.trace_jsonl)) if args.trace_jsonl else None
    if trace:
        trace.set_meta(role="talker")

    host = resolve_host(args.connect_host)
    port = int(args.connect_port)

    audio_cfg = AudioConfig(sample_rate=int(args.sample_rate), frame_ms=int(args.frame_ms))
    items = load_catalog(Path(args.catalog))
    audio_dir = Path(args.audio_dir)
    player = AudioPlayer()

    ui: TalkerUiState | None = TalkerUiState() if args.ui else None
    if ui is not None:
        threading.Thread(
            target=_run_talker_ui,
            args=(ui,),
            kwargs={"ui_mode": str(args.ui_mode), "refresh_hz": int(args.ui_refresh_hz)},
            daemon=True,
        ).start()

    if trace:
        trace.write(
            {
                "type": "talker_start",
                "connect": {"host": str(host), "port": int(port)},
                "mic_device": str(args.mic_device),
                "mic_backend": str(mic_backend),
                "audio": {"format": "s16le", "sample_rate": int(audio_cfg.sample_rate), "channels": 1, "frame_ms": int(audio_cfg.frame_ms)},
            }
        )

    experiment_id: str = ""

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
                if trace:
                    try:
                        local = s.getsockname()
                    except Exception:
                        local = None
                    trace.write({"type": "talker_connected", "remote": f"{host}:{port}", "local": local})
                if ui is not None:
                    ui.set_connected(connected=True, remote=f"{host}:{port}")
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
                if trace:
                    trace.write({"type": "hello_sent", "remote": f"{host}:{port}"})
                if not args.ui:
                    print(f"接続しました: {host}:{port}")
                return s
            except Exception as exc:
                if not args.ui:
                    print(f"接続に失敗しました: {host}:{port} ({exc})")
                if ui is not None:
                    ui.set_connected(connected=False, remote=f"{host}:{port}")
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
        if trace:
            trace.write({"type": "talker_disconnected"})
        if ui is not None:
            ui.set_connected(connected=False, remote=f"{host}:{port}")

    if not args.ui:
        print(f"接続を待ちます: {host}:{port}")
    _ensure_connected()

    def _recv_loop() -> None:
        nonlocal experiment_id
        last_guide_print = ""
        while True:
            s = _ensure_connected()
            try:
                for msg in iter_jsonl_messages(s):
                    if not isinstance(msg, dict):
                        continue
                    msg_type = str(msg.get("type", ""))
                    if msg_type == "session":
                        exp = str(msg.get("experiment_id", "") or "").strip()
                        if exp and exp != experiment_id:
                            experiment_id = exp
                            if trace:
                                trace.set_meta(experiment_id=experiment_id)
                                trace.write({"type": "session_received", "experiment_id": experiment_id})
                            if ui is not None:
                                ui.set_experiment_id(experiment_id)
                            if args.debug_net:
                                if not args.ui:
                                    print(f"受信(session): experiment_id={experiment_id}")
                        continue
                    if msg_type == "guide":
                        text = str(msg.get("text", "") or "").strip()
                        if text and ui is not None:
                            ui.set_guide(text)
                        if text and (not args.ui) and text != last_guide_print:
                            print(f"案内: {text}")
                            last_guide_print = text
                        continue

                    if msg_type != "backchannel":
                        if args.debug_net:
                            if not args.ui:
                                print(f"受信: {msg}")
                        continue
                    bid = str(msg.get("id", "") or "")
                    if not bid:
                        continue
                    btext = str(msg.get("text", "") or "")
                    if args.debug_net:
                        if not args.ui:
                            print(f"受信(backchannel): {bid} {btext}".strip())
                    reason = str(msg.get("reason", "") or "")
                    planned = bool(msg.get("planned", False))
                    latency_ms = msg.get("latency_ms")
                    call_id = str(msg.get("call_id", "") or "")
                    if trace:
                        trace.write(
                            {
                                "type": "backchannel_received",
                                "id": bid,
                                "text": btext,
                                "reason": reason,
                                "planned": planned,
                                "latency_ms": int(latency_ms) if isinstance(latency_ms, int) else None,
                                "call_id": call_id,
                            }
                        )
                    if args.no_play_backchannel:
                        if ui is not None:
                            ui.set_backchannel(bid=bid, text=btext, played=None)
                        continue
                    item = next((it for it in items if it.id == bid), None)
                    if item is None:
                        if ui is not None:
                            ui.set_backchannel(bid=bid, text=btext, played=None)
                        continue
                    path = find_audio_file(audio_dir, item)
                    if not path:
                        if ui is not None:
                            ui.set_backchannel(bid=bid, text=btext, played=None)
                        continue
                    played = player.play_effect(path, interrupt=True)
                    if ui is not None:
                        ui.set_backchannel(bid=bid, text=btext, played=played)
                    if trace:
                        trace.write(
                            {
                                "type": "backchannel_play",
                                "id": bid,
                                "text": btext,
                                "call_id": call_id,
                                "audio_path": str(path),
                                "played": bool(played),
                            }
                        )
            except Exception as exc:
                if args.debug_net:
                    if not args.ui:
                        print(f"受信が切れました（再接続します）: {exc}")
                _drop_connection()
                time.sleep(0.2)
    threading.Thread(target=_recv_loop, daemon=True).start()

    mic_device = str(args.mic_device)
    if mic_backend in {"pulse", "alsa"} and mic_device == ":0":
        mic_device = "default"
    if mic_backend == "dshow":
        if mic_device.startswith(":") and mic_device[1:].isdigit():
            idx = int(mic_device[1:])
            devices = _dshow_audio_devices(args.ffmpeg_bin)
            if 0 <= idx < len(devices):
                mic_device = "audio=" + devices[idx]
            else:
                raise RuntimeError(f"mic-device index out of range: {mic_device} (devices={len(devices)})")
        elif not (mic_device.startswith("audio=") or mic_device.startswith("video=")):
            mic_device = "audio=" + mic_device

    proc = _start_ffmpeg_mic(
        ffmpeg_bin=args.ffmpeg_bin,
        backend=mic_backend,
        device=mic_device,
        sample_rate=audio_cfg.sample_rate,
    )
    if proc.stdout is None:
        raise RuntimeError("ffmpegのstdoutが取れません。")

    if not args.ui:
        print("マイク取り込みを開始しました。止めるには Ctrl+C です。")
    try:
        frame_samples = int(audio_cfg.sample_rate * audio_cfg.frame_ms / 1000)
        frame_bytes = frame_samples * 2
        seq = 0
        stats_started = time.time()
        stats_frames = 0
        stats_bytes = 0
        stats_rms_sum = 0
        stats_rms_max = 0
        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break

            rms = _rms_s16le(raw)
            stats_frames += 1
            stats_bytes += len(raw)
            stats_rms_sum += int(rms)
            stats_rms_max = max(int(stats_rms_max), int(rms))

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
                    if not args.ui:
                        print(f"送信に失敗しました（再接続します）: {exc}")
                _drop_connection()

            if (time.time() - stats_started) >= 2.0:
                mean = (stats_rms_sum / stats_frames) if stats_frames > 0 else 0.0
                if trace:
                    trace.write(
                        {
                            "type": "audio_send_stats",
                            "frames": int(stats_frames),
                            "bytes": int(stats_bytes),
                            "rms_mean": round(float(mean), 1),
                            "rms_max": int(stats_rms_max),
                        }
                    )
                if ui is not None:
                    ui.set_mic_stats(rms_mean_2s=float(mean), rms_max_2s=int(stats_rms_max))
                stats_started = time.time()
                stats_frames = 0
                stats_bytes = 0
                stats_rms_sum = 0
                stats_rms_max = 0
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
