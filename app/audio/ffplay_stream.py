from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class FfplayConfig:
    ffplay_bin: str = "ffplay"
    sample_rate: int = 16000
    channels: int = 1


class FfplayStreamPlayer:
    def __init__(self, cfg: FfplayConfig) -> None:
        self._cfg = cfg
        self._proc: Optional[subprocess.Popen] = None

    def is_available(self) -> bool:
        return shutil.which(self._cfg.ffplay_bin) is not None

    def start(self) -> bool:
        if self._proc is not None and self._proc.poll() is None:
            return True
        if not self.is_available():
            return False
        cmd = [
            self._cfg.ffplay_bin,
            "-nodisp",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-f",
            "s16le",
            "-ar",
            str(int(self._cfg.sample_rate)),
            "-ch_layout",
            "mono" if int(self._cfg.channels) == 1 else "stereo",
            "-i",
            "-",
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True

    def write(self, pcm_s16le: bytes) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        if proc.stdin is None:
            return
        try:
            proc.stdin.write(pcm_s16le)
            proc.stdin.flush()
        except Exception:
            self.close()

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
