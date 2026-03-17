from __future__ import annotations

from collections import deque
import shutil
import subprocess
import threading
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
        self._stderr_lock = threading.Lock()
        self._stderr_tail: deque[str] = deque(maxlen=8)

    def is_available(self) -> bool:
        return shutil.which(self._cfg.ffplay_bin) is not None

    def is_running(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def stderr_tail(self) -> list[str]:
        with self._stderr_lock:
            return list(self._stderr_tail)

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        try:
            stderr = proc.stderr
            if stderr is None:
                return
            for raw in stderr:
                line = str(raw).strip()
                if not line:
                    continue
                with self._stderr_lock:
                    self._stderr_tail.append(line)
        except Exception:
            return

    def start(self) -> bool:
        if self.is_running():
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
            "-ac",
            str(int(self._cfg.channels)),
            "-i",
            "-",
        ]
        with self._stderr_lock:
            self._stderr_tail.clear()
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception:
            self._proc = None
            return False
        proc = self._proc
        if proc is None:
            return False
        if proc.poll() is not None:
            # 起動に失敗してすぐ落ちた可能性が高い
            try:
                if proc.stderr is not None:
                    err = proc.stderr.read()
                    lines = [x.strip() for x in str(err).splitlines() if x.strip()]
                    with self._stderr_lock:
                        self._stderr_tail.extend(lines[-8:])
            except Exception:
                pass
            self.close()
            return False

        threading.Thread(target=self._drain_stderr, args=(proc,), daemon=True).start()
        return True

    def write(self, pcm_s16le: bytes) -> bool:
        proc = self._proc
        if proc is None:
            return False
        if proc.poll() is not None:
            self.close()
            return False
        if proc.stdin is None:
            return False
        try:
            proc.stdin.write(pcm_s16le)
            proc.stdin.flush()
            return True
        except Exception:
            self.close()
            return False

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
        try:
            if proc.stderr is not None:
                proc.stderr.close()
        except Exception:
            pass
