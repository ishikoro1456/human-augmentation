from pathlib import Path
import threading

import pygame


class AudioPlayer:
    def __init__(self) -> None:
        self._ready = False
        self._lock = threading.Lock()

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        pygame.mixer.init()
        self._ready = True

    def play(self, path: Path) -> None:
        with self._lock:
            self._ensure_ready()
            pygame.mixer.music.load(str(path))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(30)

    def stop(self) -> None:
        if self._ready:
            pygame.mixer.music.stop()
