from pathlib import Path
import threading

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame


class AudioPlayer:
    def __init__(self) -> None:
        self._ready = False
        self._lock = threading.Lock()
        self._effect_channel: pygame.mixer.Channel | None = None

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        pygame.mixer.init()
        pygame.mixer.set_num_channels(8)
        self._effect_channel = pygame.mixer.Channel(0)
        self._ready = True

    def is_music_playing(self) -> bool:
        with self._lock:
            self._ensure_ready()
            return bool(pygame.mixer.music.get_busy())

    def is_effect_playing(self) -> bool:
        with self._lock:
            self._ensure_ready()
            channel = self._effect_channel
            return bool(channel and channel.get_busy())

    def get_music_volume(self) -> float:
        with self._lock:
            self._ensure_ready()
            return float(pygame.mixer.music.get_volume())

    def set_music_volume(self, volume: float) -> None:
        with self._lock:
            self._ensure_ready()
            pygame.mixer.music.set_volume(float(volume))

    def play_music_blocking(self, path: Path) -> None:
        with self._lock:
            self._ensure_ready()
            pygame.mixer.music.load(str(path))
            pygame.mixer.music.play()
        while self.is_music_playing():
            pygame.time.Clock().tick(30)

    def play_effect(self, path: Path, *, interrupt: bool = False) -> bool:
        with self._lock:
            self._ensure_ready()
            channel = self._effect_channel
            if not channel:
                return False
            if channel.get_busy():
                if not interrupt:
                    return False
                channel.stop()
            sound = pygame.mixer.Sound(str(path))
            channel.play(sound)
            return True

    def play(self, path: Path) -> None:
        self.play_music_blocking(path)

    def stop(self) -> None:
        if self._ready:
            pygame.mixer.music.stop()

    def estimate_duration_s(self, path: Path) -> float | None:
        with self._lock:
            self._ensure_ready()
            try:
                sound = pygame.mixer.Sound(str(path))
                return float(sound.get_length())
            except Exception:
                return None
