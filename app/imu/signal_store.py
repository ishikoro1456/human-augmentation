from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class HumanSignalSnapshot:
    updated_at: Optional[float]
    latest: Dict[str, object] = field(default_factory=dict)
    last_present_at: Optional[float] = None
    last_present_signal: Dict[str, object] = field(default_factory=dict)

    def is_recent(self, *, now: Optional[float] = None, hold_sec: float = 1.0) -> bool:
        if self.last_present_at is None:
            return False
        now_ts = time.time() if now is None else float(now)
        return (now_ts - float(self.last_present_at)) <= float(hold_sec)


@dataclass
class SignalEpisode:
    """1つの動きのエピソード（present: true が続いた期間）"""
    start_ts: float
    end_ts: float
    gesture_hint: str
    nod_likelihood_score: int
    gyro_mag_max: float
    signal_at_peak: Dict[str, object] = field(default_factory=dict)

    def duration_s(self) -> float:
        return max(0.0, self.end_ts - self.start_ts)

    def age_s(self, now: Optional[float] = None) -> float:
        now_ts = time.time() if now is None else float(now)
        return max(0.0, now_ts - self.end_ts)

    def to_dict(self) -> Dict[str, object]:
        return {
            "start_ts": round(self.start_ts, 3),
            "end_ts": round(self.end_ts, 3),
            "duration_s": round(self.duration_s(), 3),
            "gesture_hint": self.gesture_hint,
            "nod_likelihood_score": self.nod_likelihood_score,
            "gyro_mag_max": round(self.gyro_mag_max, 3),
        }


class HumanSignalStore:
    def __init__(self, *, max_episodes: int = 10, max_episode_age_s: float = 30.0) -> None:
        self._lock = threading.Lock()
        self._updated_at: Optional[float] = None
        self._latest: Dict[str, object] = {}
        self._last_present_at: Optional[float] = None
        self._last_present_signal: Dict[str, object] = {}

        # エピソード管理
        self._max_episodes = max_episodes
        self._max_episode_age_s = max_episode_age_s
        self._episodes: List[SignalEpisode] = []
        self._current_episode_start: Optional[float] = None
        self._current_episode_peak_signal: Dict[str, object] = {}
        self._current_episode_peak_mag: float = 0.0

    def update(self, *, ts: float, signal: Dict[str, object]) -> None:
        with self._lock:
            self._updated_at = float(ts)
            self._latest = dict(signal) if signal else {}
            present = bool(signal.get("present", False))

            if present:
                self._last_present_at = float(ts)
                self._last_present_signal = dict(signal)

                # エピソードの開始または継続
                if self._current_episode_start is None:
                    # 新しいエピソードの開始
                    self._current_episode_start = float(ts)
                    self._current_episode_peak_signal = dict(signal)
                    self._current_episode_peak_mag = float(signal.get("gyro_mag_max_1s", 0) or 0)
                else:
                    # エピソードの継続：ピークを更新
                    mag = float(signal.get("gyro_mag_max_1s", 0) or 0)
                    if mag > self._current_episode_peak_mag:
                        self._current_episode_peak_signal = dict(signal)
                        self._current_episode_peak_mag = mag
            else:
                # present: false になった → エピソードの終了
                if self._current_episode_start is not None:
                    self._finalize_current_episode(ts)

    def _finalize_current_episode(self, end_ts: float) -> None:
        """現在のエピソードを確定してリストに追加（ロック内で呼ぶ）"""
        if self._current_episode_start is None:
            return

        signal = self._current_episode_peak_signal
        motion_features = signal.get("motion_features", {})
        if not isinstance(motion_features, dict):
            motion_features = {}

        episode = SignalEpisode(
            start_ts=self._current_episode_start,
            end_ts=end_ts,
            gesture_hint=str(signal.get("gesture_hint", "other")),
            nod_likelihood_score=int(motion_features.get("nod_likelihood_score", 0) or 0),
            gyro_mag_max=self._current_episode_peak_mag,
            signal_at_peak=dict(signal),
        )
        self._episodes.append(episode)

        # 古いエピソードを削除
        now = end_ts
        self._episodes = [
            ep for ep in self._episodes
            if ep.age_s(now) <= self._max_episode_age_s
        ]
        # 最大数を超えたら古い順に削除
        if len(self._episodes) > self._max_episodes:
            self._episodes = self._episodes[-self._max_episodes:]

        # 現在のエピソードをリセット
        self._current_episode_start = None
        self._current_episode_peak_signal = {}
        self._current_episode_peak_mag = 0.0

    def snapshot(self) -> HumanSignalSnapshot:
        with self._lock:
            return HumanSignalSnapshot(
                updated_at=self._updated_at,
                latest=dict(self._latest),
                last_present_at=self._last_present_at,
                last_present_signal=dict(self._last_present_signal),
            )

    def get_episodes(
        self,
        *,
        since_ts: Optional[float] = None,
        max_age_s: Optional[float] = None,
        now: Optional[float] = None,
        include_current: bool = True,
    ) -> List[SignalEpisode]:
        """
        保持しているエピソードを取得する（削除しない）
        
        Args:
            since_ts: この時刻以降に終了したエピソードのみ
            max_age_s: この秒数以内に終了したエピソードのみ
            now: 現在時刻（指定しない場合は time.time()）
            include_current: 進行中のエピソードも含めるか
        """
        now_ts = time.time() if now is None else float(now)
        with self._lock:
            episodes = list(self._episodes)
            # 進行中のエピソードも含める
            if include_current and self._current_episode_start is not None:
                signal = self._current_episode_peak_signal
                motion_features = signal.get("motion_features", {})
                if not isinstance(motion_features, dict):
                    motion_features = {}
                current_ep = SignalEpisode(
                    start_ts=self._current_episode_start,
                    end_ts=now_ts,  # まだ終了していないので now を使う
                    gesture_hint=str(signal.get("gesture_hint", "other")),
                    nod_likelihood_score=int(motion_features.get("nod_likelihood_score", 0) or 0),
                    gyro_mag_max=self._current_episode_peak_mag,
                    signal_at_peak=dict(signal),
                )
                episodes.append(current_ep)

        if since_ts is not None:
            episodes = [ep for ep in episodes if ep.end_ts >= since_ts]
        if max_age_s is not None:
            episodes = [ep for ep in episodes if ep.age_s(now_ts) <= max_age_s]

        return episodes

    def consume_episodes(
        self,
        *,
        since_ts: Optional[float] = None,
        max_age_s: Optional[float] = None,
        now: Optional[float] = None,
    ) -> List[SignalEpisode]:
        """
        保持しているエピソードを取得して削除する
        進行中のエピソードも含めて消費し、進行中のエピソードはリセットする
        
        Args:
            since_ts: この時刻以降に終了したエピソードのみ消費
            max_age_s: この秒数以内に終了したエピソードのみ消費
            now: 現在時刻
        """
        now_ts = time.time() if now is None else float(now)
        with self._lock:
            # 条件に合うエピソードを取得
            consumed = []
            remaining = []
            for ep in self._episodes:
                matches = True
                if since_ts is not None and ep.end_ts < since_ts:
                    matches = False
                if max_age_s is not None and ep.age_s(now_ts) > max_age_s:
                    matches = False
                if matches:
                    consumed.append(ep)
                else:
                    remaining.append(ep)
            self._episodes = remaining

            # 進行中のエピソードも消費する
            if self._current_episode_start is not None:
                signal = self._current_episode_peak_signal
                motion_features = signal.get("motion_features", {})
                if not isinstance(motion_features, dict):
                    motion_features = {}
                current_ep = SignalEpisode(
                    start_ts=self._current_episode_start,
                    end_ts=now_ts,
                    gesture_hint=str(signal.get("gesture_hint", "other")),
                    nod_likelihood_score=int(motion_features.get("nod_likelihood_score", 0) or 0),
                    gyro_mag_max=self._current_episode_peak_mag,
                    signal_at_peak=dict(signal),
                )
                # 条件チェック
                matches = True
                if since_ts is not None and current_ep.end_ts < since_ts:
                    matches = False
                if max_age_s is not None and current_ep.age_s(now_ts) > max_age_s:
                    matches = False
                if matches:
                    consumed.append(current_ep)
                    # 進行中のエピソードをリセット
                    self._current_episode_start = None
                    self._current_episode_peak_signal = {}
                    self._current_episode_peak_mag = 0.0

            return consumed

    def summarize_episodes(
        self,
        episodes: List[SignalEpisode],
    ) -> Dict[str, object]:
        """
        エピソードのリストを要約する
        """
        if not episodes:
            return {
                "count": 0,
                "has_nod": False,
                "has_shake": False,
                "best_nod_score": 0,
                "episodes": [],
            }

        nods = [ep for ep in episodes if ep.gesture_hint == "nod"]
        shakes = [ep for ep in episodes if ep.gesture_hint == "shake"]
        best_nod_score = max((ep.nod_likelihood_score for ep in episodes), default=0)
        best_episode = max(episodes, key=lambda ep: ep.nod_likelihood_score)

        return {
            "count": len(episodes),
            "has_nod": len(nods) > 0,
            "has_shake": len(shakes) > 0,
            "nod_count": len(nods),
            "shake_count": len(shakes),
            "best_nod_score": best_nod_score,
            "best_episode": best_episode.to_dict(),
            "best_signal": best_episode.signal_at_peak,
            "episodes": [ep.to_dict() for ep in episodes],
        }
