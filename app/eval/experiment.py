"""
リアルタイム実験セッション管理。

台本をTTSで読み上げながら、頷き/首振りジェスチャーを検知して
エージェントが相槌を選択する実験用セッション。

ジェスチャー入力は2モードに対応:
  - mock: Webページ上のボタンクリック（IMU不要）
  - imu : 実際のIMUセンサー（既存の検知ロジックを再利用）
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.agents.backchannel_graph import build_backchannel_graph
from app.audio.player import AudioPlayer
from app.core.catalog import load_catalog
from app.core.selector import find_audio_file
from app.core.types import BackchannelItem
from app.tts.openai_tts import synthesize_to_file


@dataclass
class Decision:
    call_id: str
    sentence_idx: int
    sentence_text: str
    gesture_hint: str          # "nod" or "shake"
    selected_id: str           # catalog id or "NONE"
    selected_text: str
    strength: int
    reason: str
    latency_ms: int
    ts: float


class ExperimentSession:
    def __init__(
        self,
        *,
        script: Dict[str, Any],
        oai_client: OpenAI,
        catalog: List[BackchannelItem],
        catalog_path: Path,
        backchannel_dir: Path,
        tts_cache_dir: Path,
        model: str = "gpt-5.2",
        tts_voice: str = "alloy",
        imu_port: Optional[str] = None,
        imu_baud: int = 115200,
        abs_threshold: float = 8.0,
        gyro_sigma: float = 3.0,
        max_age_s: float = 1.5,
        min_consecutive: int = 3,
        nod_axis: str = "gy",
        shake_axis: str = "gz",
    ) -> None:
        self.session_id = uuid.uuid4().hex[:12]
        self.script_id = script["id"]
        self.script_title = script.get("title", "")
        self.sentences: List[Dict] = script["sentences"]
        self.state = "ready"   # ready / generating / running / done
        self.current_idx = -1
        self.transcript_lines: List[str] = []
        self.decisions: List[Decision] = []

        self._oai_client = oai_client
        self._catalog = catalog
        self._catalog_by_id = {item.id: item for item in catalog}
        self._backchannel_dir = backchannel_dir
        self._tts_cache_dir = tts_cache_dir / script["id"]
        self._model = model
        self._tts_voice = tts_voice
        self._lock = threading.Lock()

        # Audio player (single instance per session)
        self._player = AudioPlayer()

        # Build and compile LangGraph agent
        graph = build_backchannel_graph(oai_client, model, catalog)
        from langgraph.checkpoint.memory import MemorySaver
        self._graph = graph.compile(checkpointer=MemorySaver())
        self._graph_config = {"configurable": {"thread_id": self.session_id}}

        # Repetition prevention
        self._avoid_ids: List[str] = []

        # IMU mode (optional)
        self._imu_port = imu_port
        self._imu_baud = imu_baud
        self._imu_abs_threshold = abs_threshold
        self._imu_gyro_sigma = gyro_sigma
        self._imu_max_age_s = max_age_s
        self._imu_min_consecutive = min_consecutive
        self._imu_nod_axis = nod_axis
        self._imu_shake_axis = shake_axis
        self._signal_queue: queue.Queue = queue.Queue()

        # Pre-generate TTS in background thread
        self.tts_ready = False
        self.tts_error: Optional[str] = None
        self.state = "generating"
        threading.Thread(
            target=self._pregenerate_tts, daemon=True, name=f"tts-gen-{self.session_id}"
        ).start()

    # ── TTS generation ───────────────────────────────────────────────────

    def _pregenerate_tts(self) -> None:
        try:
            self._tts_cache_dir.mkdir(parents=True, exist_ok=True)
            for i, sentence in enumerate(self.sentences):
                out_path = self._tts_cache_dir / f"{i:03d}.mp3"
                if not out_path.exists():
                    synthesize_to_file(
                        self._oai_client,
                        sentence["text"],
                        out_path,
                        voice=self._tts_voice,
                        response_format="mp3",
                    )
            self.tts_ready = True
            with self._lock:
                self.state = "ready"
        except Exception as e:
            self.tts_error = str(e)
            self.tts_ready = True  # allow advance even if TTS fails
            with self._lock:
                self.state = "ready"

    # ── Session lifecycle ─────────────────────────────────────────────────

    def start_imu(self) -> None:
        """Start real IMU reading threads (optional)."""
        if not self._imu_port:
            return
        from app.imu.buffer import ImuBuffer, ImuSample
        from app.imu.reader import read_imu_lines
        from app.imu.signal_store import HumanSignalStore
        from app.runtime.listener_session import human_signal_loop, imu_loop

        imu_buffer = ImuBuffer(max_seconds=60.0)
        store = HumanSignalStore(max_episodes=10, max_episode_age_s=10.0)

        # IMU reader thread
        def _imu_thread():
            imu_loop(self._imu_port, self._imu_baud, imu_buffer)

        # Signal detection thread (reuses existing logic)
        def _signal_thread():
            human_signal_loop(
                imu_buffer,
                store,
                calibration=None,
                gyro_sigma=self._imu_gyro_sigma,
                abs_threshold=self._imu_abs_threshold,
                max_age_s=self._imu_max_age_s,
                min_consecutive_above=self._imu_min_consecutive,
                nod_axis=self._imu_nod_axis,
                shake_axis=self._imu_shake_axis,
                event_queue=self._signal_queue,
            )

        # Gesture event consumer
        def _consumer_thread():
            while self.state != "done":
                try:
                    ev = self._signal_queue.get(timeout=0.5)
                    hint = ev.get("signal", {}).get("gesture_hint", "")
                    nod_score = (
                        ev.get("signal", {})
                        .get("motion_features", {})
                        .get("nod_likelihood_score", 4)
                    )
                    if hint in ("nod", "shake") and self.state == "running":
                        self._run_agent(hint, nod_score)
                except queue.Empty:
                    continue

        for fn, name in [
            (_imu_thread, "imu-reader"),
            (_signal_thread, "imu-signal"),
            (_consumer_thread, "imu-consumer"),
        ]:
            threading.Thread(target=fn, daemon=True, name=f"{name}-{self.session_id}").start()

    # ── Playback control ──────────────────────────────────────────────────

    def advance(self) -> bool:
        """Play the next sentence. Returns False if no more sentences."""
        with self._lock:
            if self.state == "done":
                return False
            next_idx = self.current_idx + 1
            if next_idx >= len(self.sentences):
                self.state = "done"
                return False
            self.current_idx = next_idx
            self.state = "running"

        sentence = self.sentences[self.current_idx]
        text = sentence["text"]
        with self._lock:
            self.transcript_lines.append(text)

        audio_path = self._tts_cache_dir / f"{self.current_idx:03d}.mp3"
        if audio_path.exists():
            threading.Thread(
                target=self._player.play_music_blocking,
                args=(audio_path,),
                daemon=True,
                name=f"tts-play-{self.session_id}",
            ).start()
        return True

    def inject_gesture(self, gesture_hint: str) -> None:
        """Simulate a gesture (mock mode) and run the agent."""
        if self.state not in ("running",):
            return
        threading.Thread(
            target=self._run_agent,
            args=(gesture_hint, 4),
            daemon=True,
            name=f"agent-{self.session_id}",
        ).start()

    # ── Agent invocation ──────────────────────────────────────────────────

    def _run_agent(self, gesture_hint: str, nod_score: int = 4) -> None:
        t0 = time.time()
        with self._lock:
            utterance = self.transcript_lines[-1] if self.transcript_lines else ""
            context = "\n".join(self.transcript_lines[-8:])
            avoid = list(self._avoid_ids[-3:])
            sentence_idx = self.current_idx

        directory_allowlist = ["positive"] if gesture_hint == "nod" else ["negative"]

        imu_text = f"ジェスチャー={gesture_hint} スコア={nod_score}/6"
        imu = {
            "human_signal": {
                "gesture_hint": gesture_hint,
                "motion_features": {"nod_likelihood_score": nod_score},
            }
        }
        timing = {
            "is_boundary": True,
            "speaker_speaking": False,
            "speaker_pause_like_boundary": True,
            "seconds_since_signal": 0.1,
            "speaker_silence_ms": 500,
            "transcript_latest_age_s": 0.1,
        }
        state = {
            "utterance": utterance,
            "utterance_t_sec": sentence_idx,
            "imu": imu,
            "imu_text": imu_text,
            "audio_state": {"music_playing": False, "effect_playing": False},
            "recent_backchannel": {},
            "transcript_context": context,
            "timing": timing,
            "directory_allowlist": directory_allowlist,
            "avoid_ids": avoid,
            "candidates": [],
            "selection": {},
            "selected_id": "",
            "errors": [],
        }

        try:
            result = self._graph.invoke(state, config=self._graph_config)
        except Exception as e:
            result = {"selected_id": "NONE", "reason": f"error: {e}", "errors": [str(e)]}

        latency_ms = int((time.time() - t0) * 1000)
        selected_id = result.get("selected_id", "NONE")
        catalog_item = self._catalog_by_id.get(selected_id)
        selected_text = catalog_item.text if catalog_item else "送信なし"
        strength = catalog_item.strength if catalog_item else 0
        reason = result.get("reason", "") or (
            result.get("selection", {}) or {}
        ).get("reason", "")

        dec = Decision(
            call_id=uuid.uuid4().hex[:12],
            sentence_idx=sentence_idx,
            sentence_text=utterance,
            gesture_hint=gesture_hint,
            selected_id=selected_id,
            selected_text=selected_text,
            strength=strength,
            reason=str(reason),
            latency_ms=latency_ms,
            ts=time.time(),
        )

        with self._lock:
            self.decisions.append(dec)
            # Update avoid_ids
            if selected_id != "NONE":
                self._avoid_ids.append(selected_id)
                if len(self._avoid_ids) > 6:
                    self._avoid_ids = self._avoid_ids[-6:]

        # Play backchannel audio if something was selected
        if catalog_item:
            audio_path = find_audio_file(self._backchannel_dir, catalog_item)
            if audio_path:
                self._player.play_effect(audio_path, interrupt=False)

    # ── State snapshot ────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "script_id": self.script_id,
                "script_title": self.script_title,
                "state": self.state,
                "tts_ready": self.tts_ready,
                "tts_error": self.tts_error,
                "current_idx": self.current_idx,
                "total": len(self.sentences),
                "transcript": list(self.transcript_lines),
                "decisions": [
                    {
                        "call_id": d.call_id,
                        "sentence_idx": d.sentence_idx,
                        "sentence_text": d.sentence_text,
                        "gesture_hint": d.gesture_hint,
                        "selected_id": d.selected_id,
                        "selected_text": d.selected_text,
                        "strength": d.strength,
                        "reason": d.reason,
                        "latency_ms": d.latency_ms,
                    }
                    for d in self.decisions
                ],
            }


# ── Session store (in-memory, single server process) ──────────────────────────

_sessions: Dict[str, ExperimentSession] = {}
_sessions_lock = threading.Lock()


def create_session(**kwargs) -> ExperimentSession:
    sess = ExperimentSession(**kwargs)
    with _sessions_lock:
        _sessions[sess.session_id] = sess
    return sess


def get_session(session_id: str) -> Optional[ExperimentSession]:
    with _sessions_lock:
        return _sessions.get(session_id)
