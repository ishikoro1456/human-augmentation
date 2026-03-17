"""
リアルタイム実験セッション管理。

同一台本を4段階で繰り返し、段階ごとに事後評価へ進む。
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from openai import OpenAI
else:
    OpenAI = Any

from app.audio.player import AudioPlayer
from app.core.types import BackchannelItem
from app.eval.stage_policy import (
    STAGE_CONFIGS,
    StageConfig,
    estimate_intensity_level,
    generate_stage_response,
    get_stage_config,
)
from app.tts.openai_tts import synthesize_to_file


@dataclass
class Decision:
    call_id: str
    stage_index: int
    stage_name: str
    sentence_idx: int
    sentence_text: str
    gesture_hint: str  # "nod" or "shake"
    intensity_1to5: int
    selected_id: str
    selected_text: str
    generated_text: str
    generation_mode: str  # fixed / ai_generated
    constraints_ok: bool
    reason: str
    latency_ms: int
    signal_confidence: float
    imu_features: Dict[str, Any]
    audio_path: str
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
        protocol: str = "imu_4stage_v1",
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
        self.protocol = str(protocol or "imu_4stage_v1")
        self.script_id = script["id"]
        self.script_title = script.get("title", "")
        self.sentences: List[Dict[str, Any]] = list(script["sentences"])

        # generating / ready / running / stage_review / done
        self.state = "ready"
        self.stage_index = 0
        self.current_idx = -1
        self.transcript_lines: List[str] = []
        self.decisions: List[Decision] = []
        self.events: List[Dict[str, Any]] = []

        self._oai_client = oai_client
        self._catalog = catalog
        self._catalog_by_id = {item.id: item for item in catalog}
        self._catalog_path = catalog_path
        self._backchannel_dir = backchannel_dir
        self._script_tts_cache_dir = tts_cache_dir / "scripts" / script["id"]
        self._response_tts_cache_dir = tts_cache_dir / "experiment" / self.session_id
        self._model = model
        self._tts_voice = tts_voice
        self._lock = threading.Lock()

        self._player = AudioPlayer()

        self._avoid_ids: List[str] = []
        self._decision_seq = 0

        self.stage_runs: List[Dict[str, Any]] = [
            {
                "stage_index": conf.index,
                "stage_key": conf.key,
                "stage_name": conf.name,
                "state": "pending",
                "started_at": None,
                "ended_at": None,
                "decision_count": 0,
            }
            for conf in STAGE_CONFIGS
        ]
        self.stage_runs[0]["state"] = "ready"
        self.stage_runs[0]["started_at"] = round(time.time(), 3)

        # IMU mode
        self._imu_port = imu_port
        self._imu_baud = imu_baud
        self._imu_abs_threshold = abs_threshold
        self._imu_gyro_sigma = gyro_sigma
        self._imu_max_age_s = max_age_s
        self._imu_min_consecutive = min_consecutive
        self._imu_nod_axis = nod_axis
        self._imu_shake_axis = shake_axis
        self._signal_queue: queue.Queue = queue.Queue()

        self.tts_ready = False
        self.tts_error: Optional[str] = None
        self.state = "generating"
        self._log_event("session_start", stage_index=0, protocol=self.protocol)
        threading.Thread(
            target=self._pregenerate_tts,
            daemon=True,
            name=f"tts-gen-{self.session_id}",
        ).start()

    def _log_event(self, event_type: str, **kwargs: Any) -> None:
        event = {
            "type": str(event_type),
            "ts": round(time.time(), 3),
            "session_id": self.session_id,
            "experiment_id": self.session_id,
            "stage_index": int(self.stage_index),
        }
        event.update(kwargs)
        with self._lock:
            self.events.append(event)
            if len(self.events) > 500:
                self.events = self.events[-500:]

    # ── TTS generation ───────────────────────────────────────────────────

    def _pregenerate_tts(self) -> None:
        try:
            self._script_tts_cache_dir.mkdir(parents=True, exist_ok=True)
            for i, sentence in enumerate(self.sentences):
                out_path = self._script_tts_cache_dir / f"{i:03d}.mp3"
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
            self._log_event("tts_ready")
        except Exception as e:
            self.tts_error = str(e)
            self.tts_ready = True
            with self._lock:
                self.state = "ready"
            self._log_event("tts_error", error=str(e))

    # ── Session lifecycle ─────────────────────────────────────────────────

    def start_imu(self) -> None:
        """Start real IMU reading threads (optional)."""
        if not self._imu_port:
            return
        from app.imu.buffer import ImuBuffer
        from app.imu.signal_store import HumanSignalStore
        from app.runtime.listener_session import human_signal_loop, imu_loop

        imu_buffer = ImuBuffer(max_seconds=60.0)
        store = HumanSignalStore(max_episodes=10, max_episode_age_s=10.0)

        def _imu_thread() -> None:
            imu_loop(self._imu_port, self._imu_baud, imu_buffer)

        def _signal_thread() -> None:
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

        def _consumer_thread() -> None:
            while True:
                with self._lock:
                    if self.state == "done":
                        return
                try:
                    ev = self._signal_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                signal = ev.get("signal", {}) if isinstance(ev, dict) else {}
                if not isinstance(signal, dict):
                    continue
                hint = str(signal.get("gesture_hint", ""))
                with self._lock:
                    is_running = self.state == "running"
                if hint in ("nod", "shake") and is_running:
                    self._run_stage_decision(signal=signal)

        for fn, name in [
            (_imu_thread, "imu-reader"),
            (_signal_thread, "imu-signal"),
            (_consumer_thread, "imu-consumer"),
        ]:
            threading.Thread(
                target=fn,
                daemon=True,
                name=f"{name}-{self.session_id}",
            ).start()

    # ── Playback control ──────────────────────────────────────────────────

    def _mark_stage_running_locked(self) -> None:
        run = self.stage_runs[self.stage_index]
        if run["state"] in ("pending", "ready"):
            run["state"] = "running"
        if run["started_at"] is None:
            run["started_at"] = round(time.time(), 3)

    def _mark_stage_review_locked(self) -> None:
        run = self.stage_runs[self.stage_index]
        run["state"] = "review_pending"
        run["ended_at"] = round(time.time(), 3)
        run["decision_count"] = sum(1 for d in self.decisions if d.stage_index == self.stage_index)

    def advance(self) -> bool:
        """Play the next sentence in current stage."""
        with self._lock:
            if self.state in ("done", "stage_review", "generating"):
                return False
            next_idx = self.current_idx + 1
            if next_idx >= len(self.sentences):
                self._mark_stage_review_locked()
                self.state = "stage_review"
                self._log_event("stage_end", stage_index=self.stage_index)
                return False
            self.current_idx = next_idx
            self.state = "running"
            self._mark_stage_running_locked()
            sentence = self.sentences[self.current_idx]
            text = str(sentence["text"])
            self.transcript_lines.append(text)

        audio_path = self._script_tts_cache_dir / f"{self.current_idx:03d}.mp3"
        if audio_path.exists():
            threading.Thread(
                target=self._player.play_music_blocking,
                args=(audio_path,),
                daemon=True,
                name=f"tts-play-{self.session_id}",
            ).start()

        self._log_event(
            "sentence_play",
            stage_index=self.stage_index,
            sentence_idx=self.current_idx,
            sentence_text=text,
        )
        return True

    def advance_stage(self) -> bool:
        """Move from review to next stage. Returns False when session is done."""
        with self._lock:
            if self.state != "stage_review":
                return False

            cur = self.stage_runs[self.stage_index]
            cur["state"] = "reviewed"

            if self.stage_index + 1 >= len(STAGE_CONFIGS):
                self.state = "done"
                self._log_event("session_done")
                return False

            self.stage_index += 1
            self.current_idx = -1
            self.transcript_lines = []
            self.state = "ready"
            nxt = self.stage_runs[self.stage_index]
            nxt["state"] = "ready"
            if nxt["started_at"] is None:
                nxt["started_at"] = round(time.time(), 3)

        self._log_event("stage_start", stage_index=self.stage_index)
        return True

    def mark_stage_review_submitted(self, *, stage_index: int, evaluator_id: str) -> None:
        self._log_event(
            "stage_review_submitted",
            stage_index=int(stage_index),
            evaluator_id=str(evaluator_id),
        )

    def inject_gesture(self, gesture_hint: str) -> None:
        """Simulate a gesture (mock mode)."""
        with self._lock:
            if self.state != "running":
                return
        signal = {
            "gesture_hint": str(gesture_hint),
            "motion_features": {
                "nod_likelihood_score": 4,
                "intensity_level_1to5": 3,
            },
            "signal_confidence_0to1": 0.75,
            "gyro_mag_max_1s": 12.0,
            "acc_delta_mag_1s": 0.2,
            "acc_axis_stability": 0.8,
            "tilt_return_score": 0.8,
        }
        threading.Thread(
            target=self._run_stage_decision,
            kwargs={"signal": signal},
            daemon=True,
            name=f"agent-{self.session_id}",
        ).start()

    # ── Agent invocation ──────────────────────────────────────────────────

    @staticmethod
    def _build_selected_id(stage: StageConfig, gesture_hint: str, intensity_1to5: int) -> str:
        direction = "YES" if str(gesture_hint) == "nod" else "NO"
        if stage.index == 0:
            return f"ST1_{direction}"
        if stage.index == 1:
            return f"ST2_{direction}_{intensity_1to5}"
        if stage.index == 2:
            return f"ST3_{direction}"
        return f"ST4_{direction}"

    @staticmethod
    def _to_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    def _extract_intensity(self, signal: Dict[str, Any], nod_score: int, signal_confidence: float) -> int:
        gi = signal.get("gesture_intensity", {})
        if isinstance(gi, dict):
            lvl = gi.get("level_1to5")
            if isinstance(lvl, (int, float)):
                return max(1, min(5, int(lvl)))
        motion = signal.get("motion_features", {})
        if isinstance(motion, dict):
            lvl2 = motion.get("intensity_level_1to5")
            if isinstance(lvl2, (int, float)):
                return max(1, min(5, int(lvl2)))
        return estimate_intensity_level(nod_score=nod_score, signal_confidence=signal_confidence)

    def _run_stage_decision(self, *, signal: Dict[str, Any]) -> None:
        t0 = time.time()
        with self._lock:
            if self.state != "running":
                return
            if self.current_idx < 0 or self.current_idx >= len(self.sentences):
                return
            stage = get_stage_config(self.stage_index)
            utterance = self.transcript_lines[-1] if self.transcript_lines else ""
            context = "\n".join(self.transcript_lines[-8:])
            sentence_idx = int(self.current_idx)

        gesture_hint = str(signal.get("gesture_hint", "other"))
        if gesture_hint not in ("nod", "shake"):
            return

        motion_features = signal.get("motion_features", {})
        if not isinstance(motion_features, dict):
            motion_features = {}

        nod_score = int(max(0, min(6, int(motion_features.get("nod_likelihood_score", 4) or 0))))
        signal_confidence = max(
            0.0,
            min(1.0, self._to_float(signal.get("signal_confidence_0to1", 0.6), 0.6)),
        )
        intensity_1to5 = self._extract_intensity(signal, nod_score=nod_score, signal_confidence=signal_confidence)

        imu_features = {
            "nod_likelihood_score": nod_score,
            "signal_confidence_0to1": signal_confidence,
            "gyro_mag_max_1s": self._to_float(signal.get("gyro_mag_max_1s", 0.0), 0.0),
            "acc_delta_mag_1s": self._to_float(signal.get("acc_delta_mag_1s", 0.0), 0.0),
            "acc_axis_stability": self._to_float(signal.get("acc_axis_stability", 0.0), 0.0),
            "tilt_return_score": self._to_float(signal.get("tilt_return_score", 0.0), 0.0),
        }

        stage_resp = generate_stage_response(
            client=self._oai_client,
            model=self._model,
            stage=stage,
            gesture_hint=gesture_hint,
            utterance=utterance,
            context=context,
            intensity_1to5=intensity_1to5,
            imu_features=imu_features,
        )

        latency_ms = int((time.time() - t0) * 1000)
        selected_id = self._build_selected_id(stage, gesture_hint, stage_resp.intensity_1to5)

        call_id = uuid.uuid4().hex[:12]
        response_dir = self._response_tts_cache_dir / stage.key
        response_dir.mkdir(parents=True, exist_ok=True)
        response_path = response_dir / f"{self._decision_seq:04d}_{call_id}.mp3"

        try:
            synthesize_to_file(
                self._oai_client,
                stage_resp.text,
                response_path,
                voice=self._tts_voice,
                response_format="mp3",
            )
            audio_path = str(response_path)
            played = self._player.play_effect(response_path, interrupt=False)
            if not played:
                # 再生中でもログは残す
                pass
        except Exception:
            audio_path = ""

        dec = Decision(
            call_id=call_id,
            stage_index=self.stage_index,
            stage_name=stage.name,
            sentence_idx=sentence_idx,
            sentence_text=utterance,
            gesture_hint=gesture_hint,
            intensity_1to5=stage_resp.intensity_1to5,
            selected_id=selected_id,
            selected_text=stage_resp.text,
            generated_text=stage_resp.text,
            generation_mode=stage_resp.generation_mode,
            constraints_ok=bool(stage_resp.constraints_ok),
            reason=str(stage_resp.reason),
            latency_ms=latency_ms,
            signal_confidence=signal_confidence,
            imu_features=imu_features,
            audio_path=audio_path,
            ts=time.time(),
        )

        with self._lock:
            self._decision_seq += 1
            self.decisions.append(dec)
            self._avoid_ids.append(selected_id)
            if len(self._avoid_ids) > 20:
                self._avoid_ids = self._avoid_ids[-20:]
            self.stage_runs[self.stage_index]["decision_count"] = (
                self.stage_runs[self.stage_index]["decision_count"] + 1
            )

        self._log_event(
            "stage_decision",
            stage_index=self.stage_index,
            stage_name=stage.name,
            call_id=dec.call_id,
            sentence_text=dec.sentence_text,
            gesture_hint=dec.gesture_hint,
            selected_id=dec.selected_id,
            selected_text=dec.selected_text,
            reason=dec.reason,
            generation_mode=dec.generation_mode,
            intensity_1to5=dec.intensity_1to5,
            signal_confidence=dec.signal_confidence,
            imu_features=dict(dec.imu_features),
            latency_ms=dec.latency_ms,
        )

    # ── Query helpers ─────────────────────────────────────────────────────

    def get_stage_decisions(self, stage_index: int) -> List[Decision]:
        with self._lock:
            return [d for d in self.decisions if d.stage_index == int(stage_index)]

    def get_current_stage_decisions(self) -> List[Decision]:
        return self.get_stage_decisions(self.stage_index)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            stage_conf = get_stage_config(self.stage_index)
            current_stage_decisions = [d for d in self.decisions if d.stage_index == self.stage_index]
            return {
                "session_id": self.session_id,
                "protocol": self.protocol,
                "script_id": self.script_id,
                "script_title": self.script_title,
                "state": self.state,
                "tts_ready": self.tts_ready,
                "tts_error": self.tts_error,
                "stage_index": self.stage_index,
                "stage_total": len(STAGE_CONFIGS),
                "stage_name": stage_conf.name,
                "stage_key": stage_conf.key,
                "stage_description": stage_conf.description,
                "stage_runs": list(self.stage_runs),
                "current_idx": self.current_idx,
                "total": len(self.sentences),
                "transcript": list(self.transcript_lines),
                "decisions": [
                    {
                        "call_id": d.call_id,
                        "stage_index": d.stage_index,
                        "stage_name": d.stage_name,
                        "sentence_idx": d.sentence_idx,
                        "sentence_text": d.sentence_text,
                        "gesture_hint": d.gesture_hint,
                        "intensity_1to5": d.intensity_1to5,
                        "selected_id": d.selected_id,
                        "selected_text": d.selected_text,
                        "generated_text": d.generated_text,
                        "generation_mode": d.generation_mode,
                        "constraints_ok": d.constraints_ok,
                        "reason": d.reason,
                        "latency_ms": d.latency_ms,
                        "signal_confidence": d.signal_confidence,
                        "imu_features": dict(d.imu_features),
                        "audio_path": d.audio_path,
                    }
                    for d in current_stage_decisions
                ],
                "all_decision_count": len(self.decisions),
                "all_decisions": [
                    {
                        "call_id": d.call_id,
                        "stage_index": d.stage_index,
                        "stage_name": d.stage_name,
                        "sentence_idx": d.sentence_idx,
                        "sentence_text": d.sentence_text,
                        "gesture_hint": d.gesture_hint,
                        "intensity_1to5": d.intensity_1to5,
                        "selected_id": d.selected_id,
                        "selected_text": d.selected_text,
                        "generated_text": d.generated_text,
                        "generation_mode": d.generation_mode,
                        "constraints_ok": d.constraints_ok,
                        "reason": d.reason,
                        "latency_ms": d.latency_ms,
                        "signal_confidence": d.signal_confidence,
                        "imu_features": dict(d.imu_features),
                        "audio_path": d.audio_path,
                    }
                    for d in self.decisions
                ],
                "events": list(self.events[-120:]),
            }


# ── Session store (in-memory) ───────────────────────────────────────────

_sessions: Dict[str, ExperimentSession] = {}
_sessions_lock = threading.Lock()


def create_session(**kwargs: Any) -> ExperimentSession:
    sess = ExperimentSession(**kwargs)
    with _sessions_lock:
        _sessions[sess.session_id] = sess
    return sess


def get_session(session_id: str) -> Optional[ExperimentSession]:
    with _sessions_lock:
        return _sessions.get(session_id)
