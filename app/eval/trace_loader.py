import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from app.core.catalog import load_catalog
from app.core.types import BackchannelItem
from .models import AgentDecision, SessionDetail, SessionSummary, SttSegment, TimelineItem


class TraceLoader:
    def __init__(self, trace_path: Path, catalog_path: Path) -> None:
        self._trace_path = trace_path
        self._catalog_path = catalog_path
        self._sessions: Dict[str, SessionDetail] = {}
        self._catalog: List[BackchannelItem] = []

    def load(self) -> None:
        self._catalog = load_catalog(self._catalog_path)
        catalog_by_id = {item.id: item for item in self._catalog}

        events_by_session: Dict[str, list] = defaultdict(list)
        with self._trace_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                exp_id = ev.get("experiment_id")
                if exp_id:
                    events_by_session[exp_id].append(ev)

        sessions = {}
        for exp_id, events in events_by_session.items():
            session_start = next(
                (e for e in events if e.get("type") == "listener_session_start"), None
            )
            mode = session_start.get("mode", "unknown") if session_start else "unknown"
            model = session_start.get("model", "unknown") if session_start else "unknown"
            start_ts = session_start.get("ts", 0.0) if session_start else 0.0

            # Only include sessions with LLM decisions
            if mode != "llm":
                continue

            stt_segments = sorted(
                [
                    SttSegment(
                        segment_id=e.get("segment_id", 0),
                        text=e.get("text", ""),
                        ts=e["ts"],
                    )
                    for e in events
                    if e.get("type") == "stt_segment"
                ],
                key=lambda s: s.ts,
            )

            agent_calls = {
                e["call_id"]: e
                for e in events
                if e.get("type") == "agent_call" and "call_id" in e
            }
            agent_results = {
                e["call_id"]: e
                for e in events
                if e.get("type") == "agent_result" and "call_id" in e
            }

            decisions = []
            for call_id, call in agent_calls.items():
                result = agent_results.get(call_id)
                if not result:
                    continue

                selected_id = result.get("selected_id", "NONE")
                catalog_item = catalog_by_id.get(selected_id)
                if catalog_item:
                    selected_text = catalog_item.text
                    strength = catalog_item.strength
                elif selected_id == "NONE":
                    selected_text = "送信なし"
                    strength = 0
                else:
                    selected_text = selected_id
                    strength = 0

                imu = call.get("imu", {})
                human_signal = imu.get("human_signal", {})
                motion_features = human_signal.get("motion_features", {})
                timing = call.get("timing", {})

                decisions.append(
                    AgentDecision(
                        call_id=call_id,
                        experiment_id=exp_id,
                        index=0,
                        utterance=call.get("utterance", ""),
                        transcript_context=call.get("transcript_context", ""),
                        gesture_hint=human_signal.get("gesture_hint", "other"),
                        nod_likelihood_score=motion_features.get("nod_likelihood_score", 0),
                        directory_allowlist=call.get("directory_allowlist", []),
                        is_boundary=timing.get("is_boundary", False),
                        speaker_speaking=timing.get("speaker_speaking", False),
                        speaker_silence_ms=timing.get("speaker_silence_ms", 0),
                        selected_id=selected_id,
                        selected_text=selected_text,
                        strength=strength,
                        reason=result.get("reason", ""),
                        latency_ms=result.get("latency_ms", 0),
                        ts=call["ts"],
                        stage_index=int(call.get("stage_index", -1) or -1),
                        stage_name=str(call.get("stage_name", "") or ""),
                        intensity_1to5=int(result.get("intensity_1to5", 0) or 0),
                        generated_text=str(result.get("generated_text", "") or ""),
                        generation_mode=str(result.get("generation_mode", "") or ""),
                        signal_confidence=float(result.get("signal_confidence", 0.0) or 0.0),
                        imu_features=call.get("imu_features", {}) if isinstance(call.get("imu_features"), dict) else {},
                    )
                )

            # 4段階実験ログ（stage_decision）にも対応
            for e in events:
                if e.get("type") != "stage_decision":
                    continue
                call_id = str(e.get("call_id", "") or "")
                if not call_id:
                    continue
                decisions.append(
                    AgentDecision(
                        call_id=call_id,
                        experiment_id=exp_id,
                        index=0,
                        utterance=str(e.get("sentence_text", "") or ""),
                        transcript_context="",
                        gesture_hint=str(e.get("gesture_hint", "other") or "other"),
                        nod_likelihood_score=0,
                        directory_allowlist=[],
                        is_boundary=True,
                        speaker_speaking=False,
                        speaker_silence_ms=0,
                        selected_id=str(e.get("selected_id", "") or ""),
                        selected_text=str(e.get("selected_text", "") or ""),
                        strength=0,
                        reason=str(e.get("reason", "") or ""),
                        latency_ms=int(e.get("latency_ms", 0) or 0),
                        ts=float(e.get("ts", 0.0) or 0.0),
                        stage_index=int(e.get("stage_index", -1) or -1),
                        stage_name=str(e.get("stage_name", "") or ""),
                        intensity_1to5=int(e.get("intensity_1to5", 0) or 0),
                        generated_text=str(e.get("selected_text", "") or ""),
                        generation_mode=str(e.get("generation_mode", "") or ""),
                        signal_confidence=float(e.get("signal_confidence", 0.0) or 0.0),
                        imu_features=e.get("imu_features", {}) if isinstance(e.get("imu_features"), dict) else {},
                    )
                )

            decisions.sort(key=lambda d: d.ts)
            for i, d in enumerate(decisions):
                d.index = i

            if not decisions:
                continue

            sessions[exp_id] = SessionDetail(
                experiment_id=exp_id,
                mode=mode,
                model=model,
                start_ts=start_ts,
                decisions=decisions,
                stt_segments=stt_segments,
            )

        self._sessions = sessions

    def get_sessions(self) -> List[SessionSummary]:
        return sorted(
            [
                SessionSummary(
                    experiment_id=s.experiment_id,
                    mode=s.mode,
                    model=s.model,
                    start_ts=s.start_ts,
                    decision_count=len(s.decisions),
                )
                for s in self._sessions.values()
            ],
            key=lambda s: s.start_ts,
            reverse=True,
        )

    def get_session(self, experiment_id: str) -> SessionDetail | None:
        return self._sessions.get(experiment_id)

    def get_catalog(self) -> List[BackchannelItem]:
        return self._catalog

    def build_timeline(self, session: SessionDetail) -> List[TimelineItem]:
        items: List[TimelineItem] = []
        for seg in session.stt_segments:
            items.append(TimelineItem(ts=seg.ts, kind="stt", stt=seg))
        for dec in session.decisions:
            items.append(TimelineItem(ts=dec.ts, kind="decision", decision=dec))
        items.sort(key=lambda x: x.ts)
        return items
