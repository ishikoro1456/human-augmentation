#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                obj["_source_path"] = str(path)
                events.append(obj)
    return events


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value.replace("\t", " ").replace("\n", " ").strip()
    try:
        return json.dumps(value, ensure_ascii=False).replace("\t", " ").replace("\n", " ").strip()
    except Exception:
        return str(value).replace("\t", " ").replace("\n", " ").strip()


@dataclass
class CallSummary:
    experiment_id: str
    call_id: str
    mode: str = ""

    # listener (decision side)
    agent_call_ts: float | None = None
    decision_point: str = ""
    is_boundary: bool | None = None
    has_signal: bool | None = None
    speaker_speaking: bool | None = None
    speaker_silence_ms: int | None = None
    boundary_text: str = ""

    decision_action: str = ""
    decision_latency_ms: int | None = None
    decision_reason: str = ""

    planned: bool | None = None
    wait_ms: int | None = None

    selected_id: str = ""
    selected_text: str = ""

    listener_sent_ts: float | None = None
    listener_play_ts: float | None = None

    # talker (playback side)
    talker_received_ts: float | None = None
    talker_play_ts: float | None = None
    talker_played: bool | None = None
    talker_audio_path: str = ""

    extra: Dict[str, Any] = field(default_factory=dict)

    def sort_ts(self) -> float:
        for v in (self.agent_call_ts, self.listener_sent_ts, self.listener_play_ts, self.talker_received_ts, self.talker_play_ts):
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0


def _call_key(exp_id: str, call_id: str) -> Tuple[str, str]:
    return (str(exp_id), str(call_id))


def main() -> None:
    parser = argparse.ArgumentParser(description="trace.jsonl を相槌中心のTSVに要約します")
    parser.add_argument("inputs", nargs="+", help="trace.jsonl（listener/talker どちらも可）")
    parser.add_argument("--out", default="data/logs/backchannel_summary.tsv")
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    events: List[Dict[str, Any]] = []
    for p in input_paths:
        if p.exists():
            events.extend(_load_jsonl(p))

    events.sort(key=lambda e: float(e.get("ts", 0.0) or 0.0))

    exp_mode: Dict[str, str] = {}
    for ev in events:
        if str(ev.get("type", "")) == "listener_session_start":
            exp = str(ev.get("experiment_id", "") or "").strip()
            if not exp:
                continue
            if exp not in exp_mode:
                exp_mode[exp] = str(ev.get("mode", "") or "").strip()

    calls: Dict[Tuple[str, str], CallSummary] = {}

    def get_call(exp: str, call_id: str) -> CallSummary:
        key = _call_key(exp, call_id)
        if key not in calls:
            calls[key] = CallSummary(experiment_id=exp, call_id=call_id, mode=exp_mode.get(exp, ""))
        return calls[key]

    for ev in events:
        exp = str(ev.get("experiment_id", "") or "").strip()
        call_id = str(ev.get("call_id", "") or "").strip()
        ev_type = str(ev.get("type", "") or "")
        role = str(ev.get("role", "") or "")

        # call_id がないイベントはスキップ（相槌中心の要約なので）
        if not exp or not call_id:
            continue

        row = get_call(exp, call_id)

        if ev_type == "agent_call":
            row.agent_call_ts = float(ev.get("ts", row.agent_call_ts or 0.0))
            row.boundary_text = str(ev.get("utterance", "") or "").strip()
            timing = ev.get("timing", {})
            if isinstance(timing, dict):
                row.decision_point = str(timing.get("decision_point", "") or "").strip()
                row.is_boundary = bool(timing.get("is_boundary")) if "is_boundary" in timing else row.is_boundary
                row.has_signal = bool(timing.get("has_signal")) if "has_signal" in timing else row.has_signal
                row.speaker_speaking = bool(timing.get("speaker_speaking")) if "speaker_speaking" in timing else row.speaker_speaking
                sm = timing.get("speaker_silence_ms")
                if isinstance(sm, int):
                    row.speaker_silence_ms = sm

        elif ev_type == "agent_result":
            row.decision_latency_ms = int(ev.get("latency_ms")) if isinstance(ev.get("latency_ms"), int) else row.decision_latency_ms
            row.decision_reason = str(ev.get("reason", "") or "").strip()
            decision = ev.get("decision", {})
            if isinstance(decision, dict):
                row.decision_action = str(decision.get("action", "") or "").strip()
            sid = str(ev.get("selected_id", "") or "").strip()
            if sid:
                row.selected_id = sid

        elif ev_type == "planned_set":
            row.planned = True
            planned = ev.get("planned", {})
            if isinstance(planned, dict):
                sid = str(planned.get("selected_id", "") or "").strip()
                if sid:
                    row.selected_id = sid
                row.selected_text = str(planned.get("selected_text", "") or "").strip() or row.selected_text
                wm = planned.get("wait_ms")
                if isinstance(wm, int):
                    row.wait_ms = wm

        elif ev_type == "backchannel_sent":
            row.listener_sent_ts = float(ev.get("ts", row.listener_sent_ts or 0.0))
            row.planned = bool(ev.get("planned")) if "planned" in ev else row.planned
            row.selected_id = str(ev.get("id", "") or "").strip() or row.selected_id
            row.selected_text = str(ev.get("text", "") or "").strip() or row.selected_text
            row.decision_reason = str(ev.get("reason", "") or "").strip() or row.decision_reason
            lm = ev.get("latency_ms")
            if isinstance(lm, int):
                row.decision_latency_ms = lm

        elif ev_type == "backchannel_play" and role != "talker":
            # listener 側のイベント（実質: 送った/ローカル再生した）
            row.listener_play_ts = float(ev.get("ts", row.listener_play_ts or 0.0))
            row.planned = bool(ev.get("planned")) if "planned" in ev else row.planned
            row.selected_id = str(ev.get("selected_id", "") or "").strip() or row.selected_id
            row.selected_text = str(ev.get("selected_text", "") or "").strip() or row.selected_text
            if row.call_id.startswith("human-") and not row.decision_action:
                row.decision_action = "HUMAN"

        elif ev_type == "backchannel_received":
            row.talker_received_ts = float(ev.get("ts", row.talker_received_ts or 0.0))
            row.planned = bool(ev.get("planned")) if "planned" in ev else row.planned
            row.selected_id = str(ev.get("id", "") or "").strip() or row.selected_id
            row.selected_text = str(ev.get("text", "") or "").strip() or row.selected_text
            row.decision_reason = str(ev.get("reason", "") or "").strip() or row.decision_reason
            lm = ev.get("latency_ms")
            if isinstance(lm, int):
                row.decision_latency_ms = lm

        elif ev_type == "backchannel_play" and role == "talker":
            row.talker_play_ts = float(ev.get("ts", row.talker_play_ts or 0.0))
            row.talker_played = bool(ev.get("played")) if "played" in ev else row.talker_played
            row.talker_audio_path = str(ev.get("audio_path", "") or "").strip() or row.talker_audio_path
            row.selected_id = str(ev.get("id", "") or "").strip() or row.selected_id
            row.selected_text = str(ev.get("text", "") or "").strip() or row.selected_text

        # talker 側の role が入っていない古いログ向け（最低限）
        elif ev_type == "backchannel_play" and role == "":
            row.talker_play_ts = float(ev.get("ts", row.talker_play_ts or 0.0))
            row.talker_played = bool(ev.get("played")) if "played" in ev else row.talker_played
            row.talker_audio_path = str(ev.get("audio_path", "") or "").strip() or row.talker_audio_path

    rows = sorted(calls.values(), key=lambda r: r.sort_ts())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "experiment_id",
        "mode",
        "call_id",
        "ts",
        "decision_point",
        "is_boundary",
        "has_signal",
        "speaker_speaking",
        "speaker_silence_ms",
        "boundary_text",
        "decision_action",
        "planned",
        "wait_ms",
        "selected_id",
        "selected_text",
        "decision_latency_ms",
        "decision_reason",
        "listener_sent_ts",
        "listener_play_ts",
        "talker_received_ts",
        "talker_play_ts",
        "talker_played",
        "talker_audio_path",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        for r in rows:
            ts = r.sort_ts()
            writer.writerow(
                [
                    _clean_cell(r.experiment_id),
                    _clean_cell(r.mode),
                    _clean_cell(r.call_id),
                    _clean_cell(round(float(ts), 3)),
                    _clean_cell(r.decision_point),
                    _clean_cell(r.is_boundary),
                    _clean_cell(r.has_signal),
                    _clean_cell(r.speaker_speaking),
                    _clean_cell(r.speaker_silence_ms),
                    _clean_cell(r.boundary_text),
                    _clean_cell(r.decision_action),
                    _clean_cell(r.planned),
                    _clean_cell(r.wait_ms),
                    _clean_cell(r.selected_id),
                    _clean_cell(r.selected_text),
                    _clean_cell(r.decision_latency_ms),
                    _clean_cell(r.decision_reason),
                    _clean_cell(r.listener_sent_ts),
                    _clean_cell(r.listener_play_ts),
                    _clean_cell(r.talker_received_ts),
                    _clean_cell(r.talker_play_ts),
                    _clean_cell(r.talker_played),
                    _clean_cell(r.talker_audio_path),
                ]
            )

    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()

