from __future__ import annotations

import time
from datetime import timedelta
from typing import Optional

from app.runtime.status import SessionStatus, StatusStore


def _fmt_age(ts: Optional[float], now: float) -> str:
    if ts is None:
        return "-"
    sec = max(0.0, now - ts)
    if sec < 10:
        return f"{sec:0.1f}s"
    if sec < 60:
        return f"{sec:0.0f}s"
    return str(timedelta(seconds=int(sec)))


def _render(status: SessionStatus):
    from rich.align import Align
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    now = time.time()

    def _fold(text: str) -> Text:
        return Text(text, overflow="fold", no_wrap=False)

    transcript_body = Text("", overflow="fold", no_wrap=False)
    if status.transcript.current_text:
        t = status.transcript.current_t_sec
        t_str = f"[{t:04d}s]" if t is not None else ""
        transcript_body.append(f"再生中 {t_str}\n", style="bold")
        transcript_body.append(status.transcript.current_text)
    else:
        if status.transcript.last_boundary_text:
            t = status.transcript.last_boundary_t_sec
            t_str = f"[{t:04d}s]" if t is not None else ""
            transcript_body.append(f"判断点 {t_str}\n", style="bold")
            transcript_body.append(status.transcript.last_boundary_text)
        else:
            transcript_body.append("（再生待ち）")
    if status.transcript.spoken_tail:
        transcript_body.append("\n\n直近:\n", style="bold")
        for line in status.transcript.spoken_tail[-4:]:
            transcript_body.append("・")
            transcript_body.append(line)
            transcript_body.append("\n")

    transcript_panel = Panel(
        transcript_body,
        title=f"Transcript (spoken={status.transcript.spoken_count})",
        border_style="cyan",
    )

    sensors_table = Table.grid(padding=(0, 1))
    sensors_table.add_column(style="bold")
    sensors_table.add_column()
    still = status.calibration.still_summary or "-"
    active = status.calibration.active_summary or "-"
    sensors_table.add_row("still", _fold(still))
    sensors_table.add_row("active", _fold(active))
    if status.calibration.gesture_summaries:
        for line in status.calibration.gesture_summaries[:3]:
            sensors_table.add_row("gesture", _fold(line))
    if status.calibration.gesture_axis_map:
        sensors_table.add_row("axis", _fold(status.calibration.gesture_axis_map))
    if status.calibration.warnings:
        sensors_table.add_row("warn", _fold(", ".join(status.calibration.warnings)))
    sensors_table.add_row("age", _fold(_fmt_age(status.calibration.finished_at, now)))
    if status.calibration.gesture_finished_at is not None:
        sensors_table.add_row("age_g", _fold(_fmt_age(status.calibration.gesture_finished_at, now)))

    sensors_table.add_row("imu_age", _fold(_fmt_age(status.imu.last_ts, now)))
    imu_text = status.imu.last_motion_text.strip() if status.imu.last_motion_text else "-"
    sensors_table.add_row("imu_raw", _fold(imu_text))
    if status.imu.last_human_signal:
        sensors_table.add_row("signal", _fold(status.imu.last_human_signal))
    if status.imu.last_human_signal_used:
        sensors_table.add_row("used", _fold(status.imu.last_human_signal_used))
    sensors_panel = Panel(sensors_table, title="Sensors", border_style="magenta")

    agent_table = Table.grid(padding=(0, 1))
    agent_table.add_column(style="bold")
    agent_table.add_column()
    agent_table.add_row("choice", status.agent.last_choice_id or "-")
    if status.agent.last_choice_text:
        agent_table.add_row("text", _fold(status.agent.last_choice_text))
    if status.agent.last_latency_ms is not None:
        agent_table.add_row("latency", f"{status.agent.last_latency_ms}ms")
    agent_table.add_row("age", _fold(_fmt_age(status.agent.last_ts, now)))
    if status.agent.last_reason:
        agent_table.add_row("reason", _fold(status.agent.last_reason))

    bc = status.audio.last_backchannel_path
    played = status.audio.last_backchannel_played
    tr = status.audio.last_transcript_path
    agent_table.add_row("backchannel", _fold(str(bc) if bc else "-"))
    agent_table.add_row("played", _fold("-" if played is None else ("yes" if played else "no")))
    agent_table.add_row("transcript", _fold(str(tr) if tr else "-"))
    decision_panel = Panel(agent_table, title="Decision", border_style="green")

    logs = status.logs[-20:] if status.logs else []
    log_text = Text("\n".join(logs) if logs else "（ログなし）", overflow="fold", no_wrap=False)
    log_panel = Panel(log_text, title="Log", border_style="white")

    layout = Layout()
    layout.split_column(
        Layout(name="body", ratio=1),
        Layout(name="log", size=24),
    )
    layout["body"].split_row(
        Layout(name="transcript", ratio=2),
        Layout(name="side", ratio=1),
    )
    layout["transcript"].update(transcript_panel)
    layout["side"].split_column(
        Layout(name="sensors", ratio=1),
        Layout(name="decision", ratio=1),
    )
    layout["side"]["sensors"].update(sensors_panel)
    layout["side"]["decision"].update(decision_panel)
    layout["log"].update(log_panel)

    return Align.center(layout, vertical="top")


def run_dashboard(status_store: StatusStore, *, refresh_hz: int = 12) -> None:
    from rich.console import Console
    from rich.live import Live

    console = Console()
    with Live(
        _render(status_store.snapshot()),
        console=console,
        refresh_per_second=refresh_hz,
        screen=True,
    ) as live:
        while True:
            live.update(_render(status_store.snapshot()))
            time.sleep(1.0 / max(1, refresh_hz))
