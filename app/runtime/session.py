import queue
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from openai import OpenAI

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # langgraph のバージョン差異に備える
    InMemorySaver = None

from app.agents.backchannel_graph import build_backchannel_graph
from app.audio.player import AudioPlayer
from app.core.catalog import load_catalog
from app.core.selector import find_audio_file
from app.imu.detector import MotionDetector, MotionSnapshot, format_motion
from app.imu.reader import read_imu_lines
from app.transcript.speaker import TranscriptSpeaker
from app.transcript.timeline import TranscriptTimeline


def imu_loop(
    port: str,
    baud: int,
    detector: MotionDetector,
    out_queue: "queue.Queue[MotionSnapshot]",
    debug: bool = False,
) -> None:
    last_log = 0.0
    latest_snapshot: Optional[MotionSnapshot] = None
    for data in read_imu_lines(port, baud):
        snapshot = detector.update(data, now=time.time())
        latest_snapshot = snapshot
        if debug:
            now = time.time()
            if now - last_log > 1.0 and latest_snapshot:
                print(f"IMU受信中: {format_motion(latest_snapshot)}")
                last_log = now
        if snapshot.event != "none":
            out_queue.put(snapshot)


def _motion_to_dict(snapshot: Optional[MotionSnapshot]) -> Dict[str, object]:
    if snapshot is None:
        return {"event": "none"}
    return snapshot.to_dict()


def run_session(
    catalog_path: Path,
    audio_dir: Path,
    port: str,
    baud: int,
    model: str,
    thread_id: str,
    debug_imu: bool = False,
    transcript_path: Optional[Path] = None,
    transcript_start_sec: int = 0,
    debug_transcript: bool = False,
    debug_agent: bool = False,
    tts_model: str = "gpt-4o-mini-tts",
    tts_voice: str = "alloy",
    tts_format: str = "mp3",
    tts_cache_dir: Optional[Path] = None,
) -> None:
    items = load_catalog(catalog_path)
    timeline = None
    if transcript_path and transcript_path.exists():
        timeline = TranscriptTimeline.from_file(transcript_path)

    client = OpenAI()
    checkpointer = InMemorySaver() if InMemorySaver else None
    graph = build_backchannel_graph(client, model, items).compile(checkpointer=checkpointer)

    detector = MotionDetector()
    event_queue: "queue.Queue[MotionSnapshot]" = queue.Queue()
    threading.Thread(
        target=imu_loop,
        args=(port, baud, detector, event_queue, debug_imu),
        daemon=True,
    ).start()

    player = AudioPlayer()
    speaker = None
    if timeline:
        cache_dir = tts_cache_dir or Path("data/tts_cache")
        speaker = TranscriptSpeaker(
            timeline=timeline,
            client=client,
            player=player,
            cache_dir=cache_dir,
            model=tts_model,
            voice=tts_voice,
            response_format=tts_format,
            start_sec=transcript_start_sec,
        )
        speaker.start()

    while True:
        snapshot = event_queue.get()
        motion_text = format_motion(snapshot)
        motion_dict = _motion_to_dict(snapshot)
        if speaker:
            transcript_context = speaker.get_spoken_context()
            current_seg = speaker.get_current()
            utterance = current_seg.text if current_seg else ""
        else:
            transcript_context = "文字起こしはまだありません"
            utterance = ""

        if debug_transcript and timeline:
            if speaker and current_seg:
                print(f"文字起こし: [{current_seg.t_sec:04d}s] {current_seg.text}")
            else:
                print("文字起こし: まだありません")
            print("文字起こしコンテクスト:")
            print(transcript_context)

        if not utterance:
            print("文字起こしがまだ無いので、反応を作りません。")
            continue

        result = graph.invoke(
            {
                "utterance": utterance,
                "motion": motion_dict,
                "motion_text": motion_text,
                "transcript_context": transcript_context,
                "candidates": [],
                "selection": {},
                "selected_id": "",
                "errors": [],
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        selected_id = str(result.get("selected_id", ""))
        selection = result.get("selection", {})
        reason = ""
        if isinstance(selection, dict):
            reason = str(selection.get("reason", ""))
        selected_item = next((item for item in items if item.id == selected_id), None)
        if not selected_item:
            print("相槌の選択に失敗しました。")
            continue

        audio_path = find_audio_file(audio_dir, selected_item)
        if not audio_path:
            print("音声ファイルが見つかりません。")
            continue

        print(
            f"選択: {selected_item.directory} s{selected_item.strength} "
            f"n{selected_item.nod} {selected_item.text}"
        )
        if debug_agent:
            print(f"IMU: {motion_text}")
            if reason:
                print(f"理由: {reason}")
        print(f"再生: {audio_path}")
        player.play(audio_path)
