import json
from typing import Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from app.core.types import BackchannelItem


class AgentState(TypedDict):
    utterance: str
    utterance_t_sec: object
    imu: Dict[str, object]
    imu_text: str
    audio_state: Dict[str, object]
    recent_backchannel: Dict[str, object]
    transcript_context: str
    timing: Dict[str, object]
    directory_allowlist: List[str]
    avoid_ids: List[str]
    candidates: List[Dict[str, object]]
    selection: Dict[str, object]
    selected_id: str
    errors: List[str]


def _build_candidates(
    items: List[BackchannelItem],
) -> List[Dict[str, object]]:
    return [
        {
            "id": item.id,
            "directory": item.directory,
            "strength": item.strength,
            "nod": item.nod,
            "text": item.text,
        }
        for item in items
    ]


def _build_choice_schema(candidate_ids: List[str]) -> Dict[str, object]:
    """相槌を1回の呼び出しで選ぶ（NONE を含む）"""
    ids = ["NONE"] + list(candidate_ids)
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string", "enum": ids},
            "reason": {
                "type": "string",
                "description": "判断理由（20文字以内）",
            },
        },
        "required": ["id", "reason"],
        "additionalProperties": False,
    }


def _fallback_id(items: List[BackchannelItem]) -> str:
    for item in items:
        if item.strength == 3 and item.nod == 3:
            return item.id
    return items[0].id


def _extract_motion_summary(imu: Dict[str, object]) -> Dict[str, object]:
    """IMUデータから動きの要約を抽出する"""
    human_signal = imu.get("human_signal", {})
    if not isinstance(human_signal, dict):
        human_signal = {}

    # 符号変化回数
    sign_changes = human_signal.get("axis_sign_changes_1s", {})
    if not isinstance(sign_changes, dict):
        sign_changes = {}

    # 主要軸の符号変化
    dominant_axis = human_signal.get("dominant_axis", "-")
    dominant_sign_changes = sign_changes.get(dominant_axis, 0) if dominant_axis != "-" else 0

    # ジェスチャーのヒント
    gesture_hint = human_signal.get("gesture_hint", "other")

    # 動きの強さ
    gesture_intensity = imu.get("gesture_intensity", {})
    if not isinstance(gesture_intensity, dict):
        gesture_intensity = {}
    intensity_level = gesture_intensity.get("level_1to5", 0)

    # 正規化された活動量
    normalized = imu.get("normalized_activity", {})
    if not isinstance(normalized, dict):
        normalized = {}

    # 新しい特徴量（motion_features）
    motion_features = human_signal.get("motion_features", {})
    if not isinstance(motion_features, dict):
        motion_features = {}

    return {
        "gesture_hint": gesture_hint,
        "dominant_axis": dominant_axis,
        "sign_changes_on_dominant_axis": dominant_sign_changes,
        "intensity_level_1to5": intensity_level,
        "normalized_gyro_mag_max": normalized.get("gyro_mag_max"),
        "present": human_signal.get("present", False),
        # 新しい特徴量
        "has_oscillation": motion_features.get("has_oscillation", False),
        "posture_returned": motion_features.get("posture_returned", True),
        "is_symmetric": motion_features.get("is_symmetric", True),
        "motion_duration_s": motion_features.get("duration_s", 0.0),
        "nod_likelihood_score": motion_features.get("nod_likelihood_score", 0),
        "ratio_vs_5s": motion_features.get("ratio_vs_5s"),
        "ratio_vs_30s": motion_features.get("ratio_vs_30s"),
    }


def build_backchannel_graph(
    client: OpenAI,
    model: str,
    items: List[BackchannelItem],
) -> StateGraph:
    def prepare(state: AgentState) -> Dict[str, object]:
        """候補をフィルタリングする"""
        allowlist = state.get("directory_allowlist", [])
        raw_avoid = state.get("avoid_ids", [])
        avoid_ids = {str(x) for x in raw_avoid} if isinstance(raw_avoid, list) else set()
        filtered = items
        if isinstance(allowlist, list) and allowlist:
            allow = {str(x) for x in allowlist if isinstance(x, str)}
            filtered = [it for it in filtered if it.directory in allow]
        if avoid_ids:
            filtered = [it for it in filtered if it.id not in avoid_ids]
        if not filtered:
            filtered = items
        candidates = _build_candidates(filtered)
        return {"candidates": candidates}

    def choose(state: AgentState) -> Dict[str, object]:
        """相槌の種類を選ぶ（NONE を含む）"""
        candidates = state["candidates"]
        candidate_ids = [c["id"] for c in candidates]
        schema = _build_choice_schema(candidate_ids)

        timing = state.get("timing", {})
        if not isinstance(timing, dict):
            timing = {}
        is_boundary = bool(timing.get("is_boundary", False))
        speaker_speaking = bool(timing.get("speaker_speaking", False))
        speaker_pause_like_boundary = bool(timing.get("speaker_pause_like_boundary", False))
        transcript_latest_age_s = timing.get("transcript_latest_age_s")
        transcript_latest_age_s_text = (
            f"{float(transcript_latest_age_s):.1f}" if isinstance(transcript_latest_age_s, (int, float)) else "-"
        )
        seconds_since_signal = timing.get("seconds_since_signal")
        seconds_since_signal_text = (
            f"{float(seconds_since_signal):.2f}" if isinstance(seconds_since_signal, (int, float)) else "-"
        )
        transcript_context = str(state.get("transcript_context", "") or "").strip()
        if not transcript_context:
            transcript_context = "文字起こしはまだありません"

        system_text = (
            "あなたは相槌の選択役です。\n"
            "返すのに良いタイミングなら、候補から1つ選びます。\n"
            "割り込みになりそう、内容が合わない、迷うときは NONE を選びます。\n\n"
            "【目安】\n"
            "- 動きの向き: nod は肯定、shake は否定\n"
            "- 話し手が話している(speaker_speaking=true)間は、割り込みになりやすいです\n"
            "- 区切りっぽい(speaker_pause_like_boundary=true / is_boundary=true)なら返しやすいです\n"
            "- 文字起こしが古い(transcript_latest_age_s が大きい)ときは、無理に合わせないでください\n"
            "- reason は20文字以内で書いてください"
        )

        motion_summary = _extract_motion_summary(state["imu"])
        gesture_hint = str(motion_summary.get("gesture_hint", "other"))

        prompt = (
            "【状況】\n"
            f"speaker_speaking: {speaker_speaking}\n"
            f"speaker_pause_like_boundary: {speaker_pause_like_boundary}\n"
            f"is_boundary: {is_boundary}\n"
            f"transcript_latest_age_s: {transcript_latest_age_s_text}\n"
            f"seconds_since_signal: {seconds_since_signal_text}\n\n"
            "【直近の文脈】\n"
            f"{transcript_context}\n\n"
            "【候補】\n"
            f"{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
            "【動きの向き】\n"
            f"{gesture_hint}\n\n"
            "【現在の発話】\n"
            f"{state['utterance']}\n\n"
            "どの相槌を選びますか？"
        )

        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "backchannel_choice",
                    "strict": True,
                    "schema": schema,
                }
            },
        )

        raw_text = response.output_text.strip()
        selection: Dict[str, object] = {}
        errors: List[str] = list(state.get("errors", []))
        try:
            selection = json.loads(raw_text)
        except json.JSONDecodeError:
            errors.append("choose_json_parse_failed")
        return {"selection": selection, "errors": errors}

    def resolve(state: AgentState) -> Dict[str, object]:
        """選ばれたIDを検証する"""
        selection = state.get("selection", {})
        if not isinstance(selection, dict):
            return {"selected_id": "NONE"}

        selected_id = str(selection.get("id", ""))
        if selected_id == "NONE":
            return {"selected_id": selected_id}

        candidate_ids = {c["id"] for c in state.get("candidates", [])}
        if selected_id not in candidate_ids:
            selected_id = _fallback_id(items)

        return {"selected_id": selected_id, "selection": selection}

    graph = StateGraph(AgentState)
    graph.add_node("prepare", prepare)
    graph.add_node("choose", choose)
    graph.add_node("resolve", resolve)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "choose")
    graph.add_edge("choose", "resolve")
    graph.add_edge("resolve", END)

    return graph
