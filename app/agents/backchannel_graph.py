import json
from typing import Dict, List, Literal, TypedDict

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
    directory_allowlist: List[str]
    avoid_ids: List[str]
    # 第一段（decide）の出力
    decision: Dict[str, object]
    # 第二段（choose）の出力
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


def _build_decide_schema() -> Dict[str, object]:
    """第一段：返すかどうかを判断するためのスキーマ（シンプル版）"""
    return {
        "type": "object",
        "properties": {
            "should_respond": {
                "type": "boolean",
                "description": "相槌を返すべきかどうか",
            },
            "reason": {
                "type": "string",
                "description": "判断理由（20文字以内）",
            },
        },
        "required": ["should_respond", "reason"],
        "additionalProperties": False,
    }


def _build_choose_schema(candidate_ids: List[str]) -> Dict[str, object]:
    """第二段：相槌を選ぶためのスキーマ"""
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string", "enum": candidate_ids},
            "reason_short": {
                "type": "string",
                "description": "選んだ理由を一言で（10文字以内）",
            },
        },
        "required": ["id", "reason_short"],
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

    def decide(state: AgentState) -> Dict[str, object]:
        """第一段：返すかどうかを判断する（シンプル版）"""
        motion_summary = _extract_motion_summary(state["imu"])
        recent = state.get("recent_backchannel", {})
        if not isinstance(recent, dict):
            recent = {}

        # nod_likelihood_score が高ければ基本的に返す
        nod_score = motion_summary.get("nod_likelihood_score", 0)
        has_oscillation = motion_summary.get("has_oscillation", False)

        system_text = (
            "あなたは「相槌を返すかどうか」を判断する役です。\n"
            "聞き手のIMUセンサーが動きを検出しました。\n\n"
            "【判断基準】\n"
            "- nod_likelihood_score が 4 以上なら、頷きの可能性が高い → 返す\n"
            "- has_oscillation が true なら、往復運動がある → 返す傾向\n"
            "- 直近で相槌を出しすぎていたら → 返さない\n"
            "- gesture_hint が nod なら理解系、shake なら疑問系の反応\n\n"
            "【重要】\n"
            "- センサーが検出した時点で、何らかの反応があった可能性が高い\n"
            "- 明確に返すべきでない理由がなければ、返してください\n"
            "- reason は20文字以内で簡潔に"
        )

        prompt = (
            f"nod_likelihood_score: {nod_score}\n"
            f"has_oscillation: {has_oscillation}\n"
            f"gesture_hint: {motion_summary.get('gesture_hint', 'other')}\n"
            f"直近の相槌: {recent.get('seconds_ago', 'なし')}秒前\n"
            f"現在の発話: {state['utterance']}\n\n"
            "相槌を返すべきですか？"
        )

        schema = _build_decide_schema()
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "backchannel_decision",
                    "strict": True,
                    "schema": schema,
                }
            },
        )

        raw_text = response.output_text.strip()
        decision: Dict[str, object] = {}
        errors: List[str] = list(state.get("errors", []))
        try:
            decision = json.loads(raw_text)
        except json.JSONDecodeError:
            errors.append("decide_json_parse_failed")
            decision = {"should_respond": False, "reason": "解析失敗"}

        return {"decision": decision, "errors": errors}

    def route_after_decide(state: AgentState) -> Literal["choose", "skip"]:
        """decideの結果に応じて分岐する"""
        decision = state.get("decision", {})
        if not isinstance(decision, dict):
            return "skip"
        should_respond = decision.get("should_respond", False)
        if should_respond:
            return "choose"
        return "skip"

    def choose(state: AgentState) -> Dict[str, object]:
        """第二段：相槌の種類を選ぶ（should_respond: true の場合のみ）"""
        candidates = state["candidates"]
        candidate_ids = [c["id"] for c in candidates]
        schema = _build_choose_schema(candidate_ids)

        decision = state.get("decision", {})
        checks = decision.get("checks", {}) if isinstance(decision, dict) else {}

        system_text = (
            "あなたは相槌の種類を選ぶ役です。\n"
            "すでに「相槌を返す」と決まっています。\n"
            "候補の中から、文脈に最も合う相槌を1つ選んでください。\n\n"
            "【選び方】\n"
            "- 動きの向き（nod=理解、shake=疑問）に合わせる\n"
            "- 短く控えめなものを優先する\n"
            "- 直近で使った相槌と被らないようにする\n"
            "- reason_short は10文字以内で書いてください"
        )

        motion_summary = _extract_motion_summary(state["imu"])
        recent = state.get("recent_backchannel", {})

        prompt = (
            "【候補】\n"
            f"{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
            "【動きの向き】\n"
            f"{motion_summary.get('gesture_hint', 'other')}\n\n"
            "【動きの強さ（1-5）】\n"
            f"{motion_summary.get('intensity_level_1to5', 0)}\n\n"
            "【直近の相槌】\n"
            f"{json.dumps(recent, ensure_ascii=False)}\n\n"
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

    def skip(state: AgentState) -> Dict[str, object]:
        """相槌を返さない場合"""
        decision = state.get("decision", {})
        reason = ""
        if isinstance(decision, dict):
            reason = str(decision.get("reason", ""))
        return {
            "selection": {"id": "NONE", "reason": reason},
            "selected_id": "NONE",
        }

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

        # decisionの情報をselectionに追加
        decision = state.get("decision", {})
        if isinstance(decision, dict):
            selection["decision_reason"] = decision.get("reason", "")

        return {"selected_id": selected_id, "selection": selection}

    graph = StateGraph(AgentState)
    graph.add_node("prepare", prepare)
    graph.add_node("decide", decide)
    graph.add_node("choose", choose)
    graph.add_node("skip", skip)
    graph.add_node("resolve", resolve)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "decide")
    graph.add_conditional_edges("decide", route_after_decide, {"choose": "choose", "skip": "skip"})
    graph.add_edge("choose", "resolve")
    graph.add_edge("skip", END)
    graph.add_edge("resolve", END)

    return graph
