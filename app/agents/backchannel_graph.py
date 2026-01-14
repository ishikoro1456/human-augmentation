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
    directory_allowlist: List[str]
    avoid_ids: List[str]
    candidates: List[Dict[str, object]]
    selection: Dict[str, object]
    selected_id: str
    errors: List[str]


def _build_candidates(
    items: List[BackchannelItem],
) -> List[Dict[str, object]]:
    candidates = items
    return [
        {
            "id": item.id,
            "directory": item.directory,
            "strength": item.strength,
            "nod": item.nod,
            "text": item.text,
        }
        for item in candidates
    ]


def _build_schema(candidate_ids: List[str]) -> Dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string", "enum": candidate_ids + ["NONE"]},
            "reason": {"type": "string"},
        },
        "required": ["id", "reason"],
        "additionalProperties": False,
    }


def _fallback_id(items: List[BackchannelItem]) -> str:
    for item in items:
        if item.strength == 3 and item.nod == 3:
            return item.id
    return items[0].id


def build_backchannel_graph(
    client: OpenAI,
    model: str,
    items: List[BackchannelItem],
) -> StateGraph:
    def prepare(state: AgentState) -> Dict[str, object]:
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
        candidates = state["candidates"]
        candidate_ids = [c["id"] for c in candidates]
        schema = _build_schema(candidate_ids)

        system_text = (
            "あなたは相槌を選ぶ役です。\n"
            "聞き手(人間)のIMUサインが出たときに、短い相槌を返します。\n"
            "相槌は基本は『聞いている』合図で、内容の賛否を断定しないでください。\n"
            "同じ相槌を続けて繰り返すのは避けてください。\n"
            "IMUサインの向きと矛盾する相槌は選ばないでください。\n"
            "候補の中から1つだけ選んでください。\n"
            "出力はJSONだけにしてください。\n"
            "reason は短い一文にしてください。推測で話を広げないでください。"
        )

        prompt = (
            "候補:\n"
            f"{json.dumps(candidates, ensure_ascii=False)}\n\n"
            "いま話し手が話している(可能性がある)文:\n"
            f"{state['utterance']}\n\n"
            "その文の時刻(秒):\n"
            f"{state['utterance_t_sec']}\n\n"
            "ここまで読み上げた文字起こし:\n"
            f"{state['transcript_context']}\n\n"
            "IMU(生データ+要約):\n"
            f"{json.dumps(state['imu'], ensure_ascii=False)}\n\n"
            "補足: raw_samples の t_rel_s は、今から見た相対秒です。負の値ほど過去です。\n\n"
            "補足: calibration がある場合は、個人差の基準です。normalized_activity があれば、基準に対する相対値です。\n\n"
            "補足: human_signal.present が false の場合は、聞き手(人間)が相槌のサインを出していません。NONE を強く優先してください。\n\n"
            "補足: human_signal.gesture_hint は聞き手の動きの向きの推定です。"
            "shake は違和感・疑問の方向、nod は理解の方向です。\n"
            "注意: nod は『同意』ではなく『理解・追従』のことが多いです。\n\n"
            "補足: gesture_calibration があれば、弱い/強い頷き・首振りの例です。"
            "gesture_intensity があれば、今の動きが弱/強のどちら寄りかの推定です。\n\n"
            "音声の状態:\n"
            f"{json.dumps(state['audio_state'], ensure_ascii=False)}\n\n"
            "直近の相槌:\n"
            f"{json.dumps(state['recent_backchannel'], ensure_ascii=False)}\n\n"
            "候補から1つ選び、idを返してください。\n"
            "いまは聞き手が相槌サインを出した直後です。\n"
            "話し手の音声と重なる可能性があるので、短く控えめにしてください。\n"
            "直近で同じ文言が続いているなら、できるだけ別の文言か NONE を選んでください。\n"
            "時間の流れを意識して、直近の内容に合わせてください。\n"
            "会話を遮って聞こえにくくしそうなら、無理に返さないでください。\n"
            "reason には IMU と文字起こしの両方に触れてください。\n"
            "返事をしない方が自然なら id は NONE を選んでください。\n"
            "迷ったら無理に返さず、NONE を選ぶ方を優先してください。"
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
        errors: List[str] = []
        try:
            selection = json.loads(raw_text)
        except json.JSONDecodeError:
            errors.append("json_parse_failed")
        return {"selection": selection, "errors": errors}

    def resolve(state: AgentState) -> Dict[str, object]:
        selection = state.get("selection", {})
        selected_id = str(selection.get("id", ""))
        if selected_id == "NONE":
            return {"selected_id": selected_id}
        candidate_ids = {c["id"] for c in state["candidates"]}
        if selected_id not in candidate_ids:
            selected_id = _fallback_id(items)
        return {"selected_id": selected_id}

    graph = StateGraph(AgentState)
    graph.add_node("prepare", prepare)
    graph.add_node("choose", choose)
    graph.add_node("resolve", resolve)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "choose")
    graph.add_edge("choose", "resolve")
    graph.add_edge("resolve", END)
    return graph
