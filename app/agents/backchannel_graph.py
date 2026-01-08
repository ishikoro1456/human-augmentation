import json
from typing import Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from app.core.types import BackchannelItem


class AgentState(TypedDict):
    utterance: str
    motion: Dict[str, object]
    motion_text: str
    transcript_context: str
    candidates: List[Dict[str, object]]
    selection: Dict[str, object]
    selected_id: str
    errors: List[str]


def _build_candidates(
    items: List[BackchannelItem],
    motion: Dict[str, object],
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
            "id": {"type": "string", "enum": candidate_ids},
            "reason": {"type": "string"},
        },
        "required": ["id", "reason"],
        "additionalProperties": False,
    }


def _fallback_id(items: List[BackchannelItem], motion: Dict[str, object]) -> str:
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
        candidates = _build_candidates(items, state["motion"])
        return {"candidates": candidates}

    def choose(state: AgentState) -> Dict[str, object]:
        candidates = state["candidates"]
        candidate_ids = [c["id"] for c in candidates]
        schema = _build_schema(candidate_ids)

        system_text = (
            "あなたは相槌を選ぶ役です。\n"
            "候補の中から1つだけ選んでください。\n"
            "出力はJSONだけにしてください。\n"
            "reason は短い一文にしてください。"
        )

        prompt = (
            "候補:\n"
            f"{json.dumps(candidates, ensure_ascii=False)}\n\n"
            "話し手の発話:\n"
            f"{state['utterance']}\n\n"
            "ここまで読み上げた文字起こし:\n"
            f"{state['transcript_context']}\n\n"
            f"{state['motion_text']}\n\n"
            "候補から1つ選び、idを返してください。\n"
            "時間の流れを意識して、直近の内容に合わせてください。\n"
            "IMUは短い粒度の動きなので、発話の区切りや強調に合わせて選んでください。\n"
            "reason には IMU と文字起こしの両方に触れてください。"
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
        candidate_ids = {c["id"] for c in state["candidates"]}
        if selected_id not in candidate_ids:
            selected_id = _fallback_id(items, state["motion"])
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
