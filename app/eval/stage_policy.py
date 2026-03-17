from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from openai import OpenAI
else:
    OpenAI = Any


@dataclass(frozen=True)
class StageConfig:
    index: int
    key: str
    name: str
    description: str
    specific_metrics: tuple[str, str, str]


@dataclass(frozen=True)
class StageResponse:
    text: str
    stage: str
    intensity_1to5: int
    generation_mode: str
    constraints_ok: bool
    reason: str


STAGE_CONFIGS: list[StageConfig] = [
    StageConfig(
        index=0,
        key="stage1_yesno",
        name="Stage1 YES/NO",
        description="固定語彙で方向のみを返す",
        specific_metrics=(
            "yesno_direction_correctness",
            "unnecessary_response_rate",
            "missed_response_rate",
        ),
    ),
    StageConfig(
        index=1,
        key="stage2_yesno_intensity",
        name="Stage2 YES/NO+5強度",
        description="固定語彙で方向と強度を返す",
        specific_metrics=(
            "intensity_match",
            "intensity_stability",
            "intensity_controllability",
        ),
    ),
    StageConfig(
        index=2,
        key="stage3_micro_phrase",
        name="Stage3 超短フレーズ",
        description="AI生成で文にならない短い相槌を返す",
        specific_metrics=(
            "micro_phrase_naturalness",
            "brevity_appropriateness",
            "intent_transfer_clarity",
        ),
    ),
    StageConfig(
        index=3,
        key="stage4_context_sentence",
        name="Stage4 文脈あり1文",
        description="AI生成で文脈のある1文を返す",
        specific_metrics=(
            "contextual_fit",
            "sentence_naturalness",
            "information_amount_appropriateness",
        ),
    ),
]


STAGE_COMMON_METRICS: tuple[str, str, str, str] = (
    "intent_match",
    "timing_naturalness",
    "response_acceptability",
    "usability",
)


ISSUE_TAG_OPTIONS: tuple[tuple[str, str], ...] = (
    ("wrong_direction", "肯定/否定が逆"),
    ("wrong_intensity", "強度が不適切"),
    ("bad_timing", "タイミングが不自然"),
    ("too_short", "短すぎる"),
    ("too_long", "長すぎる"),
    ("hallucinated_content", "文脈と無関係な内容"),
    ("repetitive", "似た応答が続く"),
    ("should_be_silent", "返さない方が良い"),
)


_STAGE1_PHRASE = {
    "yes": "はい",
    "no": "いいえ",
}


# YES 5段階 / NO 5段階
_STAGE2_PHRASE = {
    "yes": {
        1: "はい",
        2: "うん",
        3: "そうですね",
        4: "その通りです",
        5: "まさにその通りです",
    },
    "no": {
        1: "いいえ",
        2: "ううん",
        3: "ちがうかな",
        4: "それは違うと思います",
        5: "それは違うと思いますね",
    },
}


def get_stage_config(stage_index: int) -> StageConfig:
    idx = max(0, min(int(stage_index), len(STAGE_CONFIGS) - 1))
    return STAGE_CONFIGS[idx]


def _gesture_to_polarity(gesture_hint: str) -> str:
    return "yes" if str(gesture_hint) == "nod" else "no"


def estimate_intensity_level(*, nod_score: int, signal_confidence: float | None = None) -> int:
    score = int(max(0, min(6, int(nod_score))))
    if score <= 1:
        base = 1
    elif score == 2:
        base = 2
    elif score == 3:
        base = 3
    elif score == 4:
        base = 4
    else:
        base = 5

    if signal_confidence is None:
        return base

    conf = max(0.0, min(1.0, float(signal_confidence)))
    if conf < 0.2:
        return max(1, base - 1)
    if conf > 0.9:
        return min(5, base + 1)
    return base


def _extract_text(resp: Any) -> str:
    if isinstance(resp, str):
        return resp.strip()
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    text2 = getattr(resp, "text", None)
    if isinstance(text2, str) and text2.strip():
        return text2.strip()
    if isinstance(resp, dict):
        t = resp.get("text") or resp.get("output_text")
        if isinstance(t, str):
            return t.strip()
    return str(resp).strip()


def _is_stage3_valid(text: str) -> bool:
    t = str(text).strip()
    if not t:
        return False
    if len(t) < 1 or len(t) > 6:
        return False
    if "。" in t:
        return False
    if "\n" in t:
        return False
    return True


def _is_stage4_valid(text: str) -> bool:
    t = str(text).strip()
    if len(t) < 12 or len(t) > 45:
        return False
    if "\n" in t:
        return False
    # 句点は0 or 1
    if t.count("。") > 1:
        return False
    # 1文のみ
    chunks = [x for x in re.split(r"[。！？!?]", t) if x.strip()]
    return len(chunks) == 1


def _stage3_fallback(gesture_hint: str) -> str:
    return "はい" if str(gesture_hint) == "nod" else "いいえ"


def _stage4_fallback(gesture_hint: str) -> str:
    return "はい、そうですね。" if str(gesture_hint) == "nod" else "いいえ、そうではないと思います。"


def _build_stage_prompt(
    *,
    stage: StageConfig,
    gesture_hint: str,
    intensity_1to5: int,
    utterance: str,
    context: str,
    imu_features: Dict[str, object] | None,
    second_try: bool,
) -> str:
    hint_jp = "肯定寄り" if str(gesture_hint) == "nod" else "否定寄り"
    retry_note = (
        "前回は制約違反でした。今回は制約を必ず守ってください。"
        if second_try
        else ""
    )
    imu_text = ""
    if isinstance(imu_features, dict) and imu_features:
        imu_text = str(imu_features)

    if stage.index == 2:
        return (
            "あなたは相槌生成器です。\n"
            "出力は相槌テキストのみ。説明は書かないでください。\n"
            "制約:\n"
            "- 文ではなく短いフレーズ\n"
            "- 1〜6文字\n"
            "- 句点(。)を使わない\n"
            f"- 方向: {hint_jp}\n"
            f"- 強度: {intensity_1to5}/5\n"
            f"現在の発話: {utterance}\n"
            f"直近文脈: {context}\n"
            f"IMU特徴: {imu_text}\n"
            f"{retry_note}"
        )
    return (
        "あなたは相槌生成器です。\n"
        "出力は相槌テキストのみ。説明は書かないでください。\n"
        "制約:\n"
        "- 文脈に沿う\n"
        "- 1文のみ\n"
        "- 12〜45文字\n"
        "- 句点(。)は0個か1個\n"
        f"- 方向: {hint_jp}\n"
        f"- 強度: {intensity_1to5}/5\n"
        f"現在の発話: {utterance}\n"
        f"直近文脈: {context}\n"
        f"IMU特徴: {imu_text}\n"
        f"{retry_note}"
    )


def _generate_text_once(
    *,
    client: OpenAI,
    model: str,
    prompt: str,
) -> str:
    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "日本語で自然な相槌だけを返す。余計な説明は禁止。",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return _extract_text(resp)


def generate_stage_response(
    *,
    client: OpenAI,
    model: str,
    stage: StageConfig,
    gesture_hint: str,
    utterance: str,
    context: str,
    intensity_1to5: int,
    imu_features: Dict[str, object] | None = None,
) -> StageResponse:
    polarity = _gesture_to_polarity(gesture_hint)
    level = max(1, min(5, int(intensity_1to5)))

    if stage.index == 0:
        return StageResponse(
            text=_STAGE1_PHRASE[polarity],
            stage=stage.key,
            intensity_1to5=level,
            generation_mode="fixed",
            constraints_ok=True,
            reason=f"stage1_fixed_{polarity}",
        )

    if stage.index == 1:
        text = _STAGE2_PHRASE[polarity][level]
        return StageResponse(
            text=text,
            stage=stage.key,
            intensity_1to5=level,
            generation_mode="fixed",
            constraints_ok=True,
            reason=f"stage2_fixed_{polarity}_level{level}",
        )

    # Stage3/4: 生成
    text = ""
    ok = False
    for attempt in range(2):
        prompt = _build_stage_prompt(
            stage=stage,
            gesture_hint=gesture_hint,
            intensity_1to5=level,
            utterance=utterance,
            context=context,
            imu_features=imu_features,
            second_try=bool(attempt == 1),
        )
        try:
            text = _generate_text_once(client=client, model=model, prompt=prompt)
        except Exception:
            text = ""

        if stage.index == 2:
            ok = _is_stage3_valid(text)
        else:
            ok = _is_stage4_valid(text)
        if ok:
            break

    if not ok:
        if stage.index == 2:
            text = _stage3_fallback(gesture_hint)
            ok = _is_stage3_valid(text)
        else:
            text = _stage4_fallback(gesture_hint)
            ok = _is_stage4_valid(text)

    return StageResponse(
        text=text,
        stage=stage.key,
        intensity_1to5=level,
        generation_mode="ai_generated",
        constraints_ok=bool(ok),
        reason="stage_generated",
    )


def stage_names() -> List[str]:
    return [s.name for s in STAGE_CONFIGS]
