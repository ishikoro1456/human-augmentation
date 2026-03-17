from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SttSegment:
    segment_id: int
    text: str
    ts: float


@dataclass
class AgentDecision:
    call_id: str
    experiment_id: str
    index: int
    utterance: str
    transcript_context: str
    gesture_hint: str  # "nod", "shake", "other"
    nod_likelihood_score: int  # 0-6
    directory_allowlist: list
    is_boundary: bool
    speaker_speaking: bool
    speaker_silence_ms: int
    selected_id: str  # catalog id or "NONE"
    selected_text: str
    strength: int  # 0, 1, 3, or 5
    reason: str
    latency_ms: int
    ts: float
    stage_index: int = -1
    stage_name: str = ""
    intensity_1to5: int = 0
    generated_text: str = ""
    generation_mode: str = ""
    signal_confidence: float = 0.0
    imu_features: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionSummary:
    experiment_id: str
    mode: str
    model: str
    start_ts: float
    decision_count: int


@dataclass
class SessionDetail:
    experiment_id: str
    mode: str
    model: str
    start_ts: float
    decisions: list  # list[AgentDecision]
    stt_segments: list  # list[SttSegment]


@dataclass
class TimelineItem:
    ts: float
    kind: str  # "stt" or "decision"
    stt: Optional[SttSegment] = None
    decision: Optional[AgentDecision] = None


@dataclass
class Evaluation:
    experiment_id: str
    call_id: str
    evaluator_id: str
    appropriateness: int  # 1-7
    would_have_sent: str = ""  # catalog id, "NONE", or ""
    issues: list = field(default_factory=list)
    comment: str = ""
    time_spent_ms: int = 0
    created_at: str = ""
    id: Optional[int] = None


@dataclass
class StageEvaluation:
    session_id: str
    stage_index: int
    evaluator_id: str
    common_q1: int
    common_q2: int
    common_q3: int
    common_q4: int
    specific_q1: int
    specific_q2: int
    specific_q3: int
    overall_comment: str = ""
    created_at: str = ""
    id: Optional[int] = None


@dataclass
class ResponseAnnotation:
    session_id: str
    stage_index: int
    call_id: str
    evaluator_id: str
    issue_tags: list[str] = field(default_factory=list)
    comment: str = ""
    created_at: str = ""
    id: Optional[int] = None
