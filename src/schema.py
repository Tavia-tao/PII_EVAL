"""版本化数据契约。所有偏移均相对 coordinate_text / gold.text，Unicode 码点。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ENTITY_TYPES = {
    "ID_CARD",
    "BANK_CARD",
    "PASSPORT",
    "PHONE",
    "MEDICAL_ID",
    "EMAIL",
    "NAME",
    "ADDRESS",
    "BIRTH_DATE",
    "DATE",
    "PLATE",
    "ORG",
}
STAGES = ("S1_parse", "S2_ocr", "S3_layout", "S4_detect", "S5_mask", "S6_export")
HIGH_RISK = ("ID_CARD", "BANK_CARD", "PASSPORT", "PHONE", "MEDICAL_ID")
WEIGHTS = {
    "ID_CARD": 5.0,
    "BANK_CARD": 5.0,
    "PASSPORT": 5.0,
    "PHONE": 3.0,
    "MEDICAL_ID": 3.0,
    "EMAIL": 3.0,
    "NAME": 2.0,
    "ADDRESS": 2.0,
    "BIRTH_DATE": 1.5,
    "PLATE": 1.5,
    "DATE": 0.5,
    "ORG": 0.5,
}


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Entity(Contract):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    type: str
    text: str | None = None
    risk: Literal["high", "normal"] | None = None
    conf: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid(self):
        if self.end <= self.start or self.type not in ENTITY_TYPES:
            raise ValueError("无效边界或实体类型")
        return self


class Negative(Contract):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    type: str = "NOT_PII"
    text: str | None = None
    reason: str
    critical: bool = False


class Block(Contract):
    block_id: str
    text: str
    page: int = Field(default=1, ge=1)
    order: int = 0
    type: str = "paragraph"
    bbox: tuple[float, float, float, float] | None = None


class Input(Contract):
    text: str = ""
    file_path: str | None = None
    ocr_text: str | None = None
    text_blocks: list[Block] = Field(default_factory=list)
    ocr_mode: Literal["none", "replay", "tesseract", "http"] = "none"


class Gold(Contract):
    text: str
    spans: list[Entity] = Field(default_factory=list)
    non_pii_candidates: list[Negative] = Field(default_factory=list)
    non_pii_hard_negatives: list[Negative] = Field(default_factory=list)
    ocr_text: str | None = None
    layout_blocks: list[Block] = Field(default_factory=list)


class AnnotationMeta(Contract):
    rule_version: str
    annotator_ids: list[str] = Field(default_factory=list)
    kappa: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    annotated_at: str | None = None
    quality_source: Literal["synthetic", "human"] = "synthetic"
    boundary: Literal["clear", "adhesive", "nested"] = "clear"


class Sample(Contract):
    sample_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    doc_id: str
    parent_doc_id: str
    split: Literal["train", "dev", "holdout", "gate"]
    track: Literal["standard_text", "complex_file"]
    source: Literal["customer_real", "public", "synthetic"]
    doc_type: str = "text"
    layout: str
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    lang: str = "zh"
    page_count: int = Field(default=1, ge=1)
    input: Input
    gold: Gold
    meta: AnnotationMeta

    @model_validator(mode="after")
    def annotation_integrity(self):
        if self.track == "complex_file" and not self.input.file_path:
            raise ValueError("复杂文件必须提供 file_path")
        text = self.gold.text
        seen = set()
        ordered = sorted(self.gold.spans, key=lambda s: (s.start, -s.end))
        stack = []
        for s in ordered:
            key = (s.start, s.end, s.type)
            if key in seen:
                raise ValueError("金标重复")
            seen.add(key)
            if s.end > len(text) or (s.text is not None and text[s.start : s.end] != s.text):
                raise ValueError("金标 text 与文本边界不一致")
            while stack and stack[-1].end <= s.start:
                stack.pop()
            if stack and s.end > stack[-1].end:
                raise ValueError("不允许交叉重叠金标")
            stack.append(s)
        negatives = self.gold.non_pii_candidates or self.gold.non_pii_hard_negatives
        for i, s in enumerate(sorted(negatives, key=lambda n: n.start)):
            if not 0 <= s.start < s.end <= len(text):
                raise ValueError("非PII候选边界错误")
            if s.text is not None and text[s.start : s.end] != s.text:
                raise ValueError("非PII text 与文本不一致")
            if any(max(s.start, g.start) < min(s.end, g.end) for g in self.gold.spans):
                raise ValueError("非PII候选与金标重叠")
            previous = sorted(negatives, key=lambda n: n.start)
            if i and previous[i - 1].end > s.start:
                raise ValueError("非PII候选不可重叠")
        return self


class Stage(Contract):
    ok: bool = True
    elapsed_ms: float = Field(default=0, ge=0, allow_inf_nan=False)
    metrics: dict[str, float | int | bool | None] = Field(default_factory=dict)
    artifact: str | None = None
    error: str | None = None


class Usage(Contract):
    input: int = Field(default=0, ge=0)
    output: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0, allow_inf_nan=False)
    measured: bool = False


class Prediction(Contract):
    sample_id: str
    model_version: str = "demo-student-v1"
    prompt_version: str = "pii-span-v1"
    spans: list[Entity] = Field(default_factory=list)
    coordinate_text: str = ""
    stages: dict[str, Stage] = Field(default_factory=dict)
    latency_ms: float = Field(default=0, ge=0, allow_inf_nan=False)
    usage: Usage = Field(default_factory=Usage)
    route: str = "student"
    route_reason: str = "default"
    needs_review: bool = False
    error: str | None = None
    trace_id: str | None = None
    cached: bool = False
