from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


# 意图识别返回的json
class IntentDecision(StrEnum):
    ROUTE = "ROUTE"
    CLARIFY = "CLARIFY"
    REJECT = "REJECT"

# 置信水平
class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EMERGENCY = "EMERGENCY"


class IntentEntity(BaseModel):
    type: str
    raw_text: str
    normalized_text: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IntentCandidate(BaseModel):
    intent: str
    score: float = Field(ge=0.0)
    matched_text: str | None = None
    matched_fields: list[str] = Field(default_factory=list)
    route_target: str | None = None
    risk_level: RiskLevel = RiskLevel.LOW
    metadata: dict[str, Any] = Field(default_factory=dict)


class IntentResult(BaseModel):
    schema_version: str = "1.0"
    engine: str = "ES_TOPK_SIMILARITY"
    engine_version: str = "intent-es-v1.0"

    detected_intent: str | None = None
    primary_intent: str
    secondary_intents: list[str] = Field(default_factory=list)

    confidence: float = Field(ge=0.0, le=1.0)
    decision: IntentDecision
    route_target: str | None = None
    risk_level: RiskLevel = RiskLevel.LOW

    missing_slots: list[str] = Field(default_factory=list)
    entities: list[IntentEntity] = Field(default_factory=list)
    topk_candidates: list[IntentCandidate] = Field(default_factory=list)

    safety_flags: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)
