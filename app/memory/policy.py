import re
from dataclasses import dataclass
from typing import Any

from app.agent.state import AgentState


WRITE_TRIGGERS = [
    "记住",
    "以后",
    "下次",
    "我偏好",
    "我喜欢",
    "我的习惯",
    "请保存",
    "帮我保存",
    "我叫",
    "叫我",
    "我的名字是",
]

SENSITIVE_HEALTH_KEYWORDS = [
    "胸痛",
    "呼吸困难",
    "昏迷",
    "大出血",
    "过敏",
    "用药",
    "吃药",
    "停药",
    "换药",
    "药量",
    "剂量",
    "疾病",
    "诊断",
    "病史",
    "症状",
    "舌象图片",
    "舌头照片",
]

@dataclass(frozen=True)
class MemoryCandidate:
    memory_type: str
    memory_key: str
    text: str
    summary: str
    value: str
    source: str = "explicit_user_input"


@dataclass(frozen=True)
class MemoryPolicyDecision:
    can_read: bool
    can_write: bool
    memory_candidate: MemoryCandidate | None
    skip_reason: str | None = None

    @property
    def write_candidate(self) -> str | None:
        if self.memory_candidate is None:
            return None
        return self.memory_candidate.text


def extract_user_text(state: AgentState) -> str:
    message: dict[str, Any] = state.get("message", {})
    content = message.get("content")

    if isinstance(content, str):
        return content.strip()

    return ""


def decide_memory_policy(state: AgentState) -> MemoryPolicyDecision:
    permissions = _memory_permissions(state)
    text = extract_user_text(state)

    can_read = bool(permissions.get("can_read", True))
    consent_write = bool(permissions.get("can_write", False))

    if not text:
        return MemoryPolicyDecision(
            can_read=can_read,
            can_write=False,
            memory_candidate=None,
            skip_reason="empty_user_text",
        )

    if _is_high_risk_state(state):
        return MemoryPolicyDecision(
            can_read=can_read,
            can_write=False,
            memory_candidate=None,
            skip_reason="high_risk_context_blocked",
        )

    if not _has_write_trigger(text):
        return MemoryPolicyDecision(
            can_read=can_read,
            can_write=False,
            memory_candidate=None,
            skip_reason="no_explicit_memory_request",
        )

    if not consent_write:
        return MemoryPolicyDecision(
            can_read=can_read,
            can_write=False,
            memory_candidate=None,
            skip_reason="missing_long_term_memory_consent",
        )

    if _is_sensitive_health_text(text):
        return MemoryPolicyDecision(
            can_read=can_read,
            can_write=False,
            memory_candidate=None,
            skip_reason="sensitive_health_memory_blocked",
        )

    candidate = build_memory_candidate(text)
    if candidate is None:
        return MemoryPolicyDecision(
            can_read=can_read,
            can_write=False,
            memory_candidate=None,
            skip_reason="unsupported_memory_type",
        )

    return MemoryPolicyDecision(
        can_read=can_read,
        can_write=True,
        memory_candidate=candidate,
        skip_reason=None,
    )


def build_memory_candidate(text: str) -> MemoryCandidate | None:
    normalized = text.strip()

    name_match = re.search(
        r"(?:我叫|叫我|我的名字是)\s*([A-Za-z0-9_\u4e00-\u9fa5]{1,20})",
        normalized,
    )
    if name_match:
        name = name_match.group(1)
        return MemoryCandidate(
            memory_type="user_identity",
            memory_key="user_identity:preferred_name",
            text=normalized,
            summary=f"用户希望被称为{name}。",
            value=name,
        )

    detail_value = _answer_detail_value(normalized)
    if detail_value:
        return MemoryCandidate(
            memory_type="communication_preference",
            memory_key="communication:answer_detail",
            text=normalized,
            summary=f"用户偏好回答{detail_value}。",
            value=detail_value,
        )

    if any(word in normalized for word in ["先看结论", "先给结论", "先说结论"]):
        return MemoryCandidate(
            memory_type="communication_preference",
            memory_key="communication:answer_order",
            text=normalized,
            summary="用户偏好先看结论，再看解释。",
            value="先结论后解释",
        )

    preferred_feature = _preferred_feature(normalized)
    if preferred_feature:
        return MemoryCandidate(
            memory_type="product_preference",
            memory_key="product:preferred_feature",
            text=normalized,
            summary=f"用户更关注{preferred_feature}。",
            value=preferred_feature,
        )

    if any(word in normalized for word in ["偏好", "喜欢", "习惯"]):
        return MemoryCandidate(
            memory_type="user_preference",
            memory_key="user_preference:general",
            text=normalized,
            summary=f"用户表达了一个一般偏好：{normalized[:80]}",
            value=normalized[:120],
        )

    return None


def _memory_permissions(state: AgentState) -> dict[str, Any]:
    options: dict[str, Any] = state.get("options", {}) or {}
    memory_options = options.get("memory") or {}

    if isinstance(memory_options, dict):
        return memory_options

    return {}


def _has_write_trigger(text: str) -> bool:
    return any(trigger in text for trigger in WRITE_TRIGGERS)


def _is_sensitive_health_text(text: str) -> bool:
    return any(keyword in text for keyword in SENSITIVE_HEALTH_KEYWORDS)


def _is_high_risk_state(state: AgentState) -> bool:
    intent_result = state.get("intent_result") or {}
    safety_result = state.get("safety_result") or {}
    risk_level = intent_result.get("risk_level") or safety_result.get("risk_level")
    return risk_level in {"HIGH", "EMERGENCY"}


def _answer_detail_value(text: str) -> str | None:
    if any(word in text for word in ["简短", "短一点", "少说", "别太长", "精简"]):
        return "简短"

    if any(word in text for word in ["详细", "多解释", "展开", "讲清楚"]):
        return "详细"

    return None


def _preferred_feature(text: str) -> str | None:
    if any(word in text for word in ["舌象分析", "舌像分析", "看舌头", "舌头照片"]):
        return "舌象分析"

    if "报告" in text:
        return "报告解释"

    if any(word in text for word in ["健康知识", "中医知识", "科普"]):
        return "健康知识问答"

    return None
