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
    "疾病",
    "诊断",
    "病史",
    "症状",
]


@dataclass(frozen=True)
class MemoryPolicyDecision:
    can_read: bool
    can_write: bool
    write_candidate: str | None
    skip_reason: str | None = None


def extract_user_text(state: AgentState) -> str:
    message: dict[str, Any] = state.get("message", {})
    content = message.get("content")

    if isinstance(content, str):
        return content.strip()

    return ""


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


def decide_memory_policy(state: AgentState) -> MemoryPolicyDecision:
    permissions = _memory_permissions(state)
    text = extract_user_text(state)

    can_read = bool(permissions.get("can_read", True))
    consent_write = bool(permissions.get("can_write", False))

    if not text:
        return MemoryPolicyDecision(
            can_read=can_read,
            can_write=False,
            write_candidate=None,
            skip_reason="empty_user_text",
        )

    if not _has_write_trigger(text):
        return MemoryPolicyDecision(
            can_read=can_read,
            can_write=False,
            write_candidate=None,
            skip_reason="no_explicit_memory_request",
        )

    if not consent_write:
        return MemoryPolicyDecision(
            can_read=can_read,
            can_write=False,
            write_candidate=None,
            skip_reason="missing_long_term_memory_consent",
        )

    if _is_sensitive_health_text(text):
        return MemoryPolicyDecision(
            can_read=can_read,
            can_write=False,
            write_candidate=None,
            skip_reason="sensitive_health_memory_blocked",
        )

    return MemoryPolicyDecision(
        can_read=can_read,
        can_write=True,
        write_candidate=text,
        skip_reason=None,
    )
