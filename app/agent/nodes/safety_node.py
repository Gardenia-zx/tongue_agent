from typing import Any

from app.agent.state import AgentState


EMERGENCY_KEYWORDS = [
    "胸痛",
    "呼吸困难",
    "昏迷",
    "大出血",
    "严重过敏",
    "休克",
    "意识不清",
    "抽搐",
]

DIAGNOSIS_KEYWORDS = [
    "确诊",
    "诊断",
    "是不是病",
    "判断是不是",
]


def _extract_user_text(state: AgentState) -> str:
    message: dict[str, Any] = state.get("message", {})
    content = message.get("content")

    if isinstance(content, str):
        return content.strip()

    return ""


def _detect_safety_type(text: str, intent_result: dict[str, Any]) -> tuple[str, str]:
    risk_level = intent_result.get("risk_level")
    safety_flags = intent_result.get("safety_flags") or []

    if risk_level == "EMERGENCY" or "EMERGENCY_INTENT" in safety_flags:
        return "EMERGENCY", "检测到急症或严重不适风险"

    if risk_level == "HIGH" or "HIGH_RISK_INTENT" in safety_flags:
        return "DIAGNOSIS_REQUEST", "检测到明确诊断请求"

    for keyword in EMERGENCY_KEYWORDS:
        if keyword in text:
            return "EMERGENCY", f"命中急症关键词：{keyword}"

    for keyword in DIAGNOSIS_KEYWORDS:
        if keyword in text:
            return "DIAGNOSIS_REQUEST", f"命中诊断关键词：{keyword}"

    return "GENERAL_SAFETY", "进入安全兜底处理"


def _build_safety_reply(safety_type: str) -> str:
    if safety_type == "EMERGENCY":
        return (
            "你提到的情况可能涉及急症或较高风险。"
            "我不能通过线上对话判断病情或替代医生处理。"
            "如果正在出现胸痛、呼吸困难、意识不清、大出血、严重过敏等情况，"
            "请立即联系当地急救电话或尽快前往线下医疗机构。"
        )

    if safety_type == "DIAGNOSIS_REQUEST":
        return (
            "我不能通过线上对话为你确诊疾病，或替代医生做医疗判断。"
            "我可以解释相关概念、常见可能方向、需要观察的信息和就医沟通要点。"
            "如果你希望判断是否患有某种疾病，建议结合线下检查和医生面诊。"
        )

    return (
        "这个问题需要谨慎处理。我可以提供一般健康管理信息，"
        "但不能做疾病确诊或替代医生判断。"
    )


async def safety_node(state: AgentState) -> AgentState:
    text = _extract_user_text(state)
    intent_result = state.get("intent_result") or {}

    safety_type, reason = _detect_safety_type(text, intent_result)
    content = _build_safety_reply(safety_type)

    safety_result = {
        "schema_version": "1.0",
        "stage": "INTENT_OR_INPUT",
        "passed": False,
        "risk_level": intent_result.get("risk_level", "HIGH"),
        "action": "BLOCK_OR_GUIDE",
        "rule_version": "safety-rule-mvp-v1.0",
        "risk_type": safety_type,
        "reason": reason,
        "rewrite_required": False,
        "human_review_required": safety_type in {"DIAGNOSIS_REQUEST", "EMERGENCY"},
    }

    return {
        **state,
        "current_node": "safety_node",
        "safety_result": safety_result,
        "response_message": {
            "role": "assistant",
            "content_type": "text",
            "content": content,
        },
        "next_action": {
            "type": "RESPOND_TO_USER",
            "payload": {
                "status": "REJECTED",
                "risk_type": safety_type,
                "human_review_required": safety_result["human_review_required"],
            },
        },
    }
