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

PRESCRIPTION_KEYWORDS = [
    "开药",
    "处方",
    "吃什么药",
    "药量",
    "剂量",
    "停药",
    "换药",
    "加药",
    "减药",
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
        return "HIGH_RISK_MEDICAL", "检测到高风险医疗请求"

    for keyword in EMERGENCY_KEYWORDS:
        if keyword in text:
            return "EMERGENCY", f"命中急症关键词：{keyword}"

    for keyword in PRESCRIPTION_KEYWORDS:
        if keyword in text:
            return "PRESCRIPTION_REQUEST", f"命中用药关键词：{keyword}"

    return "GENERAL_SAFETY", "进入安全兜底处理"


def _build_safety_reply(safety_type: str) -> str:
    if safety_type == "EMERGENCY":
        return (
            "你提到的情况可能涉及急症或较高风险。"
            "我不能通过线上对话判断病情或替代医生处理。"
            "如果正在出现胸痛、呼吸困难、意识不清、大出血、严重过敏等情况，"
            "请立即联系当地急救电话或尽快前往线下医疗机构。"
        )

    if safety_type == "PRESCRIPTION_REQUEST":
        return (
            "我不能提供处方、药物剂量、停药或换药建议。"
            "这些需要医生结合病史、检查结果和正在使用的药物综合判断。"
            "如果你有用药相关问题，建议咨询线下医生或药师。"
        )

    if safety_type == "HIGH_RISK_MEDICAL":
        return (
            "这个问题涉及较高风险的医疗判断。"
            "我可以提供一般健康知识说明，但不能做诊断、处方或替代医生建议。"
            "如果症状明显、持续加重，或你正在用药、孕期、儿童、老人等特殊情况，"
            "建议及时线下就医或咨询专业医生。"
        )

    return (
        "这个问题需要谨慎处理。我可以提供一般健康管理信息，"
        "但不能做疾病诊断、处方、药物剂量或替代医生判断。"
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
        "human_review_required": safety_type in {"HIGH_RISK_MEDICAL", "EMERGENCY"},
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
