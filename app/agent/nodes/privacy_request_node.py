from typing import Any

from app.agent.state import AgentState
from app.memory.policy import extract_user_text
from app.memory.service import MemoryService


DELETE_KEYWORDS = ["删除", "清除", "撤回", "取消授权", "遗忘", "不要保存"]
VIEW_KEYWORDS = ["查看", "保存了什么", "记住了什么", "已保存", "有哪些记忆"]


async def privacy_request_node(state: AgentState, *, store) -> AgentState:
    user_id = state.get("user_id")
    text = extract_user_text(state)
    service = MemoryService(store=store)

    if user_id is None:
        content = "我没有拿到用户身份信息，暂时不能查看或删除长期记忆。"
        payload: dict[str, Any] = {
            "status": "FAILED",
            "privacy_action": "missing_user_id",
        }
    elif _is_delete_request(text):
        result = await service.delete_user_memory(user_id=user_id)
        content = (
            "已删除当前用户的长期记忆、用户画像和记忆摘要。"
            "报告和图片删除需要后端报告服务支持，本次没有处理报告或图片文件。"
        )
        payload = {
            "status": "COMPLETED",
            "privacy_action": "delete_long_term_memory",
            **result,
        }
    else:
        result = await service.list_user_memory(user_id=user_id)
        content = _format_saved_memory(result)
        payload = {
            "status": "COMPLETED",
            "privacy_action": "view_long_term_memory",
            "memory_count": len(result.get("memories") or []),
            "summary_count": len(result.get("summaries") or []),
            "has_profile": bool(result.get("profile")),
        }

    return {
        **state,
        "current_node": "privacy_request_node",
        "response_message": {
            "role": "assistant",
            "content_type": "text",
            "content": content,
        },
        "next_action": {
            "type": "RESPOND_TO_USER",
            "payload": payload,
        },
    }


def _is_delete_request(text: str) -> bool:
    return any(keyword in text for keyword in DELETE_KEYWORDS)


def _format_saved_memory(result: dict[str, Any]) -> str:
    profile = result.get("profile") or {}
    memories = result.get("memories") or []
    summaries = result.get("summaries") or []

    if not profile and not memories and not summaries:
        return "当前没有保存可展示的长期记忆或用户画像。"

    lines = ["当前保存的长期记忆如下："]
    profile_summary = profile.get("summary")
    if profile_summary:
        lines.append(f"用户画像：{profile_summary}")

    for memory in memories[:10]:
        summary = memory.get("summary") or memory.get("text")
        if summary:
            lines.append(f"记忆：{summary}")

    if len(memories) > 10:
        lines.append(f"还有 {len(memories) - 10} 条记忆未展示。")

    if summaries:
        lines.append(f"已生成 {len(summaries)} 条长期记忆压缩摘要。")

    lines.append("你可以继续说“删除我的长期记忆”来清除这些信息。")
    return "\n".join(lines)
