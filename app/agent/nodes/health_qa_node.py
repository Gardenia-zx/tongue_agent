import re
from typing import Any

from app.agent.context_builder import effective_user_query, with_prompt_context
from app.agent.state import AgentState
from app.agent.nodes.rag_node_utils import answer_with_rag
from app.agent.response_contract import structured_content_to_text


DISCLAIMER = "以上内容用于一般健康知识说明和健康管理参考，不能替代医生诊断。"


def _extract_user_text(state: AgentState) -> str:
    message: dict[str, Any] = state.get("message", {})
    content = message.get("content")

    if isinstance(content, str):
        return content.strip()

    return ""


def _plain_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    text = text.replace("```json", "").replace("```", "")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text)
    text = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s*)", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = text.replace("*", "").replace("_", "")
    return re.sub(r"\s+", " ", text).strip()


def _is_disclaimer(text: str) -> bool:
    return any(marker in text for marker in ("不能替代医生", "仅供", "只作为", "建议咨询医生"))


def _section_title(raw_line: str) -> str | None:
    line = raw_line.strip()
    heading = re.match(r"^#{1,6}\s*(.+)$", line)
    if heading:
        return _plain_text(heading.group(1)).rstrip(":：")
    bold = re.match(r"^\*\*(.+?)\*\*\s*[:：]?$", line)
    if bold:
        return _plain_text(bold.group(1)).rstrip(":：")
    if not re.match(r"^(?:[-*+]\s+|\d+[.)]\s*)", line) and len(line) <= 32 and line.endswith((":", "：")):
        return _plain_text(line).rstrip(":：")
    return None


def _short_summary(lines: list[str]) -> str:
    for line in lines:
        text = _plain_text(line)
        if text and not _is_disclaimer(text):
            return text[:140].rstrip()
    return "我暂时没有检索到足够资料，可以继续补充问题后再细化说明。"


def _structured_sections(answer: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    intro: list[str] = []

    def push_current() -> None:
        nonlocal current
        if current and (current.get("content") or current.get("items")):
            sections.append(current)
        current = None

    for raw_line in answer.splitlines():
        if not raw_line.strip():
            continue
        text = _plain_text(raw_line)
        if not text or _is_disclaimer(text):
            continue

        title = _section_title(raw_line)
        if title:
            push_current()
            current = {"title": title, "items": []}
            continue

        is_item = bool(re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s*)", raw_line))
        if is_item:
            if current is None:
                current = {"title": "参考说明", "items": []}
            current.setdefault("items", []).append(text)
            continue

        if current is None:
            intro.append(text)
        elif current.get("content"):
            current["content"] = f"{current['content']} {text}"
        else:
            current["content"] = text

    push_current()
    if intro:
        sections.insert(0, {"title": "参考说明", "content": " ".join(intro)})
    if sections:
        return sections[:5]
    fallback = _plain_text(answer)
    return [{"title": "参考说明", "content": fallback or "暂无可展示内容。"}]


async def health_qa_node(state: AgentState) -> AgentState:
    state = with_prompt_context(
        state,
        mode="FULL_FOR_NODE",
        system_prompt=(
            "你是中医健康知识问答助手。请基于用户独立问题、最近对话和知识库资料回答。"
            "不要确诊疾病；药物、方剂、处方类问题可以做一般知识参考，说明常见方向、禁忌和就医沟通要点。"
            "不要建议用户自行停药、换药、加药或减药；特殊人群和具体剂量提醒咨询医生或药师。"
        ),
        node_name="health_qa_node",
        include_long_term_memory=True,
    )
    query = effective_user_query(state) or _extract_user_text(state)
    rag_context = await answer_with_rag(query)
    raw_answer = str(rag_context.get("answer") or "")
    answer_lines = [line for line in raw_answer.splitlines() if line.strip()]
    summary = _short_summary(answer_lines)
    structured_content = {
        "schema_version": "1.0",
        "answer_type": "HEALTH_QA",
        "title": "健康知识问答",
        "summary": summary,
        "highlights": ["已结合知识库资料"] if rag_context.get("grounded") else [],
        "sections": _structured_sections(raw_answer),
        "disclaimer": DISCLAIMER,
    }
    content = _plain_text(raw_answer) or structured_content_to_text(structured_content) or summary

    return {
        **state,
        "current_node": "health_qa_node",
        "rag_context": rag_context,
        "response_message": {
            "role": "assistant",
            "content_type": "text",
            "content": content,
            "structured_content": structured_content,
        },
        "next_action": {
            "type": "RESPOND_TO_USER",
            "payload": {
                "status": "COMPLETED",
                "route_target": "health_qa_subgraph",
                "answer_type": "HEALTH_QA",
                "rag_query": query,
                "grounded": rag_context.get("grounded", False),
                "hit_count": len(rag_context.get("hits") or []),
            },
        },
    }
