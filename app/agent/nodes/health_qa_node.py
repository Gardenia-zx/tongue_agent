from typing import Any

from app.agent.context_builder import effective_user_query, with_prompt_context
from app.agent.state import AgentState
from app.agent.nodes.rag_node_utils import answer_with_rag


def _extract_user_text(state: AgentState) -> str:
    message: dict[str, Any] = state.get("message", {})
    content = message.get("content")

    if isinstance(content, str):
        return content.strip()

    return ""


async def health_qa_node(state: AgentState) -> AgentState:
    state = with_prompt_context(
        state,
        mode="FULL_FOR_NODE",
        system_prompt="你是中医健康知识问答助手。请基于用户独立问题、最近对话和知识库资料回答，不做诊断，不开处方。",
        node_name="health_qa_node",
        include_long_term_memory=True,
    )
    query = effective_user_query(state) or _extract_user_text(state)
    rag_context = await answer_with_rag(query)
    structured_content = {
        "schema_version": "1.0",
        "answer_type": "HEALTH_QA",
        "title": "健康知识问答",
        "summary": rag_context["answer"],
        "highlights": ["已结合知识库资料"] if rag_context.get("grounded") else [],
        "sections": [],
        "disclaimer": "以上内容用于一般健康知识说明和健康管理参考，不能替代医生诊断。",
    }

    return {
        **state,
        "current_node": "health_qa_node",
        "rag_context": rag_context,
        "response_message": {
            "role": "assistant",
            "content_type": "text",
            "content": rag_context["answer"],
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
