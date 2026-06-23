from typing import Any

from app.agent.state import AgentState
from app.agent.nodes.rag_node_utils import answer_with_rag


def _extract_user_text(state: AgentState) -> str:
    message: dict[str, Any] = state.get("message", {})
    content = message.get("content")

    if isinstance(content, str):
        return content.strip()

    return ""


async def health_qa_node(state: AgentState) -> AgentState:
    query = _extract_user_text(state)
    rag_context = await answer_with_rag(query)

    return {
        **state,
        "current_node": "health_qa_node",
        "rag_context": rag_context,
        "response_message": {
            "role": "assistant",
            "content_type": "text",
            "content": rag_context["answer"],
        },
        "next_action": {
            "type": "RESPOND_TO_USER",
            "payload": {
                "status": "COMPLETED",
                "route_target": "health_qa_subgraph",
                "grounded": rag_context.get("grounded", False),
                "hit_count": len(rag_context.get("hits") or []),
            },
        },
    }
