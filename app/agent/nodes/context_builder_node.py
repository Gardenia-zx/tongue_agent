from app.agent.context_builder import with_query_rewrite_context
from app.agent.state import AgentState


PRE_INTENT_SYSTEM_PROMPT = """You are the pre-intent context material builder.
Only prepare trusted short-term materials for query rewrite and intent routing.
Do not resolve references, do not create standalone_query, and do not read long-term memory.
"""


async def query_rewrite_context_builder_node(state: AgentState) -> AgentState:
    next_state = with_query_rewrite_context(state)
    return {
        **next_state,
        "current_node": "query_rewrite_context_builder",
    }


context_builder_node = query_rewrite_context_builder_node
