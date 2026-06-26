from app.agent.context_builder import build_prompt_context
from app.agent.state import AgentState


PRE_INTENT_SYSTEM_PROMPT = """你是中医舌象健康 Agent 的上下文构建器。
本阶段只为追问消解和意图识别准备最小上下文，不回答用户问题。
上下文必须优先使用最近完整对话、上一轮最终回答、历史摘要和当前活动报告。
不得用长期记忆伪造上一轮对话或报告内容。
"""


async def context_builder_node(state: AgentState) -> AgentState:
    prompt_context = build_prompt_context(
        state,
        mode="MINIMAL_PRE_INTENT",
        system_prompt=PRE_INTENT_SYSTEM_PROMPT,
        node_name="context_builder_node",
        include_long_term_memory=False,
    )
    return {
        **state,
        "current_node": "context_builder_node",
        "prompt_context": prompt_context,
    }
