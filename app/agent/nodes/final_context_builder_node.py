from app.agent.context_builder import build_final_prompt_context, ensure_current_turn
from app.agent.nodes.agent_loop_node import AGENT_LOOP_SYSTEM_PROMPT
from app.agent.state import AgentState


async def final_context_builder_node(state: AgentState) -> AgentState:
    current_turn = ensure_current_turn(state)
    final_prompt_context = build_final_prompt_context(
        {**state, "current_turn": current_turn},
        system_prompt=AGENT_LOOP_SYSTEM_PROMPT,
        node_name="final_context_builder_node",
        include_long_term_memory=True,
    )
    current_turn["final_prompt_context"] = final_prompt_context
    return {
        **state,
        "current_turn": current_turn,
        "current_node": "final_context_builder_node",
        "prompt_context": final_prompt_context,
    }
