from app.agent.context_builder import build_final_prompt_context, ensure_current_turn
from app.agent.nodes.agent_loop_node import AGENT_LOOP_SYSTEM_PROMPT
from app.agent.state import AgentState


async def final_context_builder_node(state: AgentState) -> AgentState:
    current_turn = ensure_current_turn(state)
    business_context = current_turn.get("business_context") or {}
    report_context = business_context.get("report_context_resolution") or {}
    intent_result = dict(state.get("intent_result") or {})
    if report_context:
        intent_result["report_context"] = report_context

    final_prompt_context = build_final_prompt_context(
        {**state, "current_turn": current_turn, "intent_result": intent_result},
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
        "intent_result": intent_result,
    }
