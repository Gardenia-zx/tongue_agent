from langgraph.graph import END, StateGraph

from app.agent.nodes.agent_gate_node import agent_gate_node, select_agent_gate_next
from app.agent.nodes.context_builder_node import context_builder_node
from app.agent.nodes.final_context_builder_node import final_context_builder_node
from app.agent.nodes.general_chat_node import general_chat_node
from app.agent.nodes.health_qa_node import health_qa_node
from app.agent.nodes.intent_node import intent_node
from app.agent.nodes.memory_node import memory_commit_node, memory_read_node, memory_recall_node
from app.agent.nodes.privacy_request_node import privacy_request_node
from app.agent.nodes.query_rewrite_node import query_rewrite_node
from app.agent.nodes.report_followup_node import report_followup_node
from app.agent.nodes.safety_node import safety_node
from app.agent.runtime import build_agent_runtime_subgraph
from app.agent.state import AgentState
from app.agent.turn_lifecycle import (
    begin_turn_node,
    final_response_guard_node,
    runtime_context_policy_node,
)


AGENT_RUNTIME_RECURSION_LIMIT = 64



def build_agent_graph():
    graph = StateGraph(AgentState)

    # The parent checkpointer is propagated to this child graph. Each planner,
    # validator, tool and budget step remains independently recoverable.
    agent_runtime_subgraph = build_agent_runtime_subgraph().compile().with_config(
        {"recursion_limit": AGENT_RUNTIME_RECURSION_LIMIT}
    )

    graph.add_node("begin_turn_node", begin_turn_node)
    graph.add_node("memory_read_node", memory_read_node)
    graph.add_node("memory_recall_node", memory_recall_node)
    graph.add_node("memory_commit_node", memory_commit_node)
    graph.add_node("query_rewrite_context_builder", context_builder_node)
    graph.add_node("query_rewrite_node", query_rewrite_node)
    graph.add_node("intent_node", intent_node)
    graph.add_node("agent_gate_node", agent_gate_node)
    graph.add_node("final_context_builder_node", final_context_builder_node)
    graph.add_node("runtime_context_policy_node", runtime_context_policy_node)
    graph.add_node("agent_runtime_subgraph", agent_runtime_subgraph)
    graph.add_node("general_chat_node", general_chat_node)
    graph.add_node("health_qa_node", health_qa_node)
    graph.add_node("privacy_request_node", privacy_request_node)
    graph.add_node("report_followup_node", report_followup_node)
    graph.add_node("safety_node", safety_node)
    graph.add_node("final_response_guard_node", final_response_guard_node)
    graph.add_node("terminal_response_guard_node", final_response_guard_node)

    graph.set_entry_point("begin_turn_node")
    graph.add_edge("begin_turn_node", "memory_read_node")
    graph.add_edge("memory_read_node", "query_rewrite_context_builder")
    graph.add_edge("query_rewrite_context_builder", "query_rewrite_node")
    graph.add_edge("query_rewrite_node", "intent_node")
    graph.add_edge("intent_node", "agent_gate_node")
    graph.add_conditional_edges(
        "agent_gate_node",
        select_agent_gate_next,
        {
            "privacy_request_node": "privacy_request_node",
            "safety_node": "safety_node",
            "general_chat_node": "general_chat_node",
            "agent_loop_node": "memory_recall_node",
        },
    )

    graph.add_edge("memory_recall_node", "final_context_builder_node")
    graph.add_edge("final_context_builder_node", "runtime_context_policy_node")
    graph.add_edge("runtime_context_policy_node", "agent_runtime_subgraph")

    # All ordinary responses pass ownership validation before memory persistence.
    graph.add_edge("agent_runtime_subgraph", "final_response_guard_node")
    graph.add_edge("general_chat_node", "final_response_guard_node")
    graph.add_edge("health_qa_node", "final_response_guard_node")
    graph.add_edge("report_followup_node", "final_response_guard_node")
    graph.add_edge("final_response_guard_node", "memory_commit_node")
    graph.add_edge("memory_commit_node", END)

    # Privacy and high-risk branches remain non-persistent, but still receive the
    # same final response ownership and non-empty-content validation.
    graph.add_edge("safety_node", "terminal_response_guard_node")
    graph.add_edge("privacy_request_node", "terminal_response_guard_node")
    graph.add_edge("terminal_response_guard_node", END)

    return graph



def compile_agent_graph(*, checkpointer, store):
    return build_agent_graph().compile(checkpointer=checkpointer, store=store)



def select_tongue_analysis_next(state: AgentState) -> str:
    next_action = state.get("next_action") or {}
    if next_action.get("type") == "TONGUE_FEATURES_READY":
        return "tongue_report_node"
    return "memory_commit_node"
