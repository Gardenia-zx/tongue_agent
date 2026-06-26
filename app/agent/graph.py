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


AGENT_RUNTIME_RECURSION_LIMIT = 64


def build_agent_graph():
    graph = StateGraph(AgentState)

    # Compile the child graph without a checkpointer. The parent graph supplies its
    # persistence implementation when compile_agent_graph() is called, allowing
    # planner/tool/budget steps inside the runtime to participate in checkpointing.
    # A child-specific recursion limit protects legitimate multi-step tool plans
    # from LangGraph's lower default while the runtime's own budgets prevent loops.
    agent_runtime_subgraph = build_agent_runtime_subgraph().compile().with_config(
        {"recursion_limit": AGENT_RUNTIME_RECURSION_LIMIT}
    )

    graph.add_node("memory_read_node", memory_read_node)
    graph.add_node("memory_recall_node", memory_recall_node)
    graph.add_node("memory_commit_node", memory_commit_node)
    graph.add_node("query_rewrite_context_builder", context_builder_node)
    graph.add_node("query_rewrite_node", query_rewrite_node)
    graph.add_node("intent_node", intent_node)
    graph.add_node("agent_gate_node", agent_gate_node)
    graph.add_node("final_context_builder_node", final_context_builder_node)
    graph.add_node("agent_runtime_subgraph", agent_runtime_subgraph)
    graph.add_node("general_chat_node", general_chat_node)
    graph.add_node("health_qa_node", health_qa_node)
    graph.add_node("privacy_request_node", privacy_request_node)
    graph.add_node("report_followup_node", report_followup_node)
    graph.add_node("safety_node", safety_node)

    graph.set_entry_point("memory_read_node")
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
            # Keep the existing selector contract. The selected agent-loop route
            # now enters memory recall and then the checkpointable runtime subgraph.
            "agent_loop_node": "memory_recall_node",
        },
    )

    graph.add_edge("memory_recall_node", "final_context_builder_node")
    graph.add_edge("final_context_builder_node", "agent_runtime_subgraph")
    graph.add_edge("agent_runtime_subgraph", "memory_commit_node")
    graph.add_edge("general_chat_node", "memory_commit_node")
    graph.add_edge("health_qa_node", "memory_commit_node")
    graph.add_edge("report_followup_node", "memory_commit_node")
    graph.add_edge("memory_commit_node", END)
    graph.add_edge("safety_node", END)
    graph.add_edge("privacy_request_node", END)

    return graph


def compile_agent_graph(*, checkpointer, store):
    return build_agent_graph().compile(checkpointer=checkpointer, store=store)


def select_tongue_analysis_next(state: AgentState) -> str:
    next_action = state.get("next_action") or {}
    if next_action.get("type") == "TONGUE_FEATURES_READY":
        return "tongue_report_node"

    return "memory_commit_node"
