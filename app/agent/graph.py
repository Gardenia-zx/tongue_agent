from langgraph.graph import END, StateGraph

from app.agent.nodes.general_chat_node import general_chat_node
from app.agent.nodes.health_qa_node import health_qa_node
from app.agent.nodes.intent_node import intent_node
from app.agent.nodes.memory_node import memory_commit_node, memory_read_node
from app.agent.nodes.privacy_request_node import privacy_request_node
from app.agent.nodes.route_node import route_node, select_next_route
from app.agent.nodes.safety_node import safety_node
from app.agent.nodes.tongue_analysis_node import tongue_analysis_node
from app.agent.nodes.tongue_report_node import tongue_report_node
from app.agent.state import AgentState


def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("memory_read_node", memory_read_node)
    graph.add_node("memory_commit_node", memory_commit_node)
    graph.add_node("intent_node", intent_node)
    graph.add_node("route_node", route_node)
    graph.add_node("general_chat_node", general_chat_node)
    graph.add_node("health_qa_node", health_qa_node)
    graph.add_node("privacy_request_node", privacy_request_node)
    graph.add_node("safety_node", safety_node)
    graph.add_node("tongue_analysis_node", tongue_analysis_node)
    graph.add_node("tongue_report_node", tongue_report_node)

    graph.set_entry_point("memory_read_node")
    graph.add_edge("memory_read_node", "intent_node")
    graph.add_edge("intent_node", "route_node")
    graph.add_conditional_edges(
        "route_node",
        select_next_route,
        {
            "general_chat_node": "general_chat_node",
            "health_qa_node": "health_qa_node",
            "privacy_request_node": "privacy_request_node",
            "safety_node": "safety_node",
            "tongue_analysis_node": "tongue_analysis_node",
        },
    )

    graph.add_edge("general_chat_node", "memory_commit_node")
    graph.add_edge("health_qa_node", "memory_commit_node")
    graph.add_conditional_edges(
        "tongue_analysis_node",
        select_tongue_analysis_next,
        {
            "tongue_report_node": "tongue_report_node",
            "memory_commit_node": "memory_commit_node",
        },
    )
    graph.add_edge("tongue_report_node", "memory_commit_node")
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
