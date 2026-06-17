from langgraph.graph import END, StateGraph

from app.agent.nodes.general_chat_node import general_chat_node
from app.agent.nodes.intent_node import intent_node
from app.agent.nodes.memory_node import memory_node
from app.agent.nodes.route_node import route_node, select_next_route
from app.agent.nodes.safety_node import safety_node
from app.agent.nodes.tongue_analysis_node import tongue_analysis_node
from app.agent.state import AgentState


def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("memory_node", memory_node)
    graph.add_node("intent_node", intent_node)
    graph.add_node("route_node", route_node)
    graph.add_node("general_chat_node", general_chat_node)
    graph.add_node("safety_node", safety_node)
    graph.add_node("tongue_analysis_node", tongue_analysis_node)

    graph.set_entry_point("memory_node")
    graph.add_edge("memory_node", "intent_node")
    graph.add_edge("intent_node", "route_node")
    graph.add_conditional_edges(
        "route_node",
        select_next_route,
        {
            "general_chat_node": "general_chat_node",
            "safety_node": "safety_node",
            "tongue_analysis_node": "tongue_analysis_node",
        },
    )

    graph.add_edge("general_chat_node", END)
    graph.add_edge("safety_node", END)
    graph.add_edge("tongue_analysis_node", END)

    return graph


def compile_agent_graph(*, checkpointer, store):
    return build_agent_graph().compile(checkpointer=checkpointer, store=store)
