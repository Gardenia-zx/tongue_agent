from app.agent.state import AgentState
from app.memory.service import MemoryService


async def memory_node(state: AgentState, *, store) -> AgentState:
    memory_context = await MemoryService(store=store).build_memory_context(state)

    return {
        **state,
        "current_node": "memory_node",
        "memory_context": memory_context,
    }
