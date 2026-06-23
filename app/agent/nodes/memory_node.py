from typing import Any

from app.agent.state import AgentState
from app.integrations.redis_client import create_redis_client
from app.memory.service import MemoryService
from app.memory.session_cache import ShortTermMemoryCache


async def memory_read_node(state: AgentState, *, store) -> AgentState:
    session_context = await _load_session_context(state)
    memory_context = await MemoryService(store=store).build_memory_context(
        state,
        session_context=session_context,
    )

    return {
        **state,
        "current_node": "memory_read_node",
        "memory_context": memory_context,
    }


async def memory_commit_node(state: AgentState, *, store) -> AgentState:
    short_term_result = await _append_short_term_turn(state)
    long_term_result = await MemoryService(store=store).commit_after_response(state)

    memory_context = dict(state.get("memory_context") or {})
    memory_context["write_result"] = long_term_result
    memory_context["short_term_write_result"] = short_term_result

    extensions = dict(state.get("extensions") or {})
    extensions["last_internal_node"] = "memory_commit_node"

    return {
        **state,
        "current_node": state.get("current_node"),
        "memory_context": memory_context,
        "extensions": extensions,
    }


async def _load_session_context(state: AgentState) -> dict[str, Any]:
    redis = create_redis_client()
    try:
        return await ShortTermMemoryCache(redis=redis).load_context(
            state.get("thread_id")
        )
    except Exception as exc:
        return {
            "cache_hit": False,
            "source": "redis",
            "conversation_summary": "",
            "recent_turns": [],
            "turn_count": 0,
            "skip_reason": f"redis_unavailable:{type(exc).__name__}",
        }
    finally:
        await redis.close()


async def _append_short_term_turn(state: AgentState) -> dict[str, Any]:
    redis = create_redis_client()
    try:
        return await ShortTermMemoryCache(redis=redis).append_turn(state)
    except Exception as exc:
        return {
            "status": "SKIPPED",
            "reason": f"redis_unavailable:{type(exc).__name__}",
        }
    finally:
        await redis.close()
