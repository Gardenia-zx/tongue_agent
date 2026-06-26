from typing import Any

from app.agent.state import AgentState
from app.integrations.redis_client import create_redis_client
from app.memory.service import MemoryService
from app.memory.session_cache import ShortTermMemoryCache


async def memory_read_node(state: AgentState, *, store) -> AgentState:
    """Load only short-term/session material before intent and gate."""
    if _is_stateless_context(state):
        memory_context = _stateless_memory_context(state)
    else:
        session_context = await _load_session_context(state)
        memory_context = _short_term_memory_context(session_context)

    return {
        **state,
        "current_node": "memory_read_node",
        "memory_context": memory_context,
    }


async def memory_recall_node(state: AgentState, *, store) -> AgentState:
    """Recall long-term memory only after agent_gate allows normal business flow."""
    existing_context = dict(state.get("memory_context") or {})
    session_context = _session_context_from_memory_context(existing_context)
    long_term_context = await MemoryService(store=store).build_memory_context(
        state,
        session_context=session_context,
    )
    memory_context = {
        **existing_context,
        **long_term_context,
        "session": session_context,
        "conversation_summary": long_term_context.get("conversation_summary")
        or existing_context.get("conversation_summary"),
        "recent_turns": long_term_context.get("recent_turns")
        or existing_context.get("recent_turns")
        or [],
    }
    return {
        **state,
        "current_node": "memory_recall_node",
        "memory_context": memory_context,
    }


async def memory_commit_node(state: AgentState, *, store) -> AgentState:
    if _is_stateless_context(state):
        memory_context = dict(state.get("memory_context") or {})
        memory_context["write_result"] = {
            "status": "SKIPPED",
            "reason": "stateless_context_managed_by_backend",
        }
        memory_context["short_term_write_result"] = {
            "status": "SKIPPED",
            "reason": "stateless_context_managed_by_backend",
        }
        extensions = dict(state.get("extensions") or {})
        extensions["last_internal_node"] = "memory_commit_node"
        return {
            **state,
            "current_node": state.get("current_node"),
            "memory_context": memory_context,
            "extensions": extensions,
        }

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


def _is_stateless_context(state: AgentState) -> bool:
    context_bundle = state.get("context_bundle") or {}
    if context_bundle:
        return True

    options = state.get("options") or {}
    context_options = options.get("context") or {}
    return isinstance(context_options, dict) and context_options.get("mode") == "stateless"


def _short_term_memory_context(session_context: dict[str, Any]) -> dict[str, Any]:
    recent_turns = session_context.get("recent_turns") or []
    if not isinstance(recent_turns, list):
        recent_turns = []
    return {
        "used_memory_ids": [],
        "relevant_memories": [],
        "memories": [],
        "summaries": [],
        "profile": {},
        "session": session_context,
        "conversation_summary": session_context.get("conversation_summary") or "",
        "recent_turns": recent_turns,
        "write_result": None,
        "policy": {
            "can_read": False,
            "can_write": False,
            "skip_reason": "long_term_recall_deferred_until_agent_gate",
        },
        "context_summary": "short_term_context_loaded",
    }


def _stateless_memory_context(state: AgentState) -> dict[str, Any]:
    session_context = _backend_session_context(state)
    memory_context = _short_term_memory_context(session_context)
    memory_context["policy"] = {
        "can_read": _can_read_long_term_memory(state),
        "can_write": False,
        "skip_reason": "stateless_context_managed_by_backend",
    }
    memory_context["context_summary"] = "backend_context_bundle_loaded"
    return memory_context


def _session_context_from_memory_context(memory_context: dict[str, Any]) -> dict[str, Any]:
    session_context = memory_context.get("session")
    if not isinstance(session_context, dict):
        session_context = {}

    if not session_context.get("conversation_summary"):
        session_context["conversation_summary"] = memory_context.get("conversation_summary") or ""
    if not session_context.get("recent_turns"):
        session_context["recent_turns"] = memory_context.get("recent_turns") or []
    if "turn_count" not in session_context:
        session_context["turn_count"] = len(session_context.get("recent_turns") or [])
    if "cache_hit" not in session_context:
        session_context["cache_hit"] = False
    if "source" not in session_context:
        session_context["source"] = "memory_read_node"
    return session_context


def _backend_session_context(state: AgentState) -> dict[str, Any]:
    context_bundle = state.get("context_bundle") or {}
    conversation_summary = context_bundle.get("conversation_summary") or {}
    if isinstance(conversation_summary, dict):
        summary_text = conversation_summary.get("text") or ""
    elif isinstance(conversation_summary, str):
        summary_text = conversation_summary
    else:
        summary_text = ""

    recent_messages = context_bundle.get("recent_messages") or []
    return {
        "cache_hit": False,
        "source": "backend_context_bundle",
        "conversation_summary": summary_text,
        "recent_turns": recent_messages if isinstance(recent_messages, list) else [],
        "turn_count": len(recent_messages) if isinstance(recent_messages, list) else 0,
    }


def _can_read_long_term_memory(state: AgentState) -> bool:
    options = state.get("options") or {}
    memory_options = options.get("memory") or {}
    if not isinstance(memory_options, dict):
        return False
    return bool(memory_options.get("can_read"))


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
