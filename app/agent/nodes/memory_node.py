from datetime import UTC, datetime
from typing import Any

from app.agent.state import AgentState
from app.core.config import get_settings
from app.memory.service import MemoryService


CHECKPOINTER_SESSION_SOURCE = "checkpointer"
MYSQL_RECOVERY_SESSION_SOURCE = "mysql_recovery"


async def memory_read_node(state: AgentState, *, store) -> AgentState:
    """Load only short-term/session material before intent and gate."""
    if _is_stateless_context(state):
        memory_context = _stateless_memory_context(state)
    else:
        session_context = _load_session_context(state)
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

    short_term_result = _append_short_term_turn(state)
    long_term_result = await MemoryService(store=store).commit_after_response(state)

    memory_context = dict(state.get("memory_context") or {})
    memory_context["write_result"] = long_term_result
    memory_context["short_term_write_result"] = short_term_result

    extensions = dict(state.get("extensions") or {})
    extensions["last_internal_node"] = "memory_commit_node"

    return {
        **state,
        "current_node": state.get("current_node"),
        "short_term_memory": short_term_result.get("context")
        or state.get("short_term_memory")
        or {},
        "memory_context": memory_context,
        "extensions": extensions,
    }


def _load_session_context(state: AgentState) -> dict[str, Any]:
    checkpoint_context = _checkpoint_session_context(state)
    if _has_session_material(checkpoint_context):
        return checkpoint_context

    if _is_mysql_recovery_context(state):
        return _mysql_recovery_session_context(state)

    return checkpoint_context


def _is_stateless_context(state: AgentState) -> bool:
    return _context_mode(state) == "stateless"


def _is_mysql_recovery_context(state: AgentState) -> bool:
    return _context_mode(state) == "mysql_recovery"


def _context_mode(state: AgentState) -> str:
    options = state.get("options") or {}
    context_options = options.get("context") or {}
    if not isinstance(context_options, dict):
        return ""
    return str(context_options.get("mode") or "").strip().lower()


def _short_term_memory_context(session_context: dict[str, Any]) -> dict[str, Any]:
    recent_turns = session_context.get("recent_turns") or []
    if not isinstance(recent_turns, list):
        recent_turns = []
    recent_messages = session_context.get("recent_messages") or []
    if not isinstance(recent_messages, list):
        recent_messages = []
    return {
        "used_memory_ids": [],
        "relevant_memories": [],
        "memories": [],
        "summaries": [],
        "profile": {},
        "session": session_context,
        "conversation_summary": session_context.get("conversation_summary") or "",
        "recent_turns": recent_turns,
        "recent_messages": recent_messages,
        "last_final_answer": session_context.get("last_final_answer"),
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
    if not session_context.get("recent_messages"):
        session_context["recent_messages"] = memory_context.get("recent_messages") or []
    if not session_context.get("last_final_answer"):
        session_context["last_final_answer"] = memory_context.get("last_final_answer")
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
        "recent_turns": [],
        "recent_messages": recent_messages if isinstance(recent_messages, list) else [],
        "last_final_answer": context_bundle.get("last_final_answer"),
        "turn_count": len(recent_messages) if isinstance(recent_messages, list) else 0,
    }


def _can_read_long_term_memory(state: AgentState) -> bool:
    options = state.get("options") or {}
    memory_options = options.get("memory") or {}
    if not isinstance(memory_options, dict):
        return False
    return bool(memory_options.get("can_read"))


def _append_short_term_turn(state: AgentState) -> dict[str, Any]:
    memory_context = dict(state.get("memory_context") or {})
    session_context = _session_context_from_memory_context(memory_context)
    turn = _build_turn(state)
    messages = _build_current_messages(state)
    if turn is None and not messages:
        return {"status": "SKIPPED", "reason": "empty_turn"}

    settings = get_settings()
    keep_turns = max(1, int(settings.short_term_memory_keep_last_turns))
    recent_turns = [
        item for item in (session_context.get("recent_turns") or []) if isinstance(item, dict)
    ]
    turn_count = int(session_context.get("turn_count") or len(recent_turns))
    if turn is not None:
        recent_turns.append(turn)
        turn_count += 1

    recent_messages = [
        item
        for item in (session_context.get("recent_messages") or [])
        if isinstance(item, dict)
    ]
    recent_messages.extend(messages)

    conversation_summary = str(session_context.get("conversation_summary") or "")
    compressed = False
    if _should_compress(
        recent_turns=recent_turns,
        turn_count=turn_count,
        summary_turn_threshold=settings.short_term_memory_summary_turn_threshold,
        summary_char_threshold=settings.short_term_memory_summary_char_threshold,
    ) and len(recent_turns) > keep_turns:
        old_turns = recent_turns[:-keep_turns]
        recent_turns = recent_turns[-keep_turns:]
        conversation_summary = _compress_turns(
            previous_summary=conversation_summary,
            turns=old_turns,
        )
        compressed = True

    context = {
        "schema_version": "1.0",
        "source": CHECKPOINTER_SESSION_SOURCE,
        "thread_id": state.get("thread_id"),
        "conversation_id": state.get("conversation_id"),
        "conversation_summary": conversation_summary,
        "recent_turns": recent_turns[-keep_turns:],
        "recent_messages": recent_messages[-keep_turns * 2 :],
        "last_final_answer": _last_assistant_message(recent_messages),
        "turn_count": turn_count,
        "updated_at": _now(),
    }
    return {
        "status": "UPDATED",
        "source": CHECKPOINTER_SESSION_SOURCE,
        "thread_id": state.get("thread_id"),
        "turn_count": turn_count,
        "recent_turn_count": len(context["recent_turns"]),
        "compressed": compressed,
        "context": context,
    }


def _checkpoint_session_context(state: AgentState) -> dict[str, Any]:
    value = state.get("short_term_memory") or {}
    if not isinstance(value, dict):
        value = {}
    return _normalize_session_context(
        value,
        source=CHECKPOINTER_SESSION_SOURCE,
        skip_reason="checkpoint_short_term_missing",
    )


def _mysql_recovery_session_context(state: AgentState) -> dict[str, Any]:
    return _normalize_session_context(
        state.get("context_bundle") or {},
        source=MYSQL_RECOVERY_SESSION_SOURCE,
        skip_reason="mysql_recovery_context_missing",
    )


def _normalize_session_context(
    value: dict[str, Any],
    *,
    source: str,
    skip_reason: str | None,
) -> dict[str, Any]:
    conversation_summary = value.get("conversation_summary") or ""
    if isinstance(conversation_summary, dict):
        conversation_summary = conversation_summary.get("text") or conversation_summary.get("summary") or ""
    elif not isinstance(conversation_summary, str):
        conversation_summary = ""

    recent_turns = value.get("recent_turns") or []
    if not isinstance(recent_turns, list):
        recent_turns = []

    recent_messages = value.get("recent_messages") or []
    if not isinstance(recent_messages, list):
        recent_messages = []

    return {
        "cache_hit": _has_session_material(value),
        "source": source,
        "thread_id": value.get("thread_id"),
        "conversation_id": value.get("conversation_id"),
        "conversation_summary": conversation_summary,
        "recent_turns": recent_turns,
        "recent_messages": recent_messages,
        "last_final_answer": value.get("last_final_answer"),
        "turn_count": int(value.get("turn_count") or len(recent_turns)),
        "updated_at": value.get("updated_at"),
        "skip_reason": None if _has_session_material(value) else skip_reason,
    }


def _has_session_material(value: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    return bool(
        value.get("recent_turns")
        or value.get("recent_messages")
        or value.get("last_final_answer")
        or value.get("conversation_summary")
    )


def _build_turn(state: AgentState) -> dict[str, Any] | None:
    user_text = _user_text(state)
    response_message = state.get("response_message") or {}
    assistant_text = response_message.get("content")
    if not user_text and not assistant_text:
        return None
    return {
        "turn_id": state.get("turn_id"),
        "request_id": state.get("request_id"),
        "node": state.get("current_node"),
        "user": user_text,
        "assistant": assistant_text if isinstance(assistant_text, str) else "",
        "created_at": _now(),
    }


def _build_current_messages(state: AgentState) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    user_text = _user_text(state)
    if user_text:
        result.append(
            {
                "message_id": state.get("user_message_id"),
                "role": "user",
                "content_type": (state.get("message") or {}).get("content_type") or "text",
                "content": user_text,
                "turn_id": state.get("turn_id"),
                "request_id": state.get("request_id"),
                "created_at": _now(),
            }
        )

    response_message = state.get("response_message") or {}
    assistant_text = response_message.get("content")
    if isinstance(assistant_text, str) and assistant_text.strip():
        result.append(
            {
                "message_id": state.get("assistant_message_id"),
                "role": "assistant",
                "content_type": response_message.get("content_type") or "text",
                "content": assistant_text.strip(),
                "turn_id": state.get("turn_id"),
                "request_id": state.get("request_id"),
                "node_name": state.get("current_node"),
                "answer_type": (state.get("next_action") or {}).get("payload", {}).get("answer_type")
                if isinstance((state.get("next_action") or {}).get("payload"), dict)
                else None,
                "report_id": state.get("report_id"),
                "structured_content": response_message.get("structured_content") or {},
                "created_at": _now(),
            }
        )
    return result


def _last_assistant_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant" and message.get("content"):
            return message
    return None


def _user_text(state: AgentState) -> str:
    message = state.get("message") or {}
    content = message.get("content")
    return content.strip() if isinstance(content, str) else ""


def _should_compress(
    *,
    recent_turns: list[dict[str, Any]],
    turn_count: int,
    summary_turn_threshold: int,
    summary_char_threshold: int,
) -> bool:
    if turn_count >= summary_turn_threshold:
        return True
    text_size = sum(
        len(str(turn.get("user") or "")) + len(str(turn.get("assistant") or ""))
        for turn in recent_turns
    )
    return text_size >= summary_char_threshold


def _compress_turns(
    *,
    previous_summary: str,
    turns: list[dict[str, Any]],
) -> str:
    lines = []
    if previous_summary:
        lines.append(previous_summary.strip())
    for turn in turns:
        user = str(turn.get("user") or "").strip()
        assistant = str(turn.get("assistant") or "").strip()
        if user:
            lines.append(f"用户曾说：{user[:160]}")
        if assistant:
            lines.append(f"助手曾答：{assistant[:160]}")
    return "\n".join(line for line in lines if line)[-3000:]


def _now() -> str:
    return datetime.now(UTC).isoformat()
