from typing import Any, Literal

from app.agent.state import AgentState


ContextMode = Literal["MINIMAL_PRE_INTENT", "FULL_FOR_NODE"]


def extract_user_text(state: AgentState) -> str:
    # 提取当前对话信息，包含：用户输入，前几轮对话结果
    message: dict[str, Any] = state.get("message", {})
    # 提取对话消息内容
    content = message.get("content")
    return content.strip() if isinstance(content, str) else ""


def context_bundle_from_state(state: AgentState) -> dict[str, Any]:
    context_bundle = state.get("context_bundle") or {}
    if isinstance(context_bundle, dict) and context_bundle:
        return context_bundle

    client_context = state.get("client_context") or {}
    client_extra = client_context.get("extra") or {}
    extra_bundle = client_extra.get("context_bundle") or {}
    return extra_bundle if isinstance(extra_bundle, dict) else {}


def query_context_from_state(state: AgentState) -> dict[str, Any]:
    value = state.get("query_context") or {}
    return value if isinstance(value, dict) else {}


def effective_user_query(state: AgentState) -> str:
    query_context = query_context_from_state(state)
    standalone_query = query_context.get("standalone_query")
    if isinstance(standalone_query, str) and standalone_query.strip():
        return standalone_query.strip()
    return extract_user_text(state)


def recent_messages_from_state(state: AgentState) -> list[dict[str, Any]]:
    context_bundle = context_bundle_from_state(state)
    raw_messages = context_bundle.get("recent_messages")
    if not isinstance(raw_messages, list):
        raw_messages = []

    memory_context = state.get("memory_context") or {}
    memory_turns = memory_context.get("recent_turns")
    if not isinstance(memory_turns, list):
        memory_turns = []

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*memory_turns, *raw_messages]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in {"user", "assistant", "system", "tool"}:
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        key = str(item.get("message_id") or f"{role}:{content[:80]}")
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(item))
    return result[-12:]


def active_report_from_state(state: AgentState) -> dict[str, Any] | None:
    context_bundle = context_bundle_from_state(state)
    active_report = context_bundle.get("active_report")
    if isinstance(active_report, dict) and active_report:
        return active_report

    client_context = state.get("client_context") or {}
    client_extra = client_context.get("extra") or {}
    latest_report = (
        client_extra.get("latest_report")
        or client_extra.get("latest_report_context")
        or client_extra.get("frontend_latest_report")
    )
    return latest_report if isinstance(latest_report, dict) and latest_report else None


def conversation_summary_from_state(state: AgentState) -> dict[str, Any] | str | None:
    context_bundle = context_bundle_from_state(state)
    summary = context_bundle.get("conversation_summary")
    if summary:
        return summary

    memory_context = state.get("memory_context") or {}
    summary_text = memory_context.get("conversation_summary")
    return summary_text if summary_text else None


def traceback_context_from_state(state: AgentState) -> dict[str, Any]:
    context_bundle = context_bundle_from_state(state)
    traceback_context = context_bundle.get("traceback_context") or {}
    return traceback_context if isinstance(traceback_context, dict) else {}


def _message_metadata(message: dict[str, Any]) -> dict[str, Any]:
    metadata = message.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    metadata_json = message.get("metadata_json")
    return metadata_json if isinstance(metadata_json, dict) else {}


def _assistant_answer_context(message: dict[str, Any]) -> dict[str, Any] | None:
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return None

    metadata = _message_metadata(message)
    structured_content = (
        metadata.get("structured_content")
        or message.get("structured_content")
        or {}
    )
    if not isinstance(structured_content, dict):
        structured_content = {}

    return {
        "message_id": message.get("message_id"),
        "role": "assistant",
        "content": content.strip(),
        "content_type": message.get("content_type") or "text",
        "report_id": message.get("report_id") or metadata.get("report_id"),
        "node_name": message.get("node_name") or metadata.get("node_name"),
        "route_target": message.get("route_target") or metadata.get("route_target"),
        "answer_type": message.get("answer_type") or metadata.get("answer_type"),
        "rag_query": message.get("rag_query") or metadata.get("rag_query"),
        "structured_content": structured_content,
        "metadata": metadata,
        "created_at": message.get("created_at"),
    }


def last_final_answer_from_state(state: AgentState) -> dict[str, Any] | None:
    context_bundle = context_bundle_from_state(state)
    bundled = context_bundle.get("last_final_answer")
    if isinstance(bundled, dict) and bundled.get("content"):
        return _assistant_answer_context(bundled)

    for message in reversed(recent_messages_from_state(state)):
        if message.get("role") != "assistant":
            continue
        answer_context = _assistant_answer_context(message)
        if answer_context:
            return answer_context
    return None


def related_long_term_memories_from_state(state: AgentState) -> list[dict[str, Any]]:
    memory_context = state.get("memory_context") or {}
    memories = memory_context.get("relevant_memories") or memory_context.get("memories") or []
    summaries = memory_context.get("summaries") or []
    profile = memory_context.get("profile") or {}

    result: list[dict[str, Any]] = []
    for item in memories:
        if isinstance(item, dict):
            result.append({"source": "long_term_memory", **item})
    for item in summaries:
        if isinstance(item, dict):
            result.append({"source": "memory_summary", **item})
    if isinstance(profile, dict) and profile:
        result.append({"source": "profile", **profile})
    return result[:8]


# 从agentState中提取模型调用需要的上下文，做装成结构化的字典
# TODO 字典类型不稳定，建议改成Pydantic
def build_prompt_context(
    state: AgentState,
    *,
    mode: ContextMode,
    system_prompt: str,
    node_name: str | None = None,
    include_long_term_memory: bool = False,
) -> dict[str, Any]:
    # 从state里面提取信息
    query_context = query_context_from_state(state)
    prompt_context = {
        "schema_version": "1.0",
        "mode": mode,
        "node_name": node_name,
        "system_prompt": system_prompt,
        "raw_user_input": extract_user_text(state),
        "standalone_query": query_context.get("standalone_query")
        or extract_user_text(state),
        "reference_resolution": query_context.get("reference_resolution") or {},
        "recent_messages": recent_messages_from_state(state),
        "conversation_summary": conversation_summary_from_state(state),
        "last_final_answer": last_final_answer_from_state(state),
        "active_report": active_report_from_state(state),
        "traceback_context": traceback_context_from_state(state),
        "related_long_term_memories": [],
    }

    # 需要长期记忆并且需要构造完整上下文
    if include_long_term_memory and mode == "FULL_FOR_NODE":
        prompt_context["related_long_term_memories"] = (
            related_long_term_memories_from_state(state)
        )

    return prompt_context


def with_prompt_context(
    state: AgentState,
    *,
    mode: ContextMode,
    system_prompt: str,
    node_name: str | None = None,
    include_long_term_memory: bool = False,
) -> AgentState:
    prompt_context = build_prompt_context(
        state,
        mode=mode,
        system_prompt=system_prompt,
        node_name=node_name,
        include_long_term_memory=include_long_term_memory,
    )
    return {
        **state,
        "prompt_context": prompt_context,
    }
