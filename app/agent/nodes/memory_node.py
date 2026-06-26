from typing import Any

from app.agent.state import AgentState
from app.integrations.redis_client import create_redis_client
from app.memory.service import MemoryService
from app.memory.session_cache import ShortTermMemoryCache

# 读取记忆节点，参数：state:当前Langgraph的状态，store：Langgraph编译时注入的长期记忆，使用时必须通过名称传入
async def memory_read_node(state: AgentState, *, store) -> AgentState:
    # 判断是否为无状态上下文
    # 有状态：agent自己从redis,数据库读取
    # 无状态：java后端自己传递过来的上下文
    if _is_stateless_context(state):
        # 如果为无状态上下文
        # 将java后端传递过来的上下文结构化
        memory_context = _stateless_memory_context(state)
        # 判断是否需要长期记忆
        if _can_read_long_term_memory(state):
            # 从state里面提取后端上下文
            session_context = _backend_session_context(state)
            # 构造长期记忆上下文
            long_term_context = await MemoryService(store=store).build_memory_context(
                state,
                session_context=session_context,
            )
            # 合并长期记忆
            memory_context.update(
                {
                    "used_memory_ids": long_term_context.get("used_memory_ids") or [],
                    "relevant_memories": long_term_context.get("relevant_memories") or [],
                    "memories": long_term_context.get("memories") or [],
                    "summaries": long_term_context.get("summaries") or [],
                    "profile": long_term_context.get("profile") or {},
                    "policy": long_term_context.get("policy") or memory_context.get("policy"),
                    "context_summary": long_term_context.get("context_summary")
                    or memory_context.get("context_summary"),
                }
            )
            # 返回更新后的状态
        return {
            **state,
            "current_node": "memory_read_node",
            "memory_context": memory_context,
        }
    
    # 如果是有状态的
    # 加载对话内上下文
    session_context = await _load_session_context(state)
    # 提取长期记忆
    memory_context = await MemoryService(store=store).build_memory_context(
        state,
        session_context=session_context,
    )

    return {
        **state,
        "current_node": "memory_read_node",
        "memory_context": memory_context,
    }

# 保存短期记忆，提取并写入长期记忆
async def memory_commit_node(state: AgentState, *, store) -> AgentState:
    # 判断是否有状态
    if _is_stateless_context(state):
        # 如果是无状态
        # 复制state里面的memory_context字典
        memory_context = dict(state.get("memory_context") or {})
        # 跳过长期记忆
        memory_context["write_result"] = {
            "status": "SKIPPED",
            "reason": "stateless_context_managed_by_backend",
        }
        # 跳过短期记忆写入
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
    # 当前对话写入redis
    short_term_result = await _append_short_term_turn(state)
    # 判断当前对话有没有需要保存到长期记忆的
    long_term_result = await MemoryService(store=store).commit_after_response(state)

    memory_context = dict(state.get("memory_context") or {})
    memory_context["write_result"] = long_term_result
    memory_context["short_term_write_result"] = short_term_result

    # TODO extensions是干什么的
    extensions = dict(state.get("extensions") or {})
    extensions["last_internal_node"] = "memory_commit_node"

    return {
        **state,
        "current_node": state.get("current_node"),
        "memory_context": memory_context,
        "extensions": extensions,
    }

# 加载短期对话记忆
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

# 判断上下文是否有状态
def _is_stateless_context(state: AgentState) -> bool:

    context_bundle = state.get("context_bundle") or {}
    if context_bundle:
        return True

    options = state.get("options") or {}
    context_options = options.get("context") or {}
    return isinstance(context_options, dict) and context_options.get("mode") == "stateless"

# 将java传递的上下文结构化
def _stateless_memory_context(state: AgentState) -> dict[str, Any]:
    session_context = _backend_session_context(state)
    recent_messages = session_context.get("recent_turns") or []
    summary_text = session_context.get("conversation_summary") or ""
    return {
        "used_memory_ids": [],
        "relevant_memories": [],
        "memories": [],
        "summaries": [],
        "profile": {},
        "session": {
            "cache_hit": False,
            "source": "backend_context_bundle",
            "turn_count": len(recent_messages) if isinstance(recent_messages, list) else 0,
        },
        "conversation_summary": summary_text,
        "recent_turns": recent_messages if isinstance(recent_messages, list) else [],
        "write_result": None,
        "policy": {
            "can_read": False,
            "can_write": False,
            "skip_reason": "stateless_context_managed_by_backend",
        },
        "context_summary": "已使用 Java 后端组装的无状态上下文包。",
    }


def _backend_session_context(state: AgentState) -> dict[str, Any]:
    context_bundle = state.get("context_bundle") or {}
    conversation_summary = context_bundle.get("conversation_summary") or {}
    if isinstance(conversation_summary, dict):
        summary_text = conversation_summary.get("text") or ""
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
        # TODO 失败时需要被日志和监控系统捕获，避免上下文丢失
        return {
            "status": "SKIPPED",
            "reason": f"redis_unavailable:{type(exc).__name__}",
        }
    finally:
        await redis.close()
