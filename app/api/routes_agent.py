from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from app.core.config import get_settings
from app.core.locks import LockBusyError, redis_lock
from app.integrations.redis_client import create_redis_client
from app.schemas.agent import AgentRunRequest, AgentRunResponse


router = APIRouter(prefix="/agent", tags=["agent"])


def _initial_state(request: AgentRunRequest) -> dict[str, Any]:
    return {
        "schema_version": request.schema_version,
        "trace_id": request.trace_id,
        "request_id": request.request_id,
        "user_id": request.user_id,
        "thread_id": request.thread_id,
        "conversation_id": request.conversation_id,
        "report_id": request.report_id,
        "task_id": request.task_id,
        "task_version": request.task_version,
        "message": request.message.model_dump(mode="json"),
        "client_context": request.client_context.model_dump(mode="json"),
        "options": request.options,
        "errors": [],
        "extensions": {},
    }


def _response_status(final_state: dict[str, Any]) -> str:
    next_action = final_state.get("next_action") or {}
    payload = next_action.get("payload") or {}
    status_value = payload.get("status")

    if status_value in {
        "COMPLETED",
        "WAIT_USER_ANSWER",
        "CLARIFY",
        "REJECTED",
        "FAILED",
    }:
        return status_value

    return "COMPLETED"


def _state_snapshot(final_state: dict[str, Any]) -> dict[str, Any]:
    memory_context = final_state.get("memory_context") or {}
    tongue_features = final_state.get("tongue_features") or {}
    rag_context = final_state.get("rag_context") or {}
    draft_report = final_state.get("draft_report") or {}
    return {
        "current_node": final_state.get("current_node"),
        "intent_result": final_state.get("intent_result"),
        "safety_result": final_state.get("safety_result"),
        "next_action": final_state.get("next_action"),
        "memory": {
            "used_memory_ids": memory_context.get("used_memory_ids") or [],
            "policy": memory_context.get("policy") or {},
            "write_result": memory_context.get("write_result"),
            "short_term_write_result": memory_context.get("short_term_write_result"),
            "session": {
                "cache_hit": (memory_context.get("session") or {}).get("cache_hit"),
                "turn_count": (memory_context.get("session") or {}).get("turn_count"),
            },
        },
        "tongue_analysis": {
            "detected_feature_codes": tongue_features.get("detected_feature_codes")
            or [],
            "rag_query": tongue_features.get("rag_query") or "",
            "rag_grounded": rag_context.get("grounded"),
            "rag_hit_count": len(rag_context.get("hits") or []),
            "has_draft_report": bool(final_state.get("draft_report")),
            "report_status": draft_report.get("report_status"),
            "rag_evidence_count": len(draft_report.get("rag_evidence") or []),
        },
    }


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(
    http_request: Request,
    request: AgentRunRequest,
) -> AgentRunResponse:
    settings = get_settings()
    redis = create_redis_client()
    lock_key = f"lock:agent:thread:{request.thread_id}"

    try:
        async with redis_lock(
            redis,
            key=lock_key,
            ttl_seconds=settings.agent_lock_ttl_seconds,
        ):
            graph = http_request.app.state.agent_graph
            config = {
                "configurable": {
                    "thread_id": request.thread_id,
                    "user_id": request.user_id,
                    "langgraph_user_id": str(request.user_id),
                }
            }
            final_state = await graph.ainvoke(_initial_state(request), config=config)
    except LockBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "AGENT_THREAD_BUSY",
                "message": "同一个 Agent 会话正在处理中，请稍后重试。",
                "thread_id": request.thread_id,
            },
        ) from exc
    finally:
        await redis.close()

    return AgentRunResponse(
        request_id=request.request_id,
        trace_id=request.trace_id,
        thread_id=request.thread_id,
        conversation_id=request.conversation_id,
        report_id=request.report_id,
        task_id=request.task_id,
        status=_response_status(final_state),
        intent_result=final_state.get("intent_result"),
        message=final_state.get("response_message"),
        next_action=final_state.get("next_action"),
        state_snapshot=_state_snapshot(final_state),
    )
