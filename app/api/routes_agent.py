import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status

from app.core.config import get_settings
from app.core.locks import LockBusyError, redis_lock
from app.core.turn_hash import canonical_agent_request_hash, response_hash
from app.integrations.redis_client import create_redis_client
from app.integrations.turn_records import AgentTurnRecord, TurnStatus
from app.schemas.agent import (
    AgentRunRequest,
    AgentRunResponse,
    AgentTurnAckRequest,
    AgentTurnAckResponse,
)


router = APIRouter(prefix="/agent", tags=["agent"])


def _initial_state(request: AgentRunRequest) -> dict[str, Any]:
    tenant_id = _tenant_id(request)
    turn_id = _turn_id(request)
    return {
        "schema_version": request.schema_version,
        "trace_id": request.trace_id,
        "request_id": request.request_id,
        "tenant_id": tenant_id,
        "turn_id": turn_id,
        "user_message_id": request.user_message_id,
        "assistant_message_id": request.assistant_message_id,
        "user_id": request.user_id,
        "thread_id": request.thread_id,
        "thread_epoch": request.thread_epoch,
        "conversation_id": request.conversation_id,
        "report_id": request.report_id,
        "task_id": request.task_id,
        "task_version": request.task_version,
        "message": request.message.model_dump(mode="json"),
        "client_context": request.client_context.model_dump(mode="json"),
        "context_bundle": request.context_bundle,
        "options": request.options,
        "errors": [],
        "extensions": {},
    }


def _tenant_id(request: AgentRunRequest) -> str:
    return str(request.tenant_id or request.user_id)


def _turn_id(request: AgentRunRequest) -> str:
    if request.turn_id:
        return request.turn_id
    if request.conversation_id and request.user_message_id and request.assistant_message_id:
        return (
            f"{request.conversation_id}:"
            f"{request.user_message_id}:"
            f"{request.assistant_message_id}"
        )
    return f"{request.thread_id}:{request.request_id}"


def _checkpoint_thread_id(request: AgentRunRequest) -> str:
    tenant_id = _tenant_id(request)
    return f"{tenant_id}:{request.thread_id}:epoch:{request.thread_epoch}"


def _graph_config(request: AgentRunRequest) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": _checkpoint_thread_id(request),
            "user_id": request.user_id,
            "tenant_id": _tenant_id(request),
            "thread_epoch": request.thread_epoch,
            "langgraph_user_id": str(request.user_id),
        }
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
    agent_loop = final_state.get("agent_loop") or {}
    return {
        "current_node": final_state.get("current_node"),
        "turn": {
            "tenant_id": final_state.get("tenant_id"),
            "turn_id": final_state.get("turn_id"),
            "thread_epoch": final_state.get("thread_epoch"),
            "user_message_id": final_state.get("user_message_id"),
            "assistant_message_id": final_state.get("assistant_message_id"),
        },
        "intent_result": final_state.get("intent_result"),
        "safety_result": final_state.get("safety_result"),
        "next_action": final_state.get("next_action"),
        "agent_loop": {
            "mode": agent_loop.get("mode"),
            "status": agent_loop.get("status"),
            "selected_tool": agent_loop.get("selected_tool"),
            "finish_reason": agent_loop.get("finish_reason"),
            "tool_calls": agent_loop.get("tool_calls") or [],
        },
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
        "context_bundle": {
            "mode": (final_state.get("context_bundle") or {}).get("mode"),
            "conversation_id": (final_state.get("context_bundle") or {}).get("conversation_id"),
            "has_active_report": bool((final_state.get("context_bundle") or {}).get("active_report")),
            "recent_message_count": len((final_state.get("context_bundle") or {}).get("recent_messages") or []),
            "has_summary": bool((final_state.get("context_bundle") or {}).get("conversation_summary")),
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


def _build_agent_response(
    request: AgentRunRequest,
    final_state: dict[str, Any],
    *,
    response_hash_value: str | None = None,
    response_ref: dict[str, Any] | None = None,
) -> AgentRunResponse:
    return AgentRunResponse(
        request_id=request.request_id,
        trace_id=request.trace_id,
        tenant_id=_tenant_id(request),
        turn_id=_turn_id(request),
        response_hash=response_hash_value,
        response_ref=response_ref,
        thread_id=request.thread_id,
        thread_epoch=request.thread_epoch,
        conversation_id=request.conversation_id,
        report_id=request.report_id,
        task_id=request.task_id,
        status=_response_status(final_state),
        intent_result=final_state.get("intent_result"),
        message=final_state.get("response_message"),
        next_action=final_state.get("next_action"),
        state_snapshot=_state_snapshot(final_state),
    )


def _build_response_ref_response(
    request: AgentRunRequest,
    record: AgentTurnRecord,
) -> AgentRunResponse:
    return AgentRunResponse(
        request_id=request.request_id,
        trace_id=request.trace_id,
        tenant_id=_tenant_id(request),
        turn_id=_turn_id(request),
        response_hash=record.response_hash,
        response_ref=record.response_ref,
        thread_id=request.thread_id,
        thread_epoch=request.thread_epoch,
        conversation_id=request.conversation_id,
        report_id=request.report_id,
        task_id=request.task_id,
        status="COMPLETED",
        message=None,
        next_action=None,
        state_snapshot={
            "idempotency": {
                "source": "agent_turn_record",
                "status": record.status,
                "response_ref": record.response_ref,
            }
        },
    )


def _cache_key(tenant_id: str, turn_id: str) -> str:
    return f"agent:turn:{tenant_id}:{turn_id}"


async def _cache_completed_turn(redis, record: AgentTurnRecord) -> None:
    await redis.set(
        _cache_key(record.tenant_id, record.turn_id),
        json.dumps(
            {
                "status": record.status,
                "canonical_request_hash": record.canonical_request_hash,
                "response_hash": record.response_hash,
                "response_ref": record.response_ref,
            },
            ensure_ascii=False,
        ),
        ex=get_settings().idempotency_ttl_seconds,
    )


async def _read_turn_cache(redis, *, tenant_id: str, turn_id: str) -> dict[str, Any] | None:
    raw = await redis.get(_cache_key(tenant_id, turn_id))
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _http_conflict(code: str, message: str, **detail: Any) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "code": code,
            "message": message,
            **detail,
        },
    )


def _validate_record_hash(
    *,
    record: AgentTurnRecord,
    canonical_hash: str,
) -> None:
    if record.canonical_request_hash != canonical_hash:
        raise _http_conflict(
            "TURN_ID_CONFLICT",
            "同一个 turn_id 不能用于不同请求内容。",
            turn_id=record.turn_id,
        )


def _record_response(
    *,
    request: AgentRunRequest,
    record: AgentTurnRecord,
    turn_store,
    canonical_hash: str,
) -> AgentRunResponse | None:
    _validate_record_hash(record=record, canonical_hash=canonical_hash)
    if record.status == TurnStatus.COMPLETED_PENDING_ACK:
        snapshot = turn_store.decrypt_response_snapshot(record)
        if snapshot is None:
            return None
        response = AgentRunResponse.model_validate(snapshot)
        response.request_id = request.request_id
        response.trace_id = request.trace_id
        return response

    if record.status == TurnStatus.ACKED:
        return _build_response_ref_response(request, record)

    return None


async def _recover_completed_from_checkpoint(
    *,
    graph,
    request: AgentRunRequest,
) -> AgentRunResponse | None:
    try:
        snapshot = await graph.aget_state(_graph_config(request))
    except Exception:
        return None

    values = getattr(snapshot, "values", None)
    if not isinstance(values, dict):
        return None

    response_message = values.get("response_message")
    if not isinstance(response_message, dict) or not response_message.get("content"):
        return None

    return _build_agent_response(request, values)


async def _validate_or_advance_epoch(
    *,
    request: AgentRunRequest,
    turn_store,
    redis,
) -> None:
    tenant_id = _tenant_id(request)
    current_epoch = await turn_store.ensure_thread_epoch(
        tenant_id=tenant_id,
        thread_id=request.thread_id,
        thread_epoch=request.thread_epoch,
    )

    if request.thread_epoch < current_epoch:
        raise _http_conflict(
            "STALE_THREAD_EPOCH",
            "请求使用了旧会话 Epoch。",
            thread_id=request.thread_id,
            request_epoch=request.thread_epoch,
            current_epoch=current_epoch,
        )

    if request.thread_epoch == current_epoch:
        return

    if not request.reset_reason:
        raise _http_conflict(
            "FUTURE_THREAD_EPOCH",
            "只有重置会话流程可以递增 Epoch。",
            thread_id=request.thread_id,
            request_epoch=request.thread_epoch,
            current_epoch=current_epoch,
        )

    session_lock_key = f"lock:agent:session:{tenant_id}:{request.thread_id}"
    async with redis_lock(
        redis,
        key=session_lock_key,
        ttl_seconds=get_settings().agent_lock_ttl_seconds,
    ):
        active_count = await turn_store.active_turn_count(
            tenant_id=tenant_id,
            thread_id=request.thread_id,
            exclude_turn_id=_turn_id(request),
        )
        if active_count > 0:
            raise _http_conflict(
                "THREAD_RESET_BUSY",
                "旧 Epoch 仍有执行中的 Turn，暂不能重置会话。",
                thread_id=request.thread_id,
            )
        await turn_store.advance_thread_epoch(
            tenant_id=tenant_id,
            thread_id=request.thread_id,
            new_epoch=request.thread_epoch,
            reset_reason=request.reset_reason,
        )


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(
    http_request: Request,
    request: AgentRunRequest,
) -> AgentRunResponse:
    settings = get_settings()
    tenant_id = _tenant_id(request)
    turn_id = _turn_id(request)
    canonical_hash = canonical_agent_request_hash(
        request,
        tenant_id=tenant_id,
        turn_id=turn_id,
    )

    if request.request_hash and request.request_hash != canonical_hash:
        raise _http_conflict(
            "REQUEST_HASH_MISMATCH",
            "Python 重新计算的请求哈希与调用方传入值不一致。",
            turn_id=turn_id,
        )

    turn_store = http_request.app.state.agent_turn_record_store
    redis = create_redis_client()
    owner_id = f"{settings.app_name}:{uuid4().hex}"
    lock_key = f"lock:agent:{tenant_id}:thread:{request.thread_id}:epoch:{request.thread_epoch}"

    try:
        cached = await _read_turn_cache(
            redis,
            tenant_id=tenant_id,
            turn_id=turn_id,
        )
        if cached and cached.get("canonical_request_hash") == canonical_hash:
            cached_record = await turn_store.get(
                tenant_id=tenant_id,
                turn_id=turn_id,
            )
            if cached_record is not None:
                replay = _record_response(
                    request=request,
                    record=cached_record,
                    turn_store=turn_store,
                    canonical_hash=canonical_hash,
                )
                if replay is not None:
                    return replay

        record = await turn_store.ensure_created(
            tenant_id=tenant_id,
            turn_id=turn_id,
            canonical_request_hash=canonical_hash,
            thread_id=request.thread_id,
            thread_epoch=request.thread_epoch,
        )

        replay = _record_response(
            request=request,
            record=record,
            turn_store=turn_store,
            canonical_hash=canonical_hash,
        )
        if replay is not None:
            await _cache_completed_turn(redis, record)
            return replay

        if record.status == TurnStatus.PROCESSING and record.lease_active():
            raise _http_conflict(
                "TURN_PROCESSING",
                "同一个 Turn 正在处理中，请稍后重试。",
                turn_id=turn_id,
                lease_until=record.lease_until.isoformat() if record.lease_until else None,
            )

        if record.status == TurnStatus.FAILED_FINAL:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "code": "TURN_FAILED_FINAL",
                    "message": "该 Turn 已失败且不可重试。",
                    "turn_id": turn_id,
                    "reason": record.failure_reason,
                },
            )

        try:
            await _validate_or_advance_epoch(
                request=request,
                turn_store=turn_store,
                redis=redis,
            )
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            code = str(detail.get("code") or "EPOCH_VALIDATION_FAILED")
            await turn_store.mark_failed(
                tenant_id=tenant_id,
                turn_id=turn_id,
                retryable=code == "THREAD_RESET_BUSY",
                reason=code,
            )
            raise

        async with redis_lock(
            redis,
            key=lock_key,
            ttl_seconds=settings.agent_lock_ttl_seconds,
        ):
            locked_record = await turn_store.get(
                tenant_id=tenant_id,
                turn_id=turn_id,
            )
            if locked_record is None:
                raise RuntimeError("agent_turn_record disappeared")

            locked_replay = _record_response(
                request=request,
                record=locked_record,
                turn_store=turn_store,
                canonical_hash=canonical_hash,
            )
            if locked_replay is not None:
                await _cache_completed_turn(redis, locked_record)
                return locked_replay

            graph = http_request.app.state.agent_graph
            if locked_record.status == TurnStatus.PROCESSING:
                recovered = await _recover_completed_from_checkpoint(
                    graph=graph,
                    request=request,
                )
                if recovered is not None:
                    payload = recovered.model_dump(
                        mode="json",
                        exclude={"response_hash"},
                    )
                    recovered.response_hash = response_hash(payload)
                    completed_record = await turn_store.mark_completed_pending_ack(
                        tenant_id=tenant_id,
                        turn_id=turn_id,
                        response_snapshot=recovered.model_dump(mode="json"),
                        response_hash=recovered.response_hash,
                    )
                    await _cache_completed_turn(redis, completed_record)
                    return recovered

            await turn_store.mark_processing(
                tenant_id=tenant_id,
                turn_id=turn_id,
                owner_id=owner_id,
                lease_seconds=settings.agent_turn_lease_seconds,
            )

            try:
                final_state = await graph.ainvoke(
                    _initial_state(request),
                    config=_graph_config(request),
                )
            except Exception as exc:
                await turn_store.mark_failed(
                    tenant_id=tenant_id,
                    turn_id=turn_id,
                    retryable=True,
                    reason=f"{type(exc).__name__}:{exc}",
                )
                raise

            response = _build_agent_response(request, final_state)
            payload = response.model_dump(mode="json", exclude={"response_hash"})
            response.response_hash = response_hash(payload)
            completed_record = await turn_store.mark_completed_pending_ack(
                tenant_id=tenant_id,
                turn_id=turn_id,
                response_snapshot=response.model_dump(mode="json"),
                response_hash=response.response_hash,
            )
            await _cache_completed_turn(redis, completed_record)
            return response
    except LockBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "AGENT_THREAD_BUSY",
                "message": "同一个 Agent 会话正在处理中，请稍后重试。",
                "thread_id": request.thread_id,
                "thread_epoch": request.thread_epoch,
            },
        ) from exc
    finally:
        await redis.close()


@router.post("/turns/ack", response_model=AgentTurnAckResponse)
async def ack_agent_turn(
    http_request: Request,
    request: AgentTurnAckRequest,
) -> AgentTurnAckResponse:
    turn_store = http_request.app.state.agent_turn_record_store
    try:
        record = await turn_store.ack(
            tenant_id=request.tenant_id,
            turn_id=request.turn_id,
            assistant_message_id=request.assistant_message_id,
            response_hash=request.response_hash,
        )
    except ValueError as exc:
        raise _http_conflict(
            "TURN_ACK_REJECTED",
            "Turn ACK 校验失败。",
            reason=str(exc),
            turn_id=request.turn_id,
        ) from exc

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "TURN_NOT_FOUND",
                "message": "未找到需要 ACK 的 Turn。",
                "turn_id": request.turn_id,
            },
        )

    redis = create_redis_client()
    try:
        await _cache_completed_turn(redis, record)
    finally:
        await redis.close()

    return AgentTurnAckResponse(
        status="ACKED",
        tenant_id=request.tenant_id,
        turn_id=request.turn_id,
        assistant_message_id=request.assistant_message_id,
        response_hash=request.response_hash,
    )


@router.get("/turns/pending-ack")
async def list_pending_agent_turn_ack(
    http_request: Request,
    older_than_seconds: int | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    settings = get_settings()
    turn_store = http_request.app.state.agent_turn_record_store
    records = await turn_store.list_pending_ack(
        older_than_seconds=older_than_seconds
        or settings.agent_unacked_warn_after_seconds,
        limit=max(1, min(limit, 500)),
    )
    return {
        "status": "OK",
        "count": len(records),
        "records": [
            {
                "tenant_id": record.tenant_id,
                "turn_id": record.turn_id,
                "thread_id": record.thread_id,
                "thread_epoch": record.thread_epoch,
                "status": record.status,
                "java_ack_status": record.java_ack_status,
                "completed_at": record.completed_at.isoformat()
                if record.completed_at
                else None,
                "response_hash": record.response_hash,
                "retry_count": record.retry_count,
            }
            for record in records
        ],
    }
