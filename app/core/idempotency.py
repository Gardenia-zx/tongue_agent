import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from redis.asyncio import Redis



# 枚举类：枚举当前请求状态
class IdempotencyStatus(StrEnum):
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class IdempotencyConflictError(RuntimeError):
    pass


class IdempotencyProcessingError(RuntimeError):
    pass

# redis中读取的数据记录
# frozen=true,对象不可修改
@dataclass(frozen=True)
class IdempotencyRecord:
    status: IdempotencyStatus
    # 获取sha-256摘要，判断对话内容和request_id是否匹配
    request_hash: str
    response: dict[str, Any] | None


# 生成sha-256摘要
def make_request_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _record_key(scope: str, request_id: str) -> str:
    return f"idempotency:{scope}:{request_id}"

# 异步任务，开始幂等请求
async def start_idempotent_request(
    redis: Redis,
    *,
    scope: str,
    request_id: str,
    request_hash: str,
    ttl_seconds: int,
) -> IdempotencyRecord | None:
    key = _record_key(scope, request_id)
    value = {
        "status": IdempotencyStatus.PROCESSING,
        "request_hash": request_hash,
        "response": None,
    }

    created = await redis.set(
        key,
        json.dumps(value, ensure_ascii=False),
        nx=True,
        ex=ttl_seconds,
    )

    if created:
        return None

    raw = await redis.get(key)
    if raw is None:
        return None

    record_data = json.loads(raw)
    record = IdempotencyRecord(
        status=IdempotencyStatus(record_data["status"]),
        request_hash=record_data["request_hash"],
        response=record_data.get("response"),
    )

    if record.request_hash != request_hash:
        raise IdempotencyConflictError(
            f"Same request_id used with different payload: {request_id}"
        )

    if record.status == IdempotencyStatus.PROCESSING:
        raise IdempotencyProcessingError(
            f"Request is still processing: {request_id}"
        )

    return record


async def finish_idempotent_request(
    redis: Redis,
    *,
    scope: str,
    request_id: str,
    request_hash: str,
    response: dict[str, Any],
    ttl_seconds: int,
) -> None:
    key = _record_key(scope, request_id)
    value = {
        "status": IdempotencyStatus.SUCCESS,
        "request_hash": request_hash,
        "response": response,
    }
    await redis.set(
        key,
        json.dumps(value, ensure_ascii=False),
        ex=ttl_seconds,
    )


async def fail_idempotent_request(
    redis: Redis,
    *,
    scope: str,
    request_id: str,
    request_hash: str,
    response: dict[str, Any] | None,
    ttl_seconds: int,
) -> None:
    key = _record_key(scope, request_id)
    value = {
        "status": IdempotencyStatus.FAILED,
        "request_hash": request_hash,
        "response": response,
    }
    await redis.set(
        key,
        json.dumps(value, ensure_ascii=False),
        ex=ttl_seconds,
    )