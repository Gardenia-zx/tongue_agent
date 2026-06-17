import secrets
from contextlib import asynccontextmanager
from typing import AsyncIterator

from redis.asyncio import Redis


class LockBusyError(RuntimeError):
    pass


_RELEASE_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


@asynccontextmanager
async def redis_lock(
    redis: Redis,
    key: str,
    ttl_seconds: int,
) -> AsyncIterator[None]:
    token = secrets.token_urlsafe(32)
    acquired = await redis.set(
        key,
        token,
        nx=True,
        ex=ttl_seconds,
    )

    if not acquired:
        raise LockBusyError(f"Lock is busy: {key}")

    try:
        yield
    finally:
        await redis.eval(
            _RELEASE_LOCK_SCRIPT,
            1,
            key,
            token,
        )