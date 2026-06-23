import json
from datetime import UTC, datetime
from typing import Any

from redis.asyncio import Redis

from app.agent.state import AgentState
from app.core.config import get_settings
from app.memory.policy import extract_user_text


SESSION_MEMORY_KEY_PREFIX = "agent:session"


class ShortTermMemoryCache:
    def __init__(
        self,
        *,
        redis: Redis,
        ttl_seconds: int | None = None,
        keep_last_turns: int | None = None,
        summary_turn_threshold: int | None = None,
        summary_char_threshold: int | None = None,
    ) -> None:
        settings = get_settings()
        self.redis = redis
        self.ttl_seconds = ttl_seconds or settings.short_term_memory_ttl_seconds
        self.keep_last_turns = keep_last_turns or settings.short_term_memory_keep_last_turns
        self.summary_turn_threshold = (
            summary_turn_threshold or settings.short_term_memory_summary_turn_threshold
        )
        self.summary_char_threshold = (
            summary_char_threshold or settings.short_term_memory_summary_char_threshold
        )

    async def load_context(self, thread_id: str | None) -> dict[str, Any]:
        if not thread_id:
            return self._empty_context(cache_hit=False, skip_reason="missing_thread_id")

        payload = await self.redis.get(self._key(thread_id))
        if not payload:
            return self._empty_context(cache_hit=False)

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return self._empty_context(cache_hit=False, skip_reason="invalid_cache_json")

        turns = data.get("recent_turns")
        if not isinstance(turns, list):
            turns = []

        return {
            "cache_hit": True,
            "source": "redis",
            "thread_id": thread_id,
            "conversation_summary": data.get("conversation_summary") or "",
            "recent_turns": turns[-self.keep_last_turns :],
            "turn_count": int(data.get("turn_count") or len(turns)),
            "updated_at": data.get("updated_at"),
            "skip_reason": None,
        }

    async def append_turn(self, state: AgentState) -> dict[str, Any]:
        thread_id = state.get("thread_id")
        if not thread_id:
            return {"status": "SKIPPED", "reason": "missing_thread_id"}

        context = await self.load_context(thread_id)
        recent_turns = list(context.get("recent_turns") or [])
        turn = self._build_turn(state)
        if turn is None:
            return {"status": "SKIPPED", "reason": "empty_turn"}

        recent_turns.append(turn)
        turn_count = int(context.get("turn_count") or 0) + 1
        conversation_summary = context.get("conversation_summary") or ""

        should_compress = self._should_compress(
            recent_turns=recent_turns,
            turn_count=turn_count,
        )
        compressed = False
        if should_compress and len(recent_turns) > self.keep_last_turns:
            old_turns = recent_turns[: -self.keep_last_turns]
            recent_turns = recent_turns[-self.keep_last_turns :]
            conversation_summary = self._compress_turns(
                previous_summary=conversation_summary,
                turns=old_turns,
            )
            compressed = True

        payload = {
            "schema_version": "1.0",
            "thread_id": thread_id,
            "conversation_summary": conversation_summary,
            "recent_turns": recent_turns[-self.keep_last_turns :],
            "turn_count": turn_count,
            "updated_at": self._now(),
        }
        await self.redis.set(
            self._key(thread_id),
            json.dumps(payload, ensure_ascii=False),
            ex=self.ttl_seconds,
        )

        return {
            "status": "UPDATED",
            "source": "redis",
            "thread_id": thread_id,
            "turn_count": turn_count,
            "recent_turn_count": len(payload["recent_turns"]),
            "compressed": compressed,
        }

    @staticmethod
    def _key(thread_id: str) -> str:
        return f"{SESSION_MEMORY_KEY_PREFIX}:{thread_id}:state"

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _build_turn(state: AgentState) -> dict[str, Any] | None:
        user_text = extract_user_text(state)
        response_message = state.get("response_message") or {}
        assistant_text = response_message.get("content")

        if not user_text and not assistant_text:
            return None

        return {
            "request_id": state.get("request_id"),
            "node": state.get("current_node"),
            "user": user_text,
            "assistant": assistant_text if isinstance(assistant_text, str) else "",
            "created_at": ShortTermMemoryCache._now(),
        }

    def _should_compress(
        self,
        *,
        recent_turns: list[dict[str, Any]],
        turn_count: int,
    ) -> bool:
        if turn_count >= self.summary_turn_threshold:
            return True

        text_size = sum(
            len(str(turn.get("user") or "")) + len(str(turn.get("assistant") or ""))
            for turn in recent_turns
        )
        return text_size >= self.summary_char_threshold

    @staticmethod
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

    @staticmethod
    def _empty_context(
        *,
        cache_hit: bool,
        skip_reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "cache_hit": cache_hit,
            "source": "redis",
            "conversation_summary": "",
            "recent_turns": [],
            "turn_count": 0,
            "updated_at": None,
            "skip_reason": skip_reason,
        }
