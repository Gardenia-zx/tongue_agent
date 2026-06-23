import unittest

from langgraph.store.memory import InMemoryStore

from app.memory.service import (
    ACTIVE_STATUS,
    MEMORY_NAMESPACE,
    PROFILE_NAMESPACE,
    SUPERSEDED_STATUS,
    MemoryService,
)
from app.memory.session_cache import ShortTermMemoryCache


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.ttl: dict[str, int] = {}

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.data[key] = value
        if ex is not None:
            self.ttl[key] = ex


def _state(
    text: str,
    *,
    user_id: int = 1,
    thread_id: str = "memory_thread_001",
    can_write: bool = True,
    risk_level: str = "LOW",
) -> dict:
    return {
        "request_id": f"req_{abs(hash(text))}",
        "trace_id": "trace_memory_test",
        "user_id": user_id,
        "thread_id": thread_id,
        "message": {
            "role": "user",
            "content_type": "text",
            "content": text,
            "attachments": [],
        },
        "options": {
            "memory": {
                "can_read": True,
                "can_write": can_write,
            }
        },
        "intent_result": {
            "risk_level": risk_level,
        },
        "current_node": "general_chat_node",
        "response_message": {
            "role": "assistant",
            "content_type": "text",
            "content": "好的。",
        },
    }


class TestMemorySystem(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.store = InMemoryStore()
        self.service = MemoryService(store=self.store)

    async def test_explicit_consent_required_for_long_term_write(self) -> None:
        result = await self.service.commit_after_response(
            _state("以后回答我简短一点", can_write=False)
        )

        self.assertEqual(result["status"], "SKIPPED")
        self.assertEqual(result["reason"], "missing_long_term_memory_consent")

        items = await self.store.asearch((MEMORY_NAMESPACE, "1"), limit=10)
        self.assertEqual(items, [])

    async def test_conflicting_preference_latest_wins(self) -> None:
        first = await self.service.commit_after_response(
            _state("以后回答我简短一点")
        )
        second = await self.service.commit_after_response(
            _state("以后回答我详细一点")
        )

        self.assertEqual(first["status"], "CREATED")
        self.assertEqual(second["status"], "UPDATED")

        active = await self.store.asearch(
            (MEMORY_NAMESPACE, "1"),
            filter={
                "status": ACTIVE_STATUS,
                "memory_key": "communication:answer_detail",
            },
            limit=10,
        )
        superseded = await self.store.asearch(
            (MEMORY_NAMESPACE, "1"),
            filter={
                "status": SUPERSEDED_STATUS,
                "memory_key": "communication:answer_detail",
            },
            limit=10,
        )

        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].value["value"], "详细")
        self.assertEqual(len(superseded), 1)

        profile = await self.store.aget((PROFILE_NAMESPACE, "1"), "profile")
        self.assertIsNotNone(profile)
        preference = profile.value["preferences"]["communication:answer_detail"]
        self.assertEqual(preference["value"], "详细")

    async def test_high_risk_context_blocks_long_term_write(self) -> None:
        result = await self.service.commit_after_response(
            _state("以后记住我胸痛的时候想问吃什么药", risk_level="HIGH")
        )

        self.assertEqual(result["status"], "SKIPPED")
        self.assertEqual(result["reason"], "high_risk_context_blocked")

    async def test_delete_user_memory_clears_long_term_namespaces(self) -> None:
        await self.service.commit_after_response(_state("以后回答我简短一点"))

        deleted = await self.service.delete_user_memory(user_id=1)
        listed = await self.service.list_user_memory(user_id=1)

        self.assertEqual(deleted["status"], "DELETED")
        self.assertEqual(listed["profile"], {})
        self.assertEqual(listed["memories"], [])
        self.assertEqual(listed["summaries"], [])

    async def test_short_term_memory_uses_redis_and_compresses_old_turns(self) -> None:
        redis = FakeRedis()
        cache = ShortTermMemoryCache(
            redis=redis,
            ttl_seconds=3600,
            keep_last_turns=1,
            summary_turn_threshold=2,
            summary_char_threshold=99999,
        )

        await cache.append_turn(_state("第一轮问题", thread_id="t1"))
        result = await cache.append_turn(_state("第二轮问题", thread_id="t1"))
        context = await cache.load_context("t1")

        self.assertEqual(result["status"], "UPDATED")
        self.assertTrue(result["compressed"])
        self.assertEqual(context["turn_count"], 2)
        self.assertEqual(len(context["recent_turns"]), 1)
        self.assertIn("第一轮问题", context["conversation_summary"])


if __name__ == "__main__":
    unittest.main()
