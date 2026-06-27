import unittest

from app.memory.service import (
    ACTIVE_STATUS,
    MEMORY_NAMESPACE,
    PROFILE_NAMESPACE,
    SUPERSEDED_STATUS,
    MemoryService,
)
from app.agent.nodes.memory_node import memory_commit_node, memory_read_node


class FakeStoreItem:
    def __init__(self, key: str, value: dict) -> None:
        self.key = key
        self.value = value


class FakeStore:
    def __init__(self) -> None:
        self.data: dict[tuple[str, str], dict[str, dict]] = {}

    async def aput(self, namespace, key: str, value: dict, index=None) -> None:
        self.data.setdefault(tuple(namespace), {})[key] = value

    async def aget(self, namespace, key: str):
        value = self.data.get(tuple(namespace), {}).get(key)
        if value is None:
            return None
        return FakeStoreItem(key, value)

    async def adelete(self, namespace, key: str) -> None:
        self.data.get(tuple(namespace), {}).pop(key, None)

    async def asearch(self, namespace, query=None, filter=None, limit: int = 10):
        values = self.data.get(tuple(namespace), {})
        result = []
        for key, value in values.items():
            if self._matches(value, filter):
                result.append(FakeStoreItem(key, value))
        return result[:limit]

    @staticmethod
    def _matches(value: dict, filter: dict | None) -> bool:
        if not filter:
            return True
        return all(value.get(key) == expected for key, expected in filter.items())


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
        self.store = FakeStore()
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

    async def test_preferred_name_is_user_profile_memory(self) -> None:
        result = await self.service.commit_after_response(
            _state("我叫 zx，你记住了吗", thread_id="thread-a")
        )

        self.assertEqual(result["status"], "CREATED")
        self.assertEqual(result["memory_key"], "user_identity:preferred_name")

        profile = await self.store.aget((PROFILE_NAMESPACE, "1"), "profile")
        preference = profile.value["preferences"]["user_identity:preferred_name"]
        self.assertEqual(preference["memory_type"], "user_identity")
        self.assertEqual(preference["value"], "zx")

        cross_thread = await self.service.build_memory_context(
            _state("你记得我叫什么吗", thread_id="thread-b", can_write=False),
            session_context={},
        )
        self.assertEqual(
            "zx",
            cross_thread["profile"]["preferences"]["user_identity:preferred_name"]["value"],
        )

    async def test_sensitive_symptom_text_is_not_long_term_memory(self) -> None:
        result = await self.service.commit_after_response(
            _state("请记住我的症状是胸闷")
        )

        self.assertEqual(result["status"], "SKIPPED")
        self.assertEqual(result["reason"], "sensitive_health_memory_blocked")
        items = await self.store.asearch((MEMORY_NAMESPACE, "1"), limit=10)
        self.assertEqual(items, [])

    async def test_mysql_recovery_writes_back_checkpointer_short_term_memory(self) -> None:
        state = {
            **_state("继续说", can_write=False),
            "options": {
                "context": {"mode": "mysql_recovery"},
                "memory": {"can_read": True, "can_write": False},
            },
            "context_bundle": {
                "mode": "mysql_recovery",
                "conversation_summary": {"text": "恢复摘要"},
                "recent_messages": [
                    {
                        "message_id": "mysql-a1",
                        "role": "assistant",
                        "content_type": "text",
                        "content": "MySQL 里的上一轮回答",
                    }
                ],
            },
        }

        read = await memory_read_node(state, store=self.store)
        self.assertEqual("mysql_recovery", read["memory_context"]["session"]["source"])
        self.assertEqual(1, len(read["memory_context"]["recent_messages"]))

        committed = await memory_commit_node(read, store=self.store)
        self.assertEqual("checkpointer", committed["short_term_memory"]["source"])
        contents = [
            item["content"]
            for item in committed["short_term_memory"]["recent_messages"]
        ]
        self.assertIn("MySQL 里的上一轮回答", contents)
        self.assertIn("继续说", contents)

        next_read = await memory_read_node(
            {
                **_state("下一轮", can_write=False),
                "options": {
                    "context": {"mode": "stateful"},
                    "memory": {"can_read": True, "can_write": False},
                },
                "short_term_memory": committed["short_term_memory"],
            },
            store=self.store,
        )

        self.assertEqual("checkpointer", next_read["memory_context"]["session"]["source"])
        next_contents = [
            item["content"]
            for item in next_read["memory_context"]["recent_messages"]
        ]
        self.assertIn("MySQL 里的上一轮回答", next_contents)

    async def test_stateful_context_bundle_does_not_make_turn_stateless(self) -> None:
        state = {
            **_state("普通聊天", can_write=False),
            "options": {
                "context": {"mode": "stateful"},
                "memory": {"can_read": True, "can_write": False},
            },
            "context_bundle": {
                "recent_messages": [
                    {"role": "assistant", "content": "不该作为普通历史"}
                ]
            },
        }

        read = await memory_read_node(state, store=self.store)

        self.assertEqual("checkpointer", read["memory_context"]["session"]["source"])
        self.assertEqual([], read["memory_context"]["recent_messages"])

if __name__ == "__main__":
    unittest.main()
