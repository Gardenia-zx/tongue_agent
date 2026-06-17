from typing import Any
from uuid import uuid4

from langmem import create_manage_memory_tool, create_search_memory_tool

from app.agent.state import AgentState
from app.memory.policy import decide_memory_policy, extract_user_text


MEMORY_NAMESPACE = "memories"
PROFILE_NAMESPACE = "profiles"


class MemoryService:
    def __init__(self, *, store) -> None:
        self.store = store

    async def build_memory_context(self, state: AgentState) -> dict[str, Any]:
        user_id = state.get("user_id")
        if user_id is None:
            return self._empty_context(skip_reason="missing_user_id")

        user_key = str(user_id)
        text = extract_user_text(state)
        decision = decide_memory_policy(state)

        memories = []
        if decision.can_read:
            memories = await self._search_memories(
                user_key=user_key,
                query=text or "用户偏好 当前任务 最近关注点",
            )

        profile = await self._get_profile(user_key)
        write_result = None

        if decision.can_write and decision.write_candidate:
            write_result = await self._write_memory(
                user_key=user_key,
                content=decision.write_candidate,
            )

        context_summary = self._build_context_summary(
            memories=memories,
            profile=profile,
            write_result=write_result,
            skip_reason=decision.skip_reason,
        )

        return {
            "used_memory_ids": [
                item.get("memory_id") for item in memories if item.get("memory_id")
            ],
            "memories": memories,
            "profile": profile,
            "write_result": write_result,
            "policy": {
                "can_read": decision.can_read,
                "can_write": decision.can_write,
                "skip_reason": decision.skip_reason,
            },
            "context_summary": context_summary,
        }

    async def _search_memories(
        self,
        *,
        user_key: str,
        query: str,
    ) -> list[dict[str, Any]]:
        search_tool = create_search_memory_tool(
            namespace=(MEMORY_NAMESPACE, user_key),
            store=self.store,
        )

        result = await search_tool.ainvoke(
            {
                "query": query,
                "limit": 5,
            }
        )

        return self._normalize_search_result(result)

    async def _write_memory(
        self,
        *,
        user_key: str,
        content: str,
    ) -> dict[str, Any]:
        manage_tool = create_manage_memory_tool(
            namespace=(MEMORY_NAMESPACE, user_key),
            store=self.store,
        )
        memory_id = f"mem_{uuid4().hex}"

        result = await manage_tool.ainvoke(
            {
                "action": "create",
                "content": content,
                "id": memory_id,
            }
        )

        return {
            "memory_id": memory_id,
            "status": "CREATED",
            "result": result,
        }

    async def _get_profile(self, user_key: str) -> dict[str, Any]:
        item = await self.store.aget((PROFILE_NAMESPACE, user_key), "profile")
        if item is None:
            return {}

        value = getattr(item, "value", None)
        if isinstance(value, dict):
            return value

        return {}

    @staticmethod
    def _normalize_search_result(result: Any) -> list[dict[str, Any]]:
        if result is None:
            return []

        if isinstance(result, str):
            return [
                {
                    "memory_id": None,
                    "text": result,
                    "source": "langmem_search_tool",
                }
            ]

        if isinstance(result, list):
            items = result
        elif isinstance(result, tuple):
            items = list(result)
        else:
            items = [result]

        normalized: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                normalized.append(item)
                continue

            value = getattr(item, "value", None)
            key = getattr(item, "key", None)
            score = getattr(item, "score", None)

            if isinstance(value, dict):
                normalized.append(
                    {
                        "memory_id": key,
                        **value,
                        "score": score,
                    }
                )
            elif value is not None:
                normalized.append(
                    {
                        "memory_id": key,
                        "text": str(value),
                        "score": score,
                    }
                )
            else:
                normalized.append(
                    {
                        "memory_id": key,
                        "text": str(item),
                        "score": score,
                    }
                )

        return normalized

    @staticmethod
    def _build_context_summary(
        *,
        memories: list[dict[str, Any]],
        profile: dict[str, Any],
        write_result: dict[str, Any] | None,
        skip_reason: str | None,
    ) -> str:
        parts = []

        if profile:
            parts.append("已加载用户画像。")

        if memories:
            parts.append(f"检索到 {len(memories)} 条长期记忆。")

        if write_result:
            parts.append("已根据用户明确授权写入一条长期记忆。")

        if skip_reason:
            parts.append(f"本轮未写入长期记忆：{skip_reason}。")

        if not parts:
            return "没有可用的长期记忆上下文。"

        return "".join(parts)

    @staticmethod
    def _empty_context(*, skip_reason: str) -> dict[str, Any]:
        return {
            "used_memory_ids": [],
            "memories": [],
            "profile": {},
            "write_result": None,
            "policy": {
                "can_read": False,
                "can_write": False,
                "skip_reason": skip_reason,
            },
            "context_summary": "没有可用的长期记忆上下文。",
        }
