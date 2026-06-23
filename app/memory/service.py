from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.agent.state import AgentState
from app.core.config import get_settings
from app.memory.policy import (
    MemoryCandidate,
    decide_memory_policy,
    extract_user_text,
)


MEMORY_NAMESPACE = "memories"
PROFILE_NAMESPACE = "profiles"
SUMMARY_NAMESPACE = "memory_summaries"
AUDIT_NAMESPACE = "memory_audit"

ACTIVE_STATUS = "ACTIVE"
SUPERSEDED_STATUS = "SUPERSEDED"


class MemoryService:
    def __init__(self, *, store) -> None:
        self.store = store
        self.settings = get_settings()

    async def build_memory_context(
        self,
        state: AgentState,
        *,
        session_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        user_id = state.get("user_id")
        if user_id is None:
            return self._empty_context(
                session_context=session_context,
                skip_reason="missing_user_id",
            )

        decision = decide_memory_policy(state)
        text = extract_user_text(state)
        user_key = str(user_id)

        profile = await self._get_profile(user_key)
        memories: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []

        if decision.can_read and self.settings.long_term_memory_enabled:
            memories = await self._search_active_memories(
                user_key=user_key,
                query=text or "用户偏好 当前任务 最近关注点",
                limit=self.settings.long_term_memory_top_k,
            )
            summaries = await self._search_summaries(
                user_key=user_key,
                query=text or None,
                limit=self.settings.long_term_memory_summary_top_k,
            )

        context_summary = self._build_context_summary(
            memories=memories,
            summaries=summaries,
            profile=profile,
            session_context=session_context,
            skip_reason=decision.skip_reason,
        )

        return {
            "used_memory_ids": [
                item.get("memory_id") for item in memories if item.get("memory_id")
            ],
            "relevant_memories": memories,
            "memories": memories,
            "summaries": summaries,
            "profile": profile,
            "session": session_context or {},
            "conversation_summary": (session_context or {}).get(
                "conversation_summary", ""
            ),
            "recent_turns": (session_context or {}).get("recent_turns", []),
            "write_result": None,
            "policy": {
                "can_read": decision.can_read,
                "can_write": decision.can_write,
                "skip_reason": decision.skip_reason,
            },
            "context_summary": context_summary,
        }

    async def commit_after_response(self, state: AgentState) -> dict[str, Any]:
        user_id = state.get("user_id")
        if user_id is None:
            return {"status": "SKIPPED", "reason": "missing_user_id"}

        if not self.settings.long_term_memory_enabled:
            return {"status": "SKIPPED", "reason": "long_term_memory_disabled"}

        decision = decide_memory_policy(state)
        if not decision.can_write or decision.memory_candidate is None:
            return {
                "status": "SKIPPED",
                "reason": decision.skip_reason,
                "policy": {
                    "can_read": decision.can_read,
                    "can_write": decision.can_write,
                },
            }

        user_key = str(user_id)
        result = await self._upsert_memory(
            user_key=user_key,
            candidate=decision.memory_candidate,
            request_id=state.get("request_id"),
        )
        await self._upsert_profile(
            user_key=user_key,
            candidate=decision.memory_candidate,
        )
        compression_result = await self._compress_if_needed(user_key=user_key)
        result["compression"] = compression_result
        return result

    async def list_user_memory(self, *, user_id: int | str) -> dict[str, Any]:
        user_key = str(user_id)
        memories = await self._search_active_memories(
            user_key=user_key,
            query=None,
            limit=1000,
        )
        summaries = await self._search_summaries(
            user_key=user_key,
            query=None,
            limit=100,
        )
        profile = await self._get_profile(user_key)

        return {
            "profile": profile,
            "memories": memories,
            "summaries": summaries,
        }

    async def delete_user_memory(self, *, user_id: int | str) -> dict[str, Any]:
        user_key = str(user_id)
        deleted = 0

        for namespace in [
            (MEMORY_NAMESPACE, user_key),
            (SUMMARY_NAMESPACE, user_key),
            (AUDIT_NAMESPACE, user_key),
        ]:
            items = await self._search_items(namespace, query=None, filter=None, limit=1000)
            for item in items:
                await self.store.adelete(namespace, item["memory_id"])
                deleted += 1

        await self.store.adelete((PROFILE_NAMESPACE, user_key), "profile")
        deleted += 1

        return {
            "status": "DELETED",
            "deleted_items": deleted,
            "scope": "long_term_memory",
        }

    async def _upsert_memory(
        self,
        *,
        user_key: str,
        candidate: MemoryCandidate,
        request_id: str | None,
    ) -> dict[str, Any]:
        now = self._now()
        existing = await self._find_active_by_memory_key(
            user_key=user_key,
            memory_key=candidate.memory_key,
        )

        if existing and self._same_memory(existing, candidate):
            return {
                "status": "UNCHANGED",
                "memory_id": existing["memory_id"],
                "memory_key": candidate.memory_key,
            }

        superseded_id = None
        if existing:
            superseded_id = existing["memory_id"]
            old_value = dict(existing)
            old_value["status"] = SUPERSEDED_STATUS
            old_value["updated_at"] = now
            await self.store.aput(
                (MEMORY_NAMESPACE, user_key),
                superseded_id,
                old_value,
                index=["text", "summary"],
            )

        memory_id = f"mem_{uuid4().hex}"
        record = {
            "schema_version": "1.0",
            "memory_id": memory_id,
            "memory_type": candidate.memory_type,
            "memory_key": candidate.memory_key,
            "text": candidate.text,
            "summary": candidate.summary,
            "value": candidate.value,
            "status": ACTIVE_STATUS,
            "source": candidate.source,
            "created_at": now,
            "updated_at": now,
            "supersedes": superseded_id,
        }
        await self.store.aput(
            (MEMORY_NAMESPACE, user_key),
            memory_id,
            record,
            index=["text", "summary"],
        )

        action = "UPDATED" if superseded_id else "CREATED"
        await self._write_audit(
            user_key=user_key,
            action=action,
            memory_key=candidate.memory_key,
            request_id=request_id,
            old_memory_id=superseded_id,
            new_memory_id=memory_id,
        )

        return {
            "status": action,
            "memory_id": memory_id,
            "memory_key": candidate.memory_key,
            "superseded_memory_id": superseded_id,
        }

    async def _upsert_profile(
        self,
        *,
        user_key: str,
        candidate: MemoryCandidate,
    ) -> None:
        now = self._now()
        profile = await self._get_profile(user_key)
        preferences = profile.get("preferences")
        if not isinstance(preferences, dict):
            preferences = {}

        preferences[candidate.memory_key] = {
            "memory_type": candidate.memory_type,
            "value": candidate.value,
            "summary": candidate.summary,
            "updated_at": now,
        }

        profile.update(
            {
                "schema_version": "1.0",
                "profile_id": f"profile_{user_key}",
                "preferences": preferences,
                "summary": self._profile_summary(preferences),
                "updated_at": now,
            }
        )
        profile.setdefault("created_at", now)

        await self.store.aput(
            (PROFILE_NAMESPACE, user_key),
            "profile",
            profile,
            index=False,
        )

    async def _compress_if_needed(self, *, user_key: str) -> dict[str, Any]:
        active_memories = await self._search_active_memories(
            user_key=user_key,
            query=None,
            limit=1000,
        )
        if len(active_memories) < self.settings.long_term_memory_compress_threshold:
            return {
                "status": "SKIPPED",
                "reason": "below_threshold",
                "active_memory_count": len(active_memories),
            }

        grouped: dict[str, list[dict[str, Any]]] = {}
        for memory in active_memories:
            grouped.setdefault(str(memory.get("memory_type") or "general"), []).append(memory)

        now = self._now()
        for memory_type, items in grouped.items():
            summary_text = "；".join(
                str(item.get("summary") or item.get("text") or "") for item in items[:20]
            )[:2000]
            summary_id = f"summary_{memory_type}"
            value = {
                "schema_version": "1.0",
                "summary_id": summary_id,
                "memory_type": memory_type,
                "text": summary_text,
                "summary": summary_text,
                "status": ACTIVE_STATUS,
                "source_memory_count": len(items),
                "updated_at": now,
            }
            await self.store.aput(
                (SUMMARY_NAMESPACE, user_key),
                summary_id,
                value,
                index=["text", "summary"],
            )

        return {
            "status": "UPDATED",
            "summary_count": len(grouped),
            "active_memory_count": len(active_memories),
        }

    async def _get_profile(self, user_key: str) -> dict[str, Any]:
        item = await self.store.aget((PROFILE_NAMESPACE, user_key), "profile")
        if item is None:
            return {}

        value = getattr(item, "value", None)
        if isinstance(value, dict):
            return value

        return {}

    async def _search_active_memories(
        self,
        *,
        user_key: str,
        query: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        return await self._search_items(
            (MEMORY_NAMESPACE, user_key),
            query=query,
            filter={"status": ACTIVE_STATUS},
            limit=limit,
        )

    async def _search_summaries(
        self,
        *,
        user_key: str,
        query: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

        return await self._search_items(
            (SUMMARY_NAMESPACE, user_key),
            query=query,
            filter={"status": ACTIVE_STATUS},
            limit=limit,
        )

    async def _find_active_by_memory_key(
        self,
        *,
        user_key: str,
        memory_key: str,
    ) -> dict[str, Any] | None:
        items = await self._search_items(
            (MEMORY_NAMESPACE, user_key),
            query=None,
            filter={"status": ACTIVE_STATUS, "memory_key": memory_key},
            limit=1,
        )
        if not items:
            return None
        return items[0]

    async def _search_items(
        self,
        namespace: tuple[str, str],
        *,
        query: str | None,
        filter: dict[str, Any] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        try:
            raw_items = await self.store.asearch(
                namespace,
                query=query,
                filter=filter,
                limit=limit,
            )
        except Exception:
            raw_items = await self.store.asearch(
                namespace,
                query=None,
                filter=filter,
                limit=limit,
            )

        return self._normalize_store_items(raw_items)

    async def _write_audit(
        self,
        *,
        user_key: str,
        action: str,
        memory_key: str,
        request_id: str | None,
        old_memory_id: str | None,
        new_memory_id: str,
    ) -> None:
        audit_id = f"audit_{uuid4().hex}"
        await self.store.aput(
            (AUDIT_NAMESPACE, user_key),
            audit_id,
            {
                "schema_version": "1.0",
                "audit_id": audit_id,
                "action": action,
                "memory_key": memory_key,
                "request_id": request_id,
                "old_memory_id": old_memory_id,
                "new_memory_id": new_memory_id,
                "created_at": self._now(),
            },
            index=False,
        )

    @staticmethod
    def _normalize_store_items(raw_items: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in raw_items:
            value = getattr(item, "value", None)
            key = getattr(item, "key", None)
            score = getattr(item, "score", None)

            if isinstance(value, dict):
                data = dict(value)
            elif value is not None:
                data = {"text": str(value)}
            else:
                data = {"text": str(item)}

            data.setdefault("memory_id", key)
            if score is not None:
                data["score"] = score
            normalized.append(data)

        return normalized

    @staticmethod
    def _same_memory(
        existing: dict[str, Any],
        candidate: MemoryCandidate,
    ) -> bool:
        return (
            existing.get("memory_key") == candidate.memory_key
            and existing.get("value") == candidate.value
            and existing.get("status") == ACTIVE_STATUS
        )

    @staticmethod
    def _profile_summary(preferences: dict[str, Any]) -> str:
        summaries = [
            str(item.get("summary"))
            for item in preferences.values()
            if isinstance(item, dict) and item.get("summary")
        ]
        return "；".join(summaries)[:1000]

    @staticmethod
    def _build_context_summary(
        *,
        memories: list[dict[str, Any]],
        summaries: list[dict[str, Any]],
        profile: dict[str, Any],
        session_context: dict[str, Any] | None,
        skip_reason: str | None,
    ) -> str:
        parts = []

        if session_context and session_context.get("conversation_summary"):
            parts.append("已加载会话内压缩摘要。")

        if session_context and session_context.get("recent_turns"):
            parts.append(
                f"已加载最近 {len(session_context.get('recent_turns') or [])} 轮会话。"
            )

        if profile:
            parts.append("已加载用户长期画像。")

        if summaries:
            parts.append(f"检索到 {len(summaries)} 条长期记忆摘要。")

        if memories:
            parts.append(f"检索到 {len(memories)} 条相关长期记忆。")

        if skip_reason:
            parts.append(f"本轮长期记忆写入状态：{skip_reason}。")

        if not parts:
            return "没有可用的记忆上下文。"

        return "".join(parts)

    @staticmethod
    def _empty_context(
        *,
        session_context: dict[str, Any] | None,
        skip_reason: str,
    ) -> dict[str, Any]:
        return {
            "used_memory_ids": [],
            "relevant_memories": [],
            "memories": [],
            "summaries": [],
            "profile": {},
            "session": session_context or {},
            "conversation_summary": (session_context or {}).get(
                "conversation_summary", ""
            ),
            "recent_turns": (session_context or {}).get("recent_turns", []),
            "write_result": None,
            "policy": {
                "can_read": False,
                "can_write": False,
                "skip_reason": skip_reason,
            },
            "context_summary": "没有可用的长期记忆上下文。",
        }

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()
