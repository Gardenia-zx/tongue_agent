import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

from app.core.config import get_settings
from app.integrations.model_gateway import get_store_embeddings


@dataclass(frozen=True)
class LangGraphPersistence:
    checkpointer: AsyncPostgresSaver
    store: AsyncPostgresStore


@asynccontextmanager
async def open_langgraph_persistence() -> AsyncIterator[LangGraphPersistence]:
    settings = get_settings()

    if settings.langgraph_strict_msgpack:
        os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"

    async with AsyncPostgresSaver.from_conn_string(
        settings.langgraph_postgres_uri
    ) as checkpointer:
        await checkpointer.setup()

        async with AsyncPostgresStore.from_conn_string(
            settings.langgraph_postgres_uri,
            index={
                "dims": settings.embedding_dim,
                "embed": get_store_embeddings(),
                "fields": ["text", "summary"],
            },
        ) as store:
            await store.setup()
            yield LangGraphPersistence(
                checkpointer=checkpointer,
                store=store,
            )
