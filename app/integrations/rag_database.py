from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings


def create_rag_engine():
    settings = get_settings()
    return create_async_engine(
        settings.rag_database_url,
        pool_pre_ping=True,
    )


@asynccontextmanager
async def open_rag_session() -> AsyncIterator[AsyncSession]:
    engine = create_rag_engine()
    session_factory = async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        try:
            yield session
        finally:
            await engine.dispose()
