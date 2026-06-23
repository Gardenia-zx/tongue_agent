import asyncio

from app.integrations.rag_database import open_rag_session
from app.rag.repository import create_rag_tables


async def setup_rag_database() -> None:
    async with open_rag_session() as session:
        await create_rag_tables(session)
    print("RAG database tables are ready.")


if __name__ == "__main__":
    asyncio.run(setup_rag_database())
