import hashlib
from datetime import UTC, datetime
from typing import TypeVar
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.rag.models import (
    RagBase,
    RagChunkRecord,
    RagDocumentRecord,
    RagIndexJobRecord,
)
from app.schemas.rag import RagChunk, RagSourceDocument


T = TypeVar("T")


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _iter_batches(items: list[T], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


async def create_rag_tables(session: AsyncSession) -> None:
    connection = await session.connection()
    await connection.run_sync(RagBase.metadata.create_all)
    await session.commit()


async def create_index_job(
    session: AsyncSession,
    *,
    job_type: str,
    source_type: str,
    source_uri: str | None,
    source_commit: str | None,
    metadata: dict,
) -> str:
    job_id = str(uuid4())
    session.add(
        RagIndexJobRecord(
            job_id=job_id,
            job_type=job_type,
            source_type=source_type,
            source_uri=source_uri,
            source_commit=source_commit,
            status="RUNNING",
            metadata_=metadata,
        )
    )
    await session.commit()
    return job_id


async def finish_index_job(
    session: AsyncSession,
    *,
    job_id: str,
    status: str,
    total_documents: int,
    total_chunks: int,
    indexed_chunks: int,
    error_message: str | None = None,
) -> None:
    record = await session.get(RagIndexJobRecord, job_id)
    if record is None:
        return

    record.status = status
    record.total_documents = total_documents
    record.total_chunks = total_chunks
    record.indexed_chunks = indexed_chunks
    record.error_message = error_message
    record.finished_at = datetime.now(UTC)
    await session.commit()


async def upsert_documents(
    session: AsyncSession,
    *,
    documents: list[RagSourceDocument],
) -> None:
    if not documents:
        return

    settings = get_settings()
    table = RagDocumentRecord.__table__
    values = [_document_values(document) for document in documents]

    for batch in _iter_batches(values, settings.rag_db_insert_batch_size):
        statement = insert(table).values(batch)
        update_columns = {
            "title": statement.excluded.title,
            "content_hash": statement.excluded.content_hash,
            "source_type": statement.excluded.source_type,
            "source_repo": statement.excluded.source_repo,
            "source_path": statement.excluded.source_path,
            "source_uri": statement.excluded.source_uri,
            "source_commit": statement.excluded.source_commit,
            "license": statement.excluded.license,
            "content_storage": statement.excluded.content_storage,
            "content_path": statement.excluded.content_path,
            "language": statement.excluded.language,
            "tags": statement.excluded.tags,
            "metadata": statement.excluded["metadata"],
            "status": statement.excluded.status,
            "version": table.c.version + 1,
            "updated_at": datetime.now(UTC),
        }
        await session.execute(
            statement.on_conflict_do_update(
                index_elements=[table.c.doc_id],
                set_=update_columns,
            )
        )

    await session.commit()


async def replace_chunks(
    session: AsyncSession,
    *,
    chunks: list[RagChunk],
) -> None:
    if not chunks:
        return

    settings = get_settings()
    doc_ids = sorted({chunk.doc_id for chunk in chunks})
    await session.execute(delete(RagChunkRecord).where(RagChunkRecord.doc_id.in_(doc_ids)))

    values = [_chunk_values(chunk, embedding_model=settings.embedding_model_name) for chunk in chunks]
    for batch in _iter_batches(values, settings.rag_db_insert_batch_size):
        statement = insert(RagChunkRecord.__table__).values(batch)
        await session.execute(statement)

    await session.commit()


async def load_active_chunks(
    session: AsyncSession,
    *,
    doc_ids: list[str] | None = None,
) -> list[RagChunkRecord]:
    statement = select(RagChunkRecord).where(RagChunkRecord.status == "ACTIVE")
    if doc_ids:
        statement = statement.where(RagChunkRecord.doc_id.in_(doc_ids))

    result = await session.execute(statement.order_by(RagChunkRecord.doc_id, RagChunkRecord.chunk_index))
    return list(result.scalars().all())


def _document_values(document: RagSourceDocument) -> dict:
    metadata = dict(document.metadata)
    return {
        "doc_id": document.doc_id,
        "title": document.title,
        "content_hash": content_hash(document.content),
        "source_type": document.source_type,
        "source_repo": metadata.get("source_repo"),
        "source_path": metadata.get("source_path") or document.source_uri,
        "source_uri": document.source_uri,
        "source_commit": metadata.get("source_commit"),
        "license": metadata.get("license"),
        "content_storage": metadata.get("content_storage", "inline"),
        "content_path": metadata.get("content_path") or document.source_uri,
        "language": document.language,
        "tags": document.tags,
        "metadata": metadata,
        "status": metadata.get("status", "ACTIVE"),
    }


def _chunk_values(chunk: RagChunk, *, embedding_model: str) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "chunk_index": chunk.chunk_index,
        "title": chunk.title,
        "content": chunk.content,
        "content_hash": content_hash(chunk.content),
        "token_count": len(chunk.content),
        "embedding_model": embedding_model,
        "source_type": chunk.source_type,
        "source_uri": chunk.source_uri,
        "language": chunk.language,
        "tags": chunk.tags,
        "metadata": chunk.metadata,
        "status": chunk.metadata.get("status", "ACTIVE"),
    }
