import argparse
import asyncio
from pathlib import Path
from typing import Any

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from app.core.config import get_settings
from app.integrations.es_client import create_es_client
from app.integrations.model_gateway import get_embedding_model
from app.integrations.rag_database import open_rag_session
from app.rag.chunker import chunk_documents
from app.rag.document_loader import load_documents
from app.rag.models import RagChunkRecord
from app.rag.repository import (
    create_index_job,
    create_rag_tables,
    finish_index_job,
    load_active_chunks,
    replace_chunks,
    upsert_documents,
)


def _rag_mapping(embedding_dim: int) -> dict[str, Any]:
    return {
        "settings": {
            "analysis": {
                "tokenizer": {
                    "cn_ngram_tokenizer": {
                        "type": "ngram",
                        "min_gram": 2,
                        "max_gram": 3,
                        "token_chars": ["letter", "digit"],
                    }
                },
                "analyzer": {
                    "cn_ngram_analyzer": {
                        "type": "custom",
                        "tokenizer": "cn_ngram_tokenizer",
                        "filter": ["lowercase"],
                    }
                },
            }
        },
        "mappings": {
            "properties": {
                "schema_version": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "cn_ngram_analyzer",
                    "fields": {
                        "raw": {
                            "type": "keyword",
                            "ignore_above": 1024,
                        }
                    },
                },
                "content": {
                    "type": "text",
                    "analyzer": "cn_ngram_analyzer",
                    "fields": {
                        "raw": {
                            "type": "keyword",
                            "ignore_above": 4096,
                        }
                    },
                },
                "content_hash": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "source_type": {"type": "keyword"},
                "source_uri": {"type": "keyword"},
                "language": {"type": "keyword"},
                "tags": {"type": "keyword"},
                "metadata": {"type": "object", "enabled": True},
                "status": {"type": "keyword"},
                "version": {"type": "integer"},
                "embedding_model": {"type": "keyword"},
                "enabled": {"type": "boolean"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": embedding_dim,
                    "index": False,
                },
            }
        }
    }


async def ensure_rag_index(
    es: AsyncElasticsearch,
    *,
    index_name: str,
    embedding_dim: int,
    recreate: bool,
) -> None:
    exists = await es.indices.exists(index=index_name)
    if exists and recreate:
        await es.indices.delete(index=index_name)
        exists = False

    if not exists:
        await es.indices.create(
            index=index_name,
            **_rag_mapping(embedding_dim),
        )


def _bulk_actions(
    index_name: str,
    chunks: list[RagChunkRecord],
    embeddings: list[list[float]],
):
    for chunk, embedding in zip(chunks, embeddings):
        document = {
            "schema_version": "1.0",
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "content": chunk.content,
            "content_hash": chunk.content_hash,
            "chunk_index": chunk.chunk_index,
            "source_type": chunk.source_type,
            "source_uri": chunk.source_uri,
            "language": chunk.language,
            "tags": chunk.tags,
            "metadata": chunk.metadata_,
            "status": chunk.status,
            "version": chunk.version,
            "embedding_model": chunk.embedding_model,
            "enabled": chunk.status == "ACTIVE",
        }
        document["embedding"] = embedding
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": chunk.chunk_id,
            "_source": document,
        }


async def index_knowledge_base(
    *,
    source_dir: str | Path | None = None,
    source_repo: str | None = None,
    source_commit: str | None = None,
    license_name: str | None = None,
    recreate: bool = False,
) -> tuple[int, int]:
    settings = get_settings()
    es = create_es_client()
    job_id: str | None = None
    total_documents = 0
    total_chunks = 0
    indexed_chunks = 0

    try:
        async with open_rag_session() as session:
            await create_rag_tables(session)
            documents = load_documents(
                directory=source_dir,
                source_repo=source_repo,
                source_commit=source_commit,
                license_name=license_name,
            )
            chunks = chunk_documents(
                documents,
                chunk_size=settings.rag_chunk_size,
                chunk_overlap=settings.rag_chunk_overlap,
            )
            total_documents = len(documents)
            total_chunks = len(chunks)
            job_id = await create_index_job(
                session,
                job_type="FULL_REINDEX" if recreate else "UPSERT_REINDEX",
                source_type="local_directory" if source_dir else "seed",
                source_uri=str(source_dir) if source_dir else "internal_seed",
                source_commit=source_commit,
                metadata={
                    "source_repo": source_repo,
                    "license": license_name,
                    "rag_index_name": settings.rag_index_name,
                    "embedding_model": settings.embedding_model_name,
                },
            )
            await upsert_documents(session, documents=documents)
            await replace_chunks(session, chunks=chunks)
            chunk_records = await load_active_chunks(
                session,
                doc_ids=[document.doc_id for document in documents],
            )

        await ensure_rag_index(
            es,
            index_name=settings.rag_index_name,
            embedding_dim=settings.embedding_dim,
            recreate=recreate,
        )

        embedding_model = get_embedding_model()
        embeddings = await embedding_model.embed_texts(
            [f"{chunk.title}\n{chunk.content}" for chunk in chunk_records]
        )

        if chunk_records:
            await async_bulk(
                es,
                _bulk_actions(settings.rag_index_name, chunk_records, embeddings),
                refresh=True,
            )
            indexed_chunks = len(chunk_records)

        if job_id:
            async with open_rag_session() as session:
                await finish_index_job(
                    session,
                    job_id=job_id,
                    status="SUCCEEDED",
                    total_documents=total_documents,
                    total_chunks=total_chunks,
                    indexed_chunks=indexed_chunks,
                )

        return total_documents, total_chunks
    except Exception as exc:
        if job_id:
            async with open_rag_session() as session:
                await finish_index_job(
                    session,
                    job_id=job_id,
                    status="FAILED",
                    total_documents=total_documents,
                    total_chunks=total_chunks,
                    indexed_chunks=indexed_chunks,
                    error_message=str(exc),
                )
        raise
    finally:
        await es.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index RAG knowledge chunks into ES.")
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Optional directory containing .txt/.md knowledge documents.",
    )
    parser.add_argument(
        "--source-repo",
        default=None,
        help="Source repository URL, for example https://github.com/PanckooAI/TCM_Datasets.",
    )
    parser.add_argument(
        "--source-commit",
        default=None,
        help="Source repository commit hash for reproducible ingestion.",
    )
    parser.add_argument(
        "--license",
        dest="license_name",
        default=None,
        help="License or usage note for the source corpus.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the RAG index before indexing.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    doc_count, chunk_count = await index_knowledge_base(
        source_dir=args.source_dir,
        source_repo=args.source_repo,
        source_commit=args.source_commit,
        license_name=args.license_name,
        recreate=args.recreate,
    )
    print(f"Indexed RAG knowledge base: {doc_count} documents, {chunk_count} chunks.")


if __name__ == "__main__":
    asyncio.run(_main())
