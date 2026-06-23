from typing import Any

from pydantic import BaseModel, Field


class RagSourceDocument(BaseModel):
    schema_version: str = "1.0"
    doc_id: str
    title: str
    content: str
    source_type: str = "manual_seed"
    source_uri: str | None = None
    language: str = "zh-CN"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagChunk(BaseModel):
    schema_version: str = "1.0"
    chunk_id: str
    doc_id: str
    title: str
    content: str
    chunk_index: int
    source_type: str
    source_uri: str | None = None
    language: str = "zh-CN"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagRetrievalHit(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    content: str
    source_uri: str | None = None
    tags: list[str] = Field(default_factory=list)
    bm25_score: float = 0.0
    vector_score: float = 0.0
    final_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagAnswer(BaseModel):
    schema_version: str = "1.0"
    answer: str
    query: str
    hits: list[RagRetrievalHit] = Field(default_factory=list)
    retrieval_engine: str = "ES_BM25_TEXT2VEC_HYBRID"
    answer_engine: str = "deepseek_openai_compatible"
    grounded: bool = False
    debug: dict[str, Any] = Field(default_factory=dict)
