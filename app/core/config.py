from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "tongue-agent"
    app_env: str = "local"
    log_level: str = "INFO"

    redis_url: str = "redis://localhost:6379/0"
    elasticsearch_url: str = "http://localhost:9200"

    langgraph_postgres_uri: str = (
        "postgresql://postgres:postgres@localhost:5432/tongue_agent?sslmode=disable"
    )
    langgraph_strict_msgpack: bool = True
    rag_database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/tongue_agent"
    )

    model_gateway_base_url: str = "http://localhost:9000"
    model_gateway_api_key: str | None = None
    model_gateway_timeout_seconds: int = 30
    chat_model_name: str = "qwen2.5-7b-instruct"
    chat_model_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    chat_model_max_tokens: int = Field(default=800, ge=64, le=4096)
    chat_model_max_concurrency: int = Field(default=4, ge=1, le=32)
    tongue_model_base_url: str = "http://127.0.0.1:9100"
    tongue_model_api_key: str | None = None
    tongue_model_bearer_token: str | None = None
    tongue_model_timeout_seconds: int = Field(default=30, ge=1, le=180)
    tongue_model_max_concurrency: int = Field(default=2, ge=1, le=16)
    tongue_model_max_image_size_mb: int = Field(default=10, ge=1, le=50)

    agent_lock_ttl_seconds: int = Field(default=120, ge=10, le=600)
    idempotency_ttl_seconds: int = Field(default=86400, ge=60)

    intent_top_k: int = Field(default=5, ge=1, le=20)
    intent_route_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    intent_clarify_threshold: float = Field(default=0.55, ge=0.0, le=1.0)

    rag_bm25_top_k: int = Field(default=20, ge=1, le=100)
    rag_vector_top_k: int = Field(default=20, ge=1, le=100)
    rag_final_top_k: int = Field(default=8, ge=1, le=50)
    rag_index_name: str = "rag_chunk_v1"
    rag_chunk_size: int = Field(default=520, ge=100, le=2000)
    rag_chunk_overlap: int = Field(default=80, ge=0, le=500)
    rag_min_relevance_score: float = Field(default=0.2, ge=0.0, le=1.0)
    rag_db_insert_batch_size: int = Field(default=500, ge=1, le=2000)

    memory_enabled: bool = True
    long_term_memory_enabled: bool = True
    short_term_memory_ttl_seconds: int = Field(default=86400, ge=60)
    short_term_memory_keep_last_turns: int = Field(default=4, ge=1, le=20)
    short_term_memory_summary_turn_threshold: int = Field(default=6, ge=2, le=100)
    short_term_memory_summary_char_threshold: int = Field(default=4000, ge=500)
    long_term_memory_top_k: int = Field(default=5, ge=1, le=20)
    long_term_memory_summary_top_k: int = Field(default=2, ge=0, le=10)
    long_term_memory_compress_threshold: int = Field(default=50, ge=5, le=500)

    intent_index_name: str = "intent_example_v1"
    domain_term_index_name: str = "domain_term_v1"

    embedding_model_name: str = "shibing624/text2vec-base-chinese"
    embedding_dim: int = 768
    embedding_max_concurrency: int = Field(default=2, ge=1, le=8)


@lru_cache
def get_settings() -> Settings:
    return Settings()
