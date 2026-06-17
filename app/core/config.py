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

    model_gateway_base_url: str = "http://localhost:9000"
    model_gateway_api_key: str | None = None
    model_gateway_timeout_seconds: int = 30
    chat_model_name: str = "qwen2.5-7b-instruct"
    chat_model_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    chat_model_max_tokens: int = Field(default=800, ge=64, le=4096)
    chat_model_max_concurrency: int = Field(default=4, ge=1, le=32)

    agent_lock_ttl_seconds: int = Field(default=120, ge=10, le=600)
    idempotency_ttl_seconds: int = Field(default=86400, ge=60)

    intent_top_k: int = Field(default=5, ge=1, le=20)
    intent_route_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    intent_clarify_threshold: float = Field(default=0.55, ge=0.0, le=1.0)

    rag_bm25_top_k: int = Field(default=20, ge=1, le=100)
    rag_vector_top_k: int = Field(default=20, ge=1, le=100)
    rag_final_top_k: int = Field(default=8, ge=1, le=50)

    memory_enabled: bool = True
    long_term_memory_enabled: bool = True

    intent_index_name: str = "intent_example_v1"
    domain_term_index_name: str = "domain_term_v1"

    embedding_model_name: str = "shibing624/text2vec-base-chinese"
    embedding_dim: int = 768
    embedding_max_concurrency: int = Field(default=2, ge=1, le=8)


@lru_cache
def get_settings() -> Settings:
    return Settings()
