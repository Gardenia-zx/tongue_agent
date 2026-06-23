from elasticsearch import AsyncElasticsearch

from app.core.config import get_settings


# 获取es的客户端
def create_es_client() -> AsyncElasticsearch:
    settings = get_settings()
    return AsyncElasticsearch(
        hosts=[settings.elasticsearch_url],
        request_timeout=10,
        retry_on_timeout=True,
        max_retries=2,
    )