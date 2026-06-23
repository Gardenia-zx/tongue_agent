from typing import Any

from app.core.config import get_settings
from app.integrations.es_client import create_es_client
from app.integrations.model_gateway import get_embedding_model
from app.rag.retriever import HybridRagRetriever
from app.rag.service import RagQAService


def fallback_rag_response(reason: str, *, query: str = "") -> dict[str, Any]:
    return {
        "answer": (
            "健康知识库暂时不可用，我可以先基于已识别到的舌象特征做简要说明。"
            "这些内容只作为一般健康知识参考，不能替代医生诊断。"
        ),
        "query": query,
        "hits": [],
        "grounded": False,
        "debug": {"reason": reason},
    }


async def answer_with_rag(query: str) -> dict[str, Any]:
    settings = get_settings()
    es = create_es_client()

    try:
        service = RagQAService(
            retriever=HybridRagRetriever(
                es,
                index_name=settings.rag_index_name,
            ),
            embedding_model=get_embedding_model(),
        )
        rag_answer = await service.answer(query=query)
        return rag_answer.model_dump(mode="json")
    except Exception as exc:
        return fallback_rag_response(type(exc).__name__, query=query)
    finally:
        await es.close()
