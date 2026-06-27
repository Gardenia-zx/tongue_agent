from typing import Any

from app.core.config import get_settings
from app.integrations.es_client import create_es_client
from app.agent.nodes.web_search_tool import search_web
from app.rag.retriever import HybridRagRetriever


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
        from app.integrations.model_gateway import get_embedding_model
        from app.rag.service import RagQAService

        service = RagQAService(
            retriever=HybridRagRetriever(
                es,
                index_name=settings.rag_index_name,
            ),
            embedding_model=get_embedding_model(),
        )
        rag_answer = await service.answer(query=query)
        result = rag_answer.model_dump(mode="json")
    except Exception as exc:
        result = fallback_rag_response(type(exc).__name__, query=query)
    finally:
        await es.close()

    if result.get("grounded") and result.get("hits"):
        return result
    return await _with_web_search_fallback(result, query=query)


async def _with_web_search_fallback(rag_result: dict[str, Any], *, query: str) -> dict[str, Any]:
    web_result = await search_web(query)
    results = web_result.get("results") or []
    if not results:
        rag_result.setdefault("debug", {})["web_search"] = web_result.get("debug") or {}
        return rag_result

    snippets = []
    for index, item in enumerate(results, start=1):
        title = str(item.get("title") or "公开资料").strip()
        snippet = str(item.get("snippet") or "").strip()
        if snippet:
            snippets.append(f"{index}. {title}：{snippet}")

    answer = (
        "本地知识库暂未检索到足够资料，以下结合公开资料做一般健康知识参考："
        + "\n".join(snippets[:3])
        + "\n以上内容仅供健康管理参考，不能替代医生诊断；涉及用药、方剂或特殊人群时建议咨询医生或药师。"
    )
    return {
        **rag_result,
        "answer": answer,
        "hits": [
            {
                "chunk_id": f"web_{index}",
                "doc_id": "web_search",
                "title": item.get("title") or "公开资料",
                "content": item.get("snippet") or "",
                "source_uri": item.get("url") or "",
                "tags": ["web_search"],
                "bm25_score": 0.0,
                "vector_score": 0.0,
                "final_score": 0.0,
                "metadata": {"source": "web_search"},
            }
            for index, item in enumerate(results, start=1)
        ],
        "retrieval_engine": "WEB_SEARCH_FALLBACK",
        "grounded": True,
        "debug": {
            **(rag_result.get("debug") or {}),
            "reason": "rag_no_hits_web_fallback",
            "web_search": web_result.get("debug") or {},
        },
    }
