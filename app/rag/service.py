from app.core.config import get_settings
from app.integrations.model_gateway import (
    LocalEmbeddingModel,
    ModelGatewayError,
    get_chat_model_client,
)
from app.rag.retriever import HybridRagRetriever
from app.rag.query_expansion import expand_rag_query
from app.schemas.rag import RagAnswer, RagRetrievalHit


RAG_SYSTEM_PROMPT = """你是中医舌象健康管理系统中的健康知识问答助手。
你只能基于给定的知识库片段回答，不要编造来源。

安全要求：
1. 只能提供一般健康知识和健康管理建议，不能做疾病诊断。
2. 药物、方剂、处方类问题可以做一般知识参考：说明常见方向、适用边界、禁忌和就医沟通要点；不要声称已为用户确诊。
3. 不要要求用户自行停药、换药、加药或减药；涉及具体剂量和特殊人群时提醒医生或药师确认。
4. 如果资料不足，要明确说明资料不足，并给出下一步可选方向。
5. 如果涉及明显不适、持续加重或急症风险，建议线下就医。
6. 回答要简洁、通俗，适合网页端展示。
"""


class RagQAService:
    def __init__(
        self,
        *,
        retriever: HybridRagRetriever,
        embedding_model: LocalEmbeddingModel,
    ) -> None:
        self.retriever = retriever
        self.embedding_model = embedding_model
        self.settings = get_settings()

    async def answer(self, *, query: str) -> RagAnswer:
        query = query.strip()
        if not query:
            return RagAnswer(
                answer="请先告诉我你想了解的健康知识问题。",
                query=query,
                grounded=False,
                debug={"reason": "empty_query"},
            )

        expanded_query = expand_rag_query(query)
        query_embedding = await self.embedding_model.embed_text(
            expanded_query.embedding_text
        )
        hits = await self.retriever.retrieve(
            query=query,
            query_embedding=query_embedding,
            bm25_top_k=self.settings.rag_bm25_top_k,
            vector_top_k=self.settings.rag_vector_top_k,
            final_top_k=self.settings.rag_final_top_k,
            min_relevance_score=self.settings.rag_min_relevance_score,
        )

        if not hits:
            return RagAnswer(
                answer=(
                    "我暂时没有在知识库中找到足够相关的资料。"
                    "你可以换一种问法，或上传舌象图片开始个体化舌象分析。"
                ),
                query=query,
                hits=[],
                grounded=False,
                debug={"reason": "no_retrieval_hits"},
            )

        content = await self._generate_answer(query=query, hits=hits)
        return RagAnswer(
            answer=content,
            query=query,
            hits=hits,
            grounded=True,
            debug={
                "hit_count": len(hits),
                "top_score": hits[0].final_score if hits else 0.0,
            },
        )

    async def _generate_answer(
        self,
        *,
        query: str,
        hits: list[RagRetrievalHit],
    ) -> str:
        context_text = self._format_context(hits)
        messages = [
            {
                "role": "system",
                "content": RAG_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{query}\n\n"
                    f"知识库片段：\n{context_text}\n\n"
                    "请基于上述片段回答。不要输出内部推理过程。"
                ),
            },
        ]

        try:
            return await get_chat_model_client().generate(
                messages=messages,
                temperature=self.settings.chat_model_temperature,
                max_tokens=self.settings.chat_model_max_tokens,
            )
        except ModelGatewayError:
            return self._fallback_answer(query=query, hits=hits)

    @staticmethod
    def _format_context(hits: list[RagRetrievalHit]) -> str:
        lines: list[str] = []
        for index, hit in enumerate(hits, 1):
            lines.append(
                (
                    f"[{index}] 标题：{hit.title}\n"
                    f"标签：{'、'.join(hit.tags)}\n"
                    f"内容：{hit.content}\n"
                    f"相关度：{hit.final_score:.3f}"
                )
            )
        return "\n\n".join(lines)

    @staticmethod
    def _fallback_answer(*, query: str, hits: list[RagRetrievalHit]) -> str:
        top_hit = hits[0]
        return (
            f"根据知识库中“{top_hit.title}”的资料，{top_hit.content}"
            "以上只能作为一般健康知识参考，不能替代医生诊断。"
        )
