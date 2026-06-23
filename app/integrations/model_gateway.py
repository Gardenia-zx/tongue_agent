import asyncio
from functools import lru_cache
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


class ModelGatewayError(RuntimeError):
    pass


class LocalEmbeddingModel:
    def __init__(
        self,
        *,
        model_name: str,
        max_concurrency: int,
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def embed_text(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        async with self.semaphore:
            vectors = await asyncio.to_thread(
                self.model.encode,
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        return [vector.astype(float).tolist() for vector in vectors]

    def embed_text_sync(self, text: str) -> list[float]:
        vectors = self.embed_texts_sync([text])
        return vectors[0]

    def embed_texts_sync(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.astype(float).tolist() for vector in vectors]


@lru_cache
def get_embedding_model() -> LocalEmbeddingModel:
    settings = get_settings()
    return LocalEmbeddingModel(
        model_name=settings.embedding_model_name,
        max_concurrency=settings.embedding_max_concurrency,
    )


class Text2VecEmbeddings(Embeddings):
    def __init__(self, embedding_model: LocalEmbeddingModel) -> None:
        self.embedding_model = embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedding_model.embed_texts_sync(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embedding_model.embed_text_sync(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.embedding_model.embed_texts(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await self.embedding_model.embed_text(text)


@lru_cache
def get_store_embeddings() -> Text2VecEmbeddings:
    return Text2VecEmbeddings(get_embedding_model())


class ChatModelClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model_name: str,
        timeout_seconds: int,
        max_concurrency: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def generate(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if extra_body:
            payload.update(extra_body)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with self.semaphore:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(
                        f"{self.base_url}/v1/chat/completions",
                        headers=headers,
                        json=payload,
                    )
        except httpx.HTTPError as exc:
            raise ModelGatewayError(f"Chat model request failed: {exc}") from exc

        if response.status_code >= 400:
            raise ModelGatewayError(
                f"Chat model request failed: {response.status_code} {response.text}"
            )

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ModelGatewayError("Chat model returned empty choices")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ModelGatewayError("Chat model returned empty content")

        return content.strip()


@lru_cache
def get_chat_model_client() -> ChatModelClient:
    settings = get_settings()
    return ChatModelClient(
        base_url=settings.model_gateway_base_url,
        api_key=settings.model_gateway_api_key,
        model_name=settings.chat_model_name,
        timeout_seconds=settings.model_gateway_timeout_seconds,
        max_concurrency=settings.chat_model_max_concurrency,
    )
