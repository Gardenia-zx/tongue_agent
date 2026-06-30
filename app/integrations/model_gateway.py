import asyncio
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


class ModelGatewayError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatGenerationResult:
    content: str
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


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

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = await self._chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            extra_body=extra_body,
        )
        choice = (data.get("choices") or [])[0]
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            raise ModelGatewayError("Chat model returned invalid message")

        return message

    async def _chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

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

        return data

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        result = await self.generate_with_metadata(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        return result.content

    async def generate_with_metadata(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatGenerationResult:
        merged_extra_body = dict(extra_body or {})
        if response_format:
            merged_extra_body["response_format"] = response_format

        data = await self._chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=merged_extra_body or None,
        )
        choice = (data.get("choices") or [])[0]
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            raise ModelGatewayError("Chat model returned invalid message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ModelGatewayError("Chat model returned empty content")

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        return ChatGenerationResult(
            content=content.strip(),
            finish_reason=choice.get("finish_reason"),
            usage=usage,
        )


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
