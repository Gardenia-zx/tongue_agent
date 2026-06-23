import asyncio
import mimetypes
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from app.core.config import get_settings


SUPPORTED_IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

CONTENT_TYPE_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class TongueModelError(RuntimeError):
    pass


class TongueModelClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        bearer_token: str | None,
        timeout_seconds: int,
        max_concurrency: int,
        max_image_size_mb: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.bearer_token = bearer_token
        self.timeout_seconds = timeout_seconds
        self.max_image_size_bytes = max_image_size_mb * 1024 * 1024
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.transport = transport

    async def predict_image_path(self, image_path: str | Path) -> dict[str, Any]:
        path = Path(image_path)
        if not path.is_file():
            raise TongueModelError(f"Tongue image does not exist: {path}")

        image_bytes = path.read_bytes()
        content_type = self._guess_content_type(path.name)
        return await self.predict_image_bytes(
            image_bytes=image_bytes,
            filename=path.name,
            content_type=content_type,
        )

    async def predict_image_url(self, image_url: str) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(image_url)
        except httpx.HTTPError as exc:
            raise TongueModelError(f"Download tongue image failed: {exc}") from exc

        if response.status_code >= 400:
            raise TongueModelError(
                f"Download tongue image failed: {response.status_code}"
            )

        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        if content_type not in SUPPORTED_IMAGE_CONTENT_TYPES:
            filename = Path(urlparse(image_url).path).name or "tongue_image"
            content_type = self._guess_content_type(filename)
        else:
            filename = Path(urlparse(image_url).path).name or "tongue_image"

        return await self.predict_image_bytes(
            image_bytes=response.content,
            filename=filename,
            content_type=content_type,
        )

    async def predict_image_bytes(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> dict[str, Any]:
        if not image_bytes:
            raise TongueModelError("Tongue image is empty")

        if len(image_bytes) > self.max_image_size_bytes:
            raise TongueModelError("Tongue image is larger than configured limit")

        if content_type not in SUPPORTED_IMAGE_CONTENT_TYPES:
            raise TongueModelError(f"Unsupported tongue image content type: {content_type}")

        headers = {}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        if self.api_key:
            headers["x-api-key"] = self.api_key

        files = {
            "file": (
                filename or "tongue_image",
                image_bytes,
                content_type,
            )
        }

        try:
            async with self.semaphore:
                async with httpx.AsyncClient(
                    timeout=self.timeout_seconds,
                    transport=self.transport,
                ) as client:
                    response = await client.post(
                        f"{self.base_url}/v1/tongue/predict",
                        headers=headers,
                        files=files,
                    )
        except httpx.HTTPError as exc:
            raise TongueModelError(f"Tongue model request failed: {exc}") from exc

        if response.status_code >= 400:
            raise TongueModelError(
                f"Tongue model request failed: {response.status_code} {response.text}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise TongueModelError("Tongue model returned invalid JSON") from exc

        return self._extract_model_data(payload)

    @staticmethod
    def _guess_content_type(filename: str) -> str:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed in SUPPORTED_IMAGE_CONTENT_TYPES:
            return guessed

        suffix = Path(filename).suffix.lower()
        content_type = CONTENT_TYPE_BY_SUFFIX.get(suffix)
        if content_type:
            return content_type

        raise TongueModelError(f"Unsupported tongue image file type: {filename}")

    @staticmethod
    def _extract_model_data(payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TongueModelError("Tongue model returned non-object payload")

        if payload.get("code") not in {None, 0}:
            raise TongueModelError(
                f"Tongue model returned error: {payload.get('message') or payload}"
            )

        data = payload.get("data", payload)
        if not isinstance(data, dict):
            raise TongueModelError("Tongue model returned invalid data payload")

        return data


@lru_cache
def get_tongue_model_client() -> TongueModelClient:
    settings = get_settings()
    return TongueModelClient(
        base_url=settings.tongue_model_base_url,
        api_key=settings.tongue_model_api_key,
        bearer_token=settings.tongue_model_bearer_token,
        timeout_seconds=settings.tongue_model_timeout_seconds,
        max_concurrency=settings.tongue_model_max_concurrency,
        max_image_size_mb=settings.tongue_model_max_image_size_mb,
    )
