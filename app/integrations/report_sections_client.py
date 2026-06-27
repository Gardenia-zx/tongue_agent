from typing import Any

import httpx

from app.core.config import get_settings


async def load_report_sections_from_java(
    *,
    tenant_id: str | None,
    user_id: int | None,
    thread_id: str | None,
    thread_epoch: int | None,
    conversation_id: str | None,
    turn_id: str | None,
    report_id: Any,
    report_version: Any,
    sections: list[str],
) -> dict[str, Any]:
    settings = get_settings()
    url = f"{settings.java_backend_base_url.rstrip('/')}/internal/agent/reports/{report_id}/sections"
    headers: dict[str, str] = {}
    if settings.java_internal_api_key:
        headers["Authorization"] = f"Bearer {settings.java_internal_api_key}"

    payload = {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "thread_id": thread_id,
        "thread_epoch": thread_epoch,
        "conversation_id": conversation_id,
        "turn_id": turn_id,
        "report_id": report_id,
        "report_version": report_version,
        "sections": sections,
    }

    try:
        async with httpx.AsyncClient(timeout=settings.report_sections_timeout_seconds) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
    except Exception as exc:
        return {"status": "FAILED", "error": type(exc).__name__}

    return result if isinstance(result, dict) else {"status": "FAILED", "error": "invalid_response"}
