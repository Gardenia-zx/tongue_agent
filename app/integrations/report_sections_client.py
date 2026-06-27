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
        headers["X-Internal-Api-Key"] = settings.java_internal_api_key

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
    except httpx.TimeoutException:
        return {"status": "TIMEOUT", "error": "report_sections_timeout"}
    except httpx.HTTPError as exc:
        return {"status": "FAILED", "error": type(exc).__name__}

    if response.status_code == 403:
        return {"status": "FORBIDDEN", "error": "report_sections_forbidden"}
    if response.status_code == 404:
        return {"status": "NOT_FOUND", "error": "report_not_found"}
    if response.status_code == 409:
        try:
            result = response.json()
        except ValueError:
            result = {}
        return {
            "status": "VERSION_MISMATCH",
            "error": "report_version_mismatch",
            **(result if isinstance(result, dict) else {}),
        }
    if response.status_code >= 400:
        return {
            "status": "FAILED",
            "error": f"report_sections_http_{response.status_code}",
        }

    try:
        result = response.json()
    except ValueError:
        return {"status": "FAILED", "error": "invalid_response_json"}
    return result if isinstance(result, dict) else {"status": "FAILED", "error": "invalid_response"}
