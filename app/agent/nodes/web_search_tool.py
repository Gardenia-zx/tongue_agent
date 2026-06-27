from typing import Any

import httpx

from app.core.config import get_settings


async def search_web(query: str) -> dict[str, Any]:
    settings = get_settings()
    query = query.strip()
    if not settings.web_search_enabled or not query:
        return _empty("disabled_or_empty_query", query)
    if not settings.tavily_api_key:
        return _empty("missing_tavily_api_key", query)

    try:
        async with httpx.AsyncClient(timeout=settings.web_search_timeout_seconds) as client:
            response = await client.post(
                settings.web_search_endpoint,
                headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
                json={
                    "query": query,
                    "search_depth": "basic",
                    "max_results": settings.web_search_top_k,
                    "include_answer": True,
                    "include_raw_content": False,
                },
            )
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        return _empty(type(exc).__name__, query)

    results = _parse_results(payload)[: settings.web_search_top_k]
    return {
        "query": query,
        "results": results,
        "grounded": bool(results),
        "debug": {"source": "tavily", "endpoint": settings.web_search_endpoint},
    }


def _parse_results(payload: Any) -> list[dict[str, str]]:
    if not isinstance(payload, dict):
        return []

    results: list[dict[str, str]] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        snippet = str(item.get("content") or item.get("raw_content") or "").strip()
        if snippet:
            results.append(
                {
                    "title": str(item.get("title") or item.get("url") or "Tavily search result"),
                    "snippet": snippet,
                    "url": str(item.get("url") or ""),
                }
            )

    return results


def _empty(reason: str, query: str) -> dict[str, Any]:
    return {
        "query": query,
        "results": [],
        "grounded": False,
        "debug": {"reason": reason, "source": "tavily"},
    }
