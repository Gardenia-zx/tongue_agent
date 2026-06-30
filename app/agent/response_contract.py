import json
import re
from typing import Any


DISCLAIMER = "以上内容用于一般健康知识说明和健康管理参考，不能替代医生诊断。"

INTERNAL_FINAL_BLOCKERS = {
    "tool_decision",
    "tool_calls",
    "next_action",
    "quality_review",
    "state_snapshot",
}

INTERNAL_JSON_MARKERS = (
    '"type"',
    '"structured_content"',
    '"structuredContent"',
    '"tool_decision"',
    '"tool_calls"',
    '"quality_review"',
    '"next_action"',
    '"state_snapshot"',
    '"agent_loop"',
    '"final_answer"',
)


def plain_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace("```json", "").replace("```", "")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_like_internal_json(text: str) -> bool:
    compact = text.strip()
    if not compact:
        return False
    if compact.startswith("```json"):
        return True
    if "```json" in compact and any(marker in compact for marker in INTERNAL_JSON_MARKERS):
        return True
    if compact.startswith("{") and any(marker in compact for marker in INTERNAL_JSON_MARKERS):
        return True
    return False


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    for candidate in candidates:
        if candidate.get("type") == "final_answer":
            return candidate
    return candidates[0] if candidates else None


def is_final_answer_candidate(payload: dict[str, Any]) -> bool:
    if payload.get("type") == "final_answer":
        return True
    if not isinstance(payload.get("content"), str):
        return False
    return not any(field in payload for field in INTERNAL_FINAL_BLOCKERS)


def normalize_structured_content(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not value:
        return None

    normalized = dict(value)
    sections: list[dict[str, Any]] = []
    for raw_section in normalized.get("sections") or []:
        if not isinstance(raw_section, dict):
            continue
        title = plain_text(raw_section.get("title"))
        content = plain_text(raw_section.get("content"))
        items = [
            plain_text(item)
            for item in (raw_section.get("items") or [])
            if isinstance(item, (str, int, float)) and plain_text(item)
        ]
        if not title and not content and not items:
            continue
        section: dict[str, Any] = {}
        section_key = raw_section.get("section_key") or raw_section.get("sectionKey")
        if isinstance(section_key, str) and section_key.strip():
            section["section_key"] = section_key.strip()
        if title:
            section["title"] = title
        if content:
            section["content"] = content
        if items:
            section["items"] = items
        sections.append(section)

    normalized["sections"] = sections
    normalized.setdefault("schema_version", normalized.get("schemaVersion") or "1.0")
    normalized.setdefault("disclaimer", DISCLAIMER)
    return normalized


def structured_content_to_text(structured_content: dict[str, Any] | None) -> str:
    if not structured_content:
        return ""
    parts: list[str] = []
    for key in ("summary", "title"):
        text = plain_text(structured_content.get(key))
        if text and text not in parts:
            parts.append(text)
    for section in structured_content.get("sections") or []:
        if not isinstance(section, dict):
            continue
        title = plain_text(section.get("title"))
        content = plain_text(section.get("content"))
        items = [plain_text(item) for item in section.get("items") or [] if plain_text(item)]
        if title:
            parts.append(f"{title}：")
        if content:
            parts.append(content)
        parts.extend(f"{index}. {item}" for index, item in enumerate(items, start=1))
    disclaimer = plain_text(structured_content.get("disclaimer"))
    if disclaimer:
        parts.append(disclaimer)
    return "\n".join(part for part in parts if part).strip()


def content_for_display(content: str, structured_content: dict[str, Any] | None) -> str:
    natural = plain_text(content)
    if natural and not looks_like_internal_json(natural):
        return natural
    return structured_content_to_text(structured_content)


def final_answer_from_text(text: str) -> tuple[str, dict[str, Any] | None] | None:
    payload = extract_json_object(text)
    if isinstance(payload, dict) and is_final_answer_candidate(payload):
        structured = normalize_structured_content(
            payload.get("structured_content") or payload.get("structuredContent")
        )
        content = content_for_display(str(payload.get("content") or ""), structured)
        return (content, structured) if content else None
    if looks_like_internal_json(text):
        return None
    content = plain_text(text)
    return (content, None) if content else None
