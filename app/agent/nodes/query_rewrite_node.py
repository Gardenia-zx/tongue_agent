import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.agent.context_builder import (
    build_query_rewrite_context,
    ensure_current_turn,
    extract_user_text,
    query_rewrite_context_from_state,
    with_query_context,
)
from app.agent.state import AgentState


REF_LAST_ANSWER = "LAST_ANSWER"
REF_LAST_ANSWER_ITEM = "LAST_ANSWER_ITEM"
REF_ACTIVE_REPORT = "ACTIVE_REPORT"
REF_REPORT_ITEM = "REPORT_ITEM"
REF_PREVIOUS_TASK = "PREVIOUS_TASK"
REF_GENERAL_TOPIC = "GENERAL_TOPIC"
REF_UNKNOWN = "UNKNOWN"
REF_AMBIGUOUS = "AMBIGUOUS"

REFERENCE_TARGET_TYPES = {
    REF_LAST_ANSWER,
    REF_LAST_ANSWER_ITEM,
    REF_ACTIVE_REPORT,
    REF_REPORT_ITEM,
    REF_PREVIOUS_TASK,
    REF_GENERAL_TOPIC,
    REF_UNKNOWN,
    REF_AMBIGUOUS,
}

FOCUS_DIET_ADVICE = "DIET_ADVICE"
FOCUS_DETAILED_REPORT = "DETAILED_REPORT"
FOCUS_GENERAL_DETAIL = "GENERAL_DETAIL"
FOCUS_FEATURE_EXPLANATION = "FEATURE_EXPLANATION"
FOCUS_UNKNOWN = "UNKNOWN"

ROUTE_REPORT_FOLLOWUP = "report_followup_subgraph"
ROUTE_HEALTH_QA = "health_qa_subgraph"
ROUTE_GENERAL_CHAT = "general_chat_subgraph"
ROUTE_TONGUE_ANALYSIS = "tongue_analysis_subgraph"
ALLOWED_ROUTE_HINTS = {
    ROUTE_REPORT_FOLLOWUP,
    ROUTE_HEALTH_QA,
    ROUTE_GENERAL_CHAT,
    ROUTE_TONGUE_ANALYSIS,
}
DEFAULT_HIGH_CONFIDENCE_THRESHOLD = 0.85
EVIDENCE_SOURCES = {
    "raw_user_input",
    "recent_messages",
    "last_final_answer",
    "conversation_summary",
    "active_report_ref",
}


class LLMReferenceResolution(BaseModel):
    status: str = "AMBIGUOUS"
    target_type: str = REF_AMBIGUOUS
    target_focus: str = FOCUS_UNKNOWN
    is_context_dependent: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    target_message_id: Any = None
    target_report_id: Any = None
    evidence_sources: list[str] = Field(default_factory=list)


class LLMRewriteOutput(BaseModel):
    standalone_query: str
    reference_resolution: LLMReferenceResolution
    route_hint: str | None = None
    rewrite_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

QUERY_REWRITE_SYSTEM_PROMPT = """你是中医舌象健康 Agent 的追问消解器。
你的职责是把依赖上下文的问题改写为独立问题，并解析用户指代的对象。
reference_resolution.target_type 只能表示指代对象，例如 LAST_ANSWER、ACTIVE_REPORT、REPORT_ITEM。
业务路由建议必须单独写入 route_hint，不能混入 target_type。
本节点不回答用户问题。
"""


def _compact(text: str) -> str:
    return "".join(text.split())


def _contains_any(text: str, keywords: list[str]) -> bool:
    compact = _compact(text)
    return any(keyword in compact for keyword in keywords)


def _is_short_followup(text: str) -> bool:
    return _contains_any(
        text,
        [
            "详细一点",
            "详细点",
            "更详细",
            "再详细一点",
            "讲详细点",
            "说详细点",
            "展开说说",
            "继续说",
            "继续",
            "具体一点",
            "再具体点",
            "多说一点",
            "不够详细",
            "太简单",
            "这个呢",
            "这个是什么意思",
            "为什么",
        ],
    )


def _is_context_dependent_input(text: str) -> bool:
    return _is_short_followup(text) or _contains_any(
        text,
        [
            "这个",
            "那个",
            "刚才",
            "上面",
            "前面",
            "上一轮",
            "第二个",
            "第一个",
            "第三个",
            "继续",
            "展开",
            "详细点",
            "再说说",
            "它",
            "这份",
            "那份",
        ],
    )


def _mentions_report(text: str) -> bool:
    return _contains_any(
        text,
        [
            "报告",
            "舌象报告",
            "分析报告",
            "我的舌象",
            "舌象分析",
            "图片",
            "舌头",
            "识别结果",
        ],
    )


def _mentions_health_qa_followup(text: str) -> bool:
    return _contains_any(
        text,
        [
            "饮食",
            "吃什么",
            "怎么吃",
            "忌口",
            "调理",
            "建议",
            "注意",
            "症状",
            "表现",
        ],
    )


def _focus_from_text(text: str) -> str:
    if _contains_any(
        text,
        [
            "饮食",
            "吃什么",
            "怎么吃",
            "忌口",
            "食物",
            "推荐",
            "调理",
            "清淡",
            "生冷",
            "油腻",
        ],
    ):
        return FOCUS_DIET_ADVICE

    if _contains_any(text, ["详细报告", "完整报告", "更详细的报告", "重新生成报告"]):
        return FOCUS_DETAILED_REPORT

    if _contains_any(text, ["白苔", "黄苔", "舌苔", "舌质", "齿痕", "裂纹", "厚薄", "润燥"]):
        return FOCUS_FEATURE_EXPLANATION

    if _is_short_followup(text):
        return FOCUS_GENERAL_DETAIL

    return FOCUS_UNKNOWN


def _focus_from_last_answer(last_answer: dict[str, Any] | None) -> str:
    if not last_answer:
        return FOCUS_UNKNOWN

    answer_type = _answer_type(last_answer)
    if answer_type in {"TONGUE_REPORT", "DETAILED_TONGUE_REPORT"}:
        return FOCUS_DETAILED_REPORT

    structured = last_answer.get("structured_content")
    title = ""
    section_titles: list[str] = []
    if isinstance(structured, dict):
        title = str(structured.get("title") or "")
        for section in structured.get("sections") or []:
            if isinstance(section, dict):
                section_titles.append(str(section.get("title") or ""))

    structured_marker = " ".join([title, *section_titles])
    content = str((last_answer or {}).get("content") or "")
    combined = " ".join([structured_marker, content[:800]])

    if answer_type == "REPORT_FOLLOWUP" and _contains_any(
        structured_marker or content[:120],
        ["饮食", "吃什么", "忌口", "食物", "清淡", "生冷", "油腻"],
    ):
        return FOCUS_DIET_ADVICE

    if _contains_any(combined, ["详细舌象报告", "完整报告"]):
        return FOCUS_DETAILED_REPORT

    if _contains_any(combined, ["舌苔", "舌质", "白苔", "黄苔", "齿痕", "厚薄", "润燥"]):
        return FOCUS_FEATURE_EXPLANATION

    return FOCUS_UNKNOWN


def _resolve_focus(raw_text: str, last_answer: dict[str, Any] | None) -> str:
    text_focus = _focus_from_text(raw_text)
    if text_focus != FOCUS_GENERAL_DETAIL and text_focus != FOCUS_UNKNOWN:
        return text_focus

    inherited_focus = _focus_from_last_answer(last_answer)
    if inherited_focus != FOCUS_UNKNOWN:
        return inherited_focus

    return text_focus


def _mentions_tongue_analysis_start(text: str) -> bool:
    return bool(re.search(r"(上传|拍|看|分析|识别).*?(舌|舌象|舌头|图片)", _compact(text)))


def _answer_type(last_answer: dict[str, Any] | None) -> str:
    if not last_answer:
        return ""
    answer_type = last_answer.get("answer_type")
    if isinstance(answer_type, str):
        return answer_type
    structured = last_answer.get("structured_content")
    if isinstance(structured, dict):
        value = structured.get("answer_type") or structured.get("answerType")
        if isinstance(value, str):
            return value
    return ""


def _node_name(last_answer: dict[str, Any] | None) -> str:
    if not last_answer:
        return ""
    value = last_answer.get("node_name")
    return value if isinstance(value, str) else ""


def _last_answer_route_hint(last_answer: dict[str, Any] | None) -> str | None:
    node_name = _node_name(last_answer)
    answer_type = _answer_type(last_answer)
    route_target = str((last_answer or {}).get("route_target") or "")

    if node_name in {"health_qa_node"} or answer_type == "HEALTH_QA" or route_target == "health_qa_subgraph":
        return ROUTE_HEALTH_QA

    if (
        node_name in {"tongue_report_node", "report_followup_node"}
        or answer_type in {"TONGUE_REPORT", "REPORT_FOLLOWUP", "DETAILED_TONGUE_REPORT"}
        or route_target in {"tongue_analysis_subgraph", "report_followup_subgraph"}
        or (last_answer or {}).get("report_id")
    ):
        return ROUTE_REPORT_FOLLOWUP

    if node_name == "tongue_analysis_node":
        return ROUTE_TONGUE_ANALYSIS

    if node_name == "general_chat_node" or answer_type == "GENERAL_CHAT":
        return ROUTE_GENERAL_CHAT

    return None


def _standalone_for_target(
    *,
    raw_text: str,
    route_hint: str | None,
    target_focus: str,
    last_answer: dict[str, Any] | None,
    active_report: dict[str, Any] | None,
) -> str:
    raw_text = raw_text.strip()
    last_content = str((last_answer or {}).get("content") or "").strip()

    if route_hint == ROUTE_REPORT_FOLLOWUP:
        report_summary = ""
        if active_report:
            report_summary = str(
                active_report.get("summary")
                or active_report.get("feature_summary")
                or active_report.get("featureSummary")
                or ""
            ).strip()
        subject = report_summary[:500] or last_content[:500] or "当前舌象报告"
        if target_focus == FOCUS_DIET_ADVICE:
            return (
                f"请延续上一轮饮食建议，并结合当前活动舌象报告，展开饮食推荐、忌口、"
                f"日常安排和观察要点：{raw_text}。报告上下文：{subject}。上一轮回答：{last_content[:500]}"
            )
        if target_focus == FOCUS_DETAILED_REPORT:
            return f"请基于当前活动舌象报告生成更详细的健康参考报告：{raw_text}。报告上下文：{subject}"
        if target_focus == FOCUS_FEATURE_EXPLANATION:
            return f"请基于当前活动舌象报告详细解释相关舌象特征：{raw_text}。报告上下文：{subject}"
        return f"请基于当前活动舌象报告回答：{raw_text}。报告上下文：{subject}"

    if route_hint == ROUTE_HEALTH_QA:
        subject = last_content[:500] or "上一轮健康知识问答"
        if target_focus == FOCUS_DIET_ADVICE:
            return f"请基于上一轮健康问答继续详细说明饮食和日常注意事项：{raw_text}。上一轮内容：{subject}"
        return f"请基于上一轮健康问答内容回答：{raw_text}。上一轮内容：{subject}"

    if route_hint == ROUTE_GENERAL_CHAT:
        subject = last_content[:300] or "上一轮普通聊天"
        return f"请基于上一轮普通聊天内容回答：{raw_text}。上一轮内容：{subject}"

    return raw_text


def _resolve_reference(state: AgentState) -> dict[str, Any]:
    rewrite_context = query_rewrite_context_from_state(state)
    raw_text = str(rewrite_context.get("raw_user_input") or extract_user_text(state))
    last_answer_value = rewrite_context.get("last_final_answer")
    last_answer = last_answer_value if isinstance(last_answer_value, dict) else None
    active_report_value = rewrite_context.get("active_report_ref")
    active_report = active_report_value if isinstance(active_report_value, dict) else None

    if _is_context_dependent_input(raw_text) and not last_answer and not active_report:
        return {
            "status": "MISSING_CONTEXT",
            "target_type": REF_UNKNOWN,
            "target_focus": FOCUS_UNKNOWN,
            "is_context_dependent": True,
            "confidence": 0.0,
            "rule_confidence": 0.0,
            "route_hint": None,
            "reason": "context_dependent_input_without_required_short_term_materials",
            "evidence_sources": ["context_health"],
        }

    last_route_hint = _last_answer_route_hint(last_answer)
    target_focus = _resolve_focus(raw_text, last_answer)

    if _mentions_tongue_analysis_start(raw_text) and not _is_short_followup(raw_text):
        return {
            "status": "NOT_NEEDED",
            "target_type": REF_GENERAL_TOPIC,
            "target_focus": FOCUS_UNKNOWN,
            "is_context_dependent": False,
            "confidence": 0.9,
            "rule_confidence": 0.9,
            "route_hint": ROUTE_TONGUE_ANALYSIS,
            "reason": "用户明确表达舌象图片分析意图",
            "evidence_sources": ["raw_user_input"],
        }

    if _mentions_report(raw_text) and active_report:
        return {
            "status": "RESOLVED",
            "target_type": REF_ACTIVE_REPORT,
            "target_focus": target_focus,
            "is_context_dependent": True,
            "confidence": 0.94,
            "rule_confidence": 0.94,
            "route_hint": ROUTE_REPORT_FOLLOWUP,
            "reason": "用户明确提到报告/舌象/图片等报告相关对象",
            "target_report_id": active_report.get("report_id"),
            "evidence_sources": ["raw_user_input", "active_report_ref"],
        }

    if (
        active_report
        and _mentions_health_qa_followup(raw_text)
        and target_focus == FOCUS_DIET_ADVICE
        and last_route_hint != ROUTE_HEALTH_QA
        and _answer_type(last_answer) != "HEALTH_QA"
    ):
        return {
            "status": "RESOLVED",
            "target_type": REF_ACTIVE_REPORT,
            "target_focus": target_focus,
            "is_context_dependent": True,
            "confidence": 0.86,
            "rule_confidence": 0.86,
            "route_hint": ROUTE_REPORT_FOLLOWUP,
            "reason": "用户提出饮食/调理类追问，当前存在可信活动报告",
            "target_report_id": active_report.get("report_id"),
            "evidence_sources": ["raw_user_input", "active_report_ref"],
        }

    if active_report and _is_short_followup(raw_text) and not last_route_hint:
        return {
            "status": "RESOLVED",
            "target_type": REF_ACTIVE_REPORT,
            "target_focus": target_focus,
            "is_context_dependent": True,
            "confidence": 0.84,
            "rule_confidence": 0.84,
            "route_hint": ROUTE_REPORT_FOLLOWUP,
            "reason": "短追问且当前存在可信活动报告",
            "target_report_id": active_report.get("report_id"),
            "evidence_sources": ["raw_user_input", "active_report_ref"],
        }

    if _is_short_followup(raw_text) and last_route_hint:
        is_item_ref = _contains_any(raw_text, ["第一个", "第二个", "第三个", "这一项", "那一项"])
        return {
            "status": "RESOLVED",
            "target_type": REF_LAST_ANSWER_ITEM if is_item_ref else REF_LAST_ANSWER,
            "target_focus": target_focus,
            "is_context_dependent": True,
            "confidence": 0.64 if is_item_ref else 0.88,
            "rule_confidence": 0.64 if is_item_ref else 0.88,
            "route_hint": last_route_hint,
            "reason": "短追问优先继承上一轮最终回答的上下文对象",
            "target_message_id": (last_answer or {}).get("message_id"),
            "target_report_id": (last_answer or {}).get("report_id"),
            "evidence_sources": ["raw_user_input", "last_final_answer"],
        }

    if _mentions_health_qa_followup(raw_text) and last_route_hint == ROUTE_HEALTH_QA:
        return {
            "status": "RESOLVED",
            "target_type": REF_LAST_ANSWER,
            "target_focus": target_focus,
            "is_context_dependent": True,
            "confidence": 0.88,
            "rule_confidence": 0.88,
            "route_hint": ROUTE_HEALTH_QA,
            "reason": "用户追问饮食/建议等内容，上一轮是健康问答",
            "target_message_id": (last_answer or {}).get("message_id"),
            "evidence_sources": ["raw_user_input", "last_final_answer"],
        }

    if _mentions_health_qa_followup(raw_text) and last_route_hint == ROUTE_REPORT_FOLLOWUP and active_report:
        return {
            "status": "RESOLVED",
            "target_type": REF_ACTIVE_REPORT,
            "target_focus": target_focus,
            "is_context_dependent": True,
            "confidence": 0.88,
            "rule_confidence": 0.88,
            "route_hint": ROUTE_REPORT_FOLLOWUP,
            "reason": "用户追问建议类内容，上一轮和活动报告均指向报告",
            "target_message_id": (last_answer or {}).get("message_id"),
            "target_report_id": active_report.get("report_id"),
            "evidence_sources": ["raw_user_input", "last_final_answer", "active_report_ref"],
        }

    return {
        "status": "NOT_NEEDED",
        "target_type": REF_GENERAL_TOPIC,
        "target_focus": target_focus,
        "is_context_dependent": False,
        "confidence": 1.0,
        "rule_confidence": 1.0,
        "route_hint": None,
        "reason": "未发现明确上下文依赖对象",
        "evidence_sources": [],
    }


def _high_confidence_threshold(state: AgentState) -> float:
    options = state.get("options") or {}
    query_rewrite_options = options.get("query_rewrite") if isinstance(options, dict) else {}
    if isinstance(query_rewrite_options, dict):
        try:
            value = float(query_rewrite_options.get("high_confidence_threshold"))
        except (TypeError, ValueError):
            value = DEFAULT_HIGH_CONFIDENCE_THRESHOLD
        return min(1.0, max(0.0, value))
    return DEFAULT_HIGH_CONFIDENCE_THRESHOLD


def _available_message_ids(rewrite_context: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for item in rewrite_context.get("recent_messages") or []:
        if not isinstance(item, dict):
            continue
        for key in ("message_id", "external_message_id"):
            value = item.get(key)
            if value is not None:
                ids.add(str(value))
    last_answer = rewrite_context.get("last_final_answer")
    if isinstance(last_answer, dict):
        for key in ("message_id", "external_message_id"):
            value = last_answer.get(key)
            if value is not None:
                ids.add(str(value))
    return ids


def _active_report_id(rewrite_context: dict[str, Any]) -> str | None:
    active_report = rewrite_context.get("active_report_ref")
    if not isinstance(active_report, dict):
        return None
    report_id = active_report.get("report_id") or active_report.get("id")
    return str(report_id) if report_id is not None else None


def _ambiguous_reference(reason: str) -> dict[str, Any]:
    return {
        "status": "AMBIGUOUS",
        "target_type": REF_AMBIGUOUS,
        "target_focus": FOCUS_UNKNOWN,
        "is_context_dependent": True,
        "confidence": 0.0,
        "reason": reason,
        "evidence_sources": [],
    }


def _validate_llm_output(
    value: dict[str, Any],
    *,
    rewrite_context: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        output = LLMRewriteOutput.model_validate(value)
    except ValidationError:
        return None

    reference = output.reference_resolution.model_dump(mode="json", exclude_none=True)
    if reference.get("target_type") not in REFERENCE_TARGET_TYPES:
        return None
    if output.route_hint is not None and output.route_hint not in ALLOWED_ROUTE_HINTS:
        return None

    evidence_sources = reference.get("evidence_sources") or []
    if any(source not in EVIDENCE_SOURCES for source in evidence_sources):
        return None

    target_message_id = reference.get("target_message_id")
    if target_message_id is not None and str(target_message_id) not in _available_message_ids(rewrite_context):
        return None

    target_report_id = reference.get("target_report_id")
    trusted_report_id = _active_report_id(rewrite_context)
    if target_report_id is not None and str(target_report_id) != trusted_report_id:
        return None

    reference["confidence"] = min(
        float(reference.get("confidence") or 0.0),
        float(output.rewrite_confidence or 0.0),
    )
    return {
        "standalone_query": output.standalone_query.strip(),
        "reference_resolution": reference,
        "route_hint": output.route_hint,
        "rewrite_confidence": float(output.rewrite_confidence or 0.0),
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _llm_prompt_payload(
    *,
    raw_text: str,
    rule_reference: dict[str, Any],
    rewrite_context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": "rewrite_context_dependent_query",
        "raw_user_input": raw_text,
        "rule_candidate": rule_reference,
        "allowed_reference_target_types": sorted(REFERENCE_TARGET_TYPES),
        "allowed_route_hints": sorted(ALLOWED_ROUTE_HINTS),
        "allowed_evidence_sources": sorted(EVIDENCE_SOURCES),
        "context": {
            "recent_messages": rewrite_context.get("recent_messages") or [],
            "last_final_answer": rewrite_context.get("last_final_answer"),
            "conversation_summary": rewrite_context.get("conversation_summary"),
            "active_report_ref": rewrite_context.get("active_report_ref"),
            "context_health": rewrite_context.get("context_health") or {},
        },
        "output_schema": {
            "standalone_query": "string",
            "route_hint": "one allowed_route_hints or null",
            "rewrite_confidence": "0..1",
            "reference_resolution": {
                "status": "RESOLVED | AMBIGUOUS | NOT_NEEDED",
                "target_type": "one allowed_reference_target_types",
                "target_focus": "string",
                "is_context_dependent": "boolean",
                "confidence": "0..1",
                "target_message_id": "existing message id or null",
                "target_report_id": "trusted active report id or null",
                "evidence_sources": "allowed evidence source names only",
                "reason": "short string",
            },
        },
    }


async def _generate_with_model(messages: list[dict[str, Any]]) -> str:
    from app.core.config import get_settings
    from app.integrations.model_gateway import get_chat_model_client

    settings = get_settings()
    return await get_chat_model_client().generate(
        messages=messages,
        temperature=0.0,
        max_tokens=min(settings.chat_model_max_tokens, 800),
    )


async def _llm_fallback(
    *,
    raw_text: str,
    rule_reference: dict[str, Any],
    rewrite_context: dict[str, Any],
) -> dict[str, Any] | None:
    messages = [
        {
            "role": "system",
            "content": QUERY_REWRITE_SYSTEM_PROMPT + "\n只返回一个 JSON 对象。",
        },
        {
            "role": "user",
            "content": json.dumps(
                _llm_prompt_payload(
                    raw_text=raw_text,
                    rule_reference=rule_reference,
                    rewrite_context=rewrite_context,
                ),
                ensure_ascii=False,
            ),
        },
    ]
    try:
        raw = await _generate_with_model(messages)
    except Exception:
        return None
    payload = _extract_json_object(raw)
    if payload is None:
        return None
    return _validate_llm_output(payload, rewrite_context=rewrite_context)


def _query_context_still_current(
    *,
    current_turn: dict[str, Any],
    query_context: dict[str, Any],
) -> bool:
    rewrite_context = current_turn.get("query_rewrite_context") or {}
    return (
        query_context.get("turn_id") == current_turn.get("turn_id")
        and query_context.get("context_version") == rewrite_context.get("context_version")
    )


async def query_rewrite_node(state: AgentState) -> AgentState:
    current_turn = ensure_current_turn(state)
    existing_query_context = current_turn.get("query_context")
    if (
        isinstance(existing_query_context, dict)
        and existing_query_context
        and _query_context_still_current(
            current_turn=current_turn,
            query_context=existing_query_context,
        )
    ):
        return {
            **state,
            "current_turn": current_turn,
            "current_node": "query_rewrite_node",
            "query_context": existing_query_context,
            "prompt_context": current_turn.get("query_rewrite_context") or {},
        }

    rewrite_context = query_rewrite_context_from_state(
        {**state, "current_turn": current_turn}
    )
    if (
        rewrite_context.get("turn_id") != current_turn.get("turn_id")
        or not rewrite_context.get("context_version")
    ):
        rewrite_context = build_query_rewrite_context({**state, "current_turn": current_turn})
        current_turn["query_rewrite_context"] = rewrite_context
        current_turn["context_health"] = rewrite_context.get("context_health") or {}
    elif not current_turn.get("query_rewrite_context"):
        current_turn["query_rewrite_context"] = rewrite_context
        current_turn["context_health"] = rewrite_context.get("context_health") or {}
    state = {**state, "current_turn": current_turn}

    raw_text = str(rewrite_context.get("raw_user_input") or extract_user_text(state))
    last_answer = rewrite_context.get("last_final_answer")
    active_report = rewrite_context.get("active_report_ref")
    # 指代消解
    threshold = _high_confidence_threshold(state)
    rule_reference = _resolve_reference(state)
    reference = rule_reference
    # 提取结果中的目标类型
    route_hint = reference.get("route_hint")
    # 提取指代焦点
    target_focus = str(reference.get("target_focus") or FOCUS_UNKNOWN)
    clarification_status = (
        "NEEDS_CLARIFICATION"
        if reference.get("status") == "MISSING_CONTEXT"
        else "NOT_NEEDED"
    )
    llm_result: dict[str, Any] | None = None
    if clarification_status != "NEEDS_CLARIFICATION" and float(reference.get("rule_confidence") or 0.0) < threshold:
        llm_result = await _llm_fallback(
            raw_text=raw_text,
            rule_reference=rule_reference,
            rewrite_context=rewrite_context,
        )
        if llm_result is None:
            clarification_status = "NEEDS_CLARIFICATION"
            reference = _ambiguous_reference("low_confidence_rule_and_llm_fallback_failed")
            route_hint = None
        else:
            reference = llm_result["reference_resolution"]
            target_focus = str(reference.get("target_focus") or FOCUS_UNKNOWN)
            route_hint = llm_result.get("route_hint")
    # 生成可以独立理解的问题，不依赖上下文
    if clarification_status == "NEEDS_CLARIFICATION":
        standalone_query = raw_text
    elif llm_result is not None:
        standalone_query = llm_result["standalone_query"] or raw_text
    else:
        standalone_query = _standalone_for_target(
            raw_text=raw_text,
            route_hint=route_hint if isinstance(route_hint, str) else None,
            target_focus=target_focus,
            last_answer=last_answer if isinstance(last_answer, dict) else None,
            active_report=active_report if isinstance(active_report, dict) else None,
        )
    # 路由建议


    # 重写后的查询字典
    query_context = {
        "schema_version": "1.0",
        "turn_id": current_turn.get("turn_id"),
        "context_version": rewrite_context.get("context_version"),
        "raw_user_input": raw_text,
        "standalone_query": standalone_query,
        "reference_resolution": reference,
        "clarification_status": clarification_status,
        "route_hint": route_hint if route_hint in ALLOWED_ROUTE_HINTS else None,
        "rewrite_confidence": float(reference.get("confidence") or 0.0),
        "rule_confidence": float(rule_reference.get("rule_confidence") or 0.0),
        "resolution_strategy": (
            "CLARIFICATION"
            if clarification_status == "NEEDS_CLARIFICATION"
            else "RULE"
            if float(rule_reference.get("rule_confidence") or 0.0) >= threshold
            else "LLM_FALLBACK"
        ),
        "high_confidence_threshold": threshold,
        "prompt_context": rewrite_context,
    }
    next_state = with_query_context(state, query_context)

    return {
        **next_state,
        "current_node": "query_rewrite_node",
        "prompt_context": rewrite_context,
    }
