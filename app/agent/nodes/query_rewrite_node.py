import re
from typing import Any

from app.agent.context_builder import (
    active_report_from_state,
    build_prompt_context,
    extract_user_text,
    last_final_answer_from_state,
)
from app.agent.state import AgentState


TARGET_REPORT = "REPORT"
TARGET_HEALTH_QA = "HEALTH_QA"
TARGET_GENERAL_CHAT = "GENERAL_CHAT"
TARGET_TONGUE_ANALYSIS = "TONGUE_ANALYSIS"
TARGET_UNKNOWN = "UNKNOWN"

FOCUS_DIET_ADVICE = "DIET_ADVICE"
FOCUS_DETAILED_REPORT = "DETAILED_REPORT"
FOCUS_GENERAL_DETAIL = "GENERAL_DETAIL"
FOCUS_FEATURE_EXPLANATION = "FEATURE_EXPLANATION"
FOCUS_UNKNOWN = "UNKNOWN"


ROUTE_HINTS = {
    TARGET_REPORT: "report_followup_subgraph",
    TARGET_HEALTH_QA: "health_qa_subgraph",
    TARGET_GENERAL_CHAT: "general_chat_subgraph",
    TARGET_TONGUE_ANALYSIS: "tongue_analysis_subgraph",
}

# TODO 提示词重写，不符合要求
QUERY_REWRITE_SYSTEM_PROMPT = """你是中医舌象健康 Agent 的追问消解器。
你的职责是把依赖上下文的问题改写为独立问题，并判断它依赖的是报告、健康问答、普通聊天还是舌象分析流程。
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


def _last_answer_target(last_answer: dict[str, Any] | None) -> str:
    node_name = _node_name(last_answer)
    answer_type = _answer_type(last_answer)
    route_target = str((last_answer or {}).get("route_target") or "")

    if node_name in {"health_qa_node"} or answer_type == "HEALTH_QA" or route_target == "health_qa_subgraph":
        return TARGET_HEALTH_QA

    if (
        node_name in {"tongue_report_node", "report_followup_node"}
        or answer_type in {"TONGUE_REPORT", "REPORT_FOLLOWUP", "DETAILED_TONGUE_REPORT"}
        or route_target in {"tongue_analysis_subgraph", "report_followup_subgraph"}
        or (last_answer or {}).get("report_id")
    ):
        return TARGET_REPORT

    if node_name == "tongue_analysis_node":
        return TARGET_TONGUE_ANALYSIS

    if node_name == "general_chat_node" or answer_type == "GENERAL_CHAT":
        return TARGET_GENERAL_CHAT

    return TARGET_UNKNOWN


def _standalone_for_target(
    *,
    raw_text: str,
    target_type: str,
    target_focus: str,
    last_answer: dict[str, Any] | None,
    active_report: dict[str, Any] | None,
) -> str:
    raw_text = raw_text.strip()
    last_content = str((last_answer or {}).get("content") or "").strip()

    if target_type == TARGET_REPORT:
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

    if target_type == TARGET_HEALTH_QA:
        subject = last_content[:500] or "上一轮健康知识问答"
        if target_focus == FOCUS_DIET_ADVICE:
            return f"请基于上一轮健康问答继续详细说明饮食和日常注意事项：{raw_text}。上一轮内容：{subject}"
        return f"请基于上一轮健康问答内容回答：{raw_text}。上一轮内容：{subject}"

    if target_type == TARGET_GENERAL_CHAT:
        subject = last_content[:300] or "上一轮普通聊天"
        return f"请基于上一轮普通聊天内容回答：{raw_text}。上一轮内容：{subject}"

    return raw_text


def _resolve_reference(state: AgentState) -> dict[str, Any]:
    raw_text = extract_user_text(state)
    last_answer = last_final_answer_from_state(state)
    active_report = active_report_from_state(state)
    last_target = _last_answer_target(last_answer)
    target_focus = _resolve_focus(raw_text, last_answer)

    if _mentions_tongue_analysis_start(raw_text) and not _is_short_followup(raw_text):
        return {
            "target_type": TARGET_TONGUE_ANALYSIS,
            "target_focus": FOCUS_UNKNOWN,
            "is_context_dependent": False,
            "confidence": 0.9,
            "reason": "用户明确表达舌象图片分析意图",
        }

    if _mentions_report(raw_text) and active_report:
        return {
            "target_type": TARGET_REPORT,
            "target_focus": target_focus,
            "is_context_dependent": True,
            "confidence": 0.94,
            "reason": "用户明确提到报告/舌象/图片等报告相关对象",
            "target_report_id": active_report.get("report_id"),
        }

    if _is_short_followup(raw_text) and last_target != TARGET_UNKNOWN:
        return {
            "target_type": last_target,
            "target_focus": target_focus,
            "is_context_dependent": True,
            "confidence": 0.86,
            "reason": "短追问优先继承上一轮最终回答的上下文对象",
            "target_message_id": (last_answer or {}).get("message_id"),
            "target_report_id": (last_answer or {}).get("report_id"),
        }

    if _mentions_health_qa_followup(raw_text) and last_target == TARGET_HEALTH_QA:
        return {
            "target_type": TARGET_HEALTH_QA,
            "target_focus": target_focus,
            "is_context_dependent": True,
            "confidence": 0.82,
            "reason": "用户追问饮食/建议等内容，上一轮是健康问答",
            "target_message_id": (last_answer or {}).get("message_id"),
        }

    if _mentions_health_qa_followup(raw_text) and last_target == TARGET_REPORT and active_report:
        return {
            "target_type": TARGET_REPORT,
            "target_focus": target_focus,
            "is_context_dependent": True,
            "confidence": 0.82,
            "reason": "用户追问建议类内容，上一轮和活动报告均指向报告",
            "target_message_id": (last_answer or {}).get("message_id"),
            "target_report_id": active_report.get("report_id"),
        }

    return {
        "target_type": TARGET_UNKNOWN,
        "target_focus": target_focus,
        "is_context_dependent": False,
        "confidence": 0.0,
        "reason": "未发现明确上下文依赖对象",
    }


async def query_rewrite_node(state: AgentState) -> AgentState:
    # 构造提示词，用户查询重写
    prompt_context = build_prompt_context(
        state,
        mode="MINIMAL_PRE_INTENT",
        system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
        node_name="query_rewrite_node",
        include_long_term_memory=False,
    )
    # 提取用户文本消息
    raw_text = extract_user_text(state)
    # 提取上一轮答案
    last_answer = prompt_context.get("last_final_answer")
    # 提取会话讨论报告
    active_report = prompt_context.get("active_report")
    # 指代消解
    reference = _resolve_reference(state)
    # 提取结果中的目标类型
    target_type = reference.get("target_type") or TARGET_UNKNOWN
    # 提取指代焦点
    target_focus = str(reference.get("target_focus") or FOCUS_UNKNOWN)
    # 生成可以独立理解的问题，不依赖上下文
    standalone_query = _standalone_for_target(
        raw_text=raw_text,
        target_type=target_type,
        target_focus=target_focus,
        last_answer=last_answer if isinstance(last_answer, dict) else None,
        active_report=active_report if isinstance(active_report, dict) else None,
    )
    # 路由建议
    route_hint = ROUTE_HINTS.get(str(target_type))


    # 重写后的查询字典
    query_context = {
        "schema_version": "1.0",
        "raw_user_input": raw_text,
        "standalone_query": standalone_query,
        "reference_resolution": reference,
        "route_hint": route_hint,
        "prompt_context": prompt_context,
    }

    return {
        **state,
        "current_node": "query_rewrite_node",
        "query_context": query_context,
        "prompt_context": prompt_context,
    }
