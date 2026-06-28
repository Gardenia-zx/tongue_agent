import json
import re
from typing import Any

from app.agent.context_builder import current_turn_from_state, effective_user_query, with_prompt_context
from app.agent.nodes.rag_node_utils import answer_with_rag
from app.agent.state import AgentState
from app.core.config import get_settings


def get_chat_model_client():
    from app.integrations.model_gateway import get_chat_model_client as _get_chat_model_client

    return _get_chat_model_client()


FOLLOWUP_REPORT_PROMPT = """你是舌象健康报告的追问助手。
用户正在追问前面已经生成的舌象报告，你必须基于本轮 prompt_context.active_report 中已加载的报告章节展开回答。

要求：
1. 不要说没有看到图片。
2. 不要要求用户重新上传图片。
3. 不要把用户的问题当成新的舌象分析任务。
4. 不输出疾病确诊结论，不替代医生判断。
5. 药物、方剂、处方类问题可以做一般知识参考：说明常见方向、适用边界、禁忌和就医沟通要点。
6. 不要建议用户自行停药、换药、加药或减药；涉及具体剂量、孕期、儿童、老人、慢病或正在用药时提醒医生或药师确认。
7. 直接回答用户关心的结果和建议，语气自然。
8. 不要输出 Markdown 标题符号，不要输出分隔线。
9. 如果提供了 rag_context，要结合知识库检索结果给出一般健康管理参考。
10. 如果用户要求“更详细的报告、完整报告、重新生成详细报告”，要输出一份完整的舌象健康参考报告，不要只给几个观察点。
11. content 只写 1 到 3 句摘要，详细内容只放在 structured_content.sections，避免重复输出。
12. 如果用户要求饮食推荐、运动推荐或详细建议，sections 必须包含“饮食建议”和“每日运动建议”。
13. 返回 JSON：
{
  "content": "给用户看的中文回答",
  "structured_content": {
    "schema_version": "1.0",
    "answer_type": "REPORT_FOLLOWUP 或 DETAILED_TONGUE_REPORT",
    "title": "报告追问 或 详细舌象报告",
    "summary": "直接回答用户关心的结果。若用户要详细报告，这里写完整报告摘要，2 到 4 句话",
    "highlights": ["可选重点标签"],
    "sections": [
      {"title": "识别结果", "items": ["基于本轮已加载报告章节的图像识别结果"]},
      {"title": "舌象说明", "content": "解释舌象特征的一般健康含义"},
      {"title": "可能相关表现", "items": ["结合用户描述和最近消息"]},
      {"title": "饮食建议", "items": ["最多 5 条，写具体吃法和忌口"]},
      {"title": "每日运动建议", "items": ["最多 5 条，写每天或每周怎么运动、强度和注意事项"]},
      {"title": "继续观察", "items": ["最多 5 条"]}
    ],
    "disclaimer": "安全提醒"
  }
}
"""


def extract_user_text(state: AgentState) -> str:
    message: dict[str, Any] = state.get("message", {})
    content = message.get("content")
    return content.strip() if isinstance(content, str) else ""


def context_bundle_from_state(state: AgentState) -> dict[str, Any]:
    context_bundle = state.get("context_bundle") or {}
    if isinstance(context_bundle, dict) and context_bundle:
        return context_bundle

    client_context = state.get("client_context") or {}
    client_extra = client_context.get("extra") or {}
    extra_bundle = client_extra.get("context_bundle") or {}
    return extra_bundle if isinstance(extra_bundle, dict) else {}


def latest_report_from_state(state: AgentState) -> dict[str, Any] | None:
    current_turn = current_turn_from_state(state)
    for container_name in ("final_prompt_context", "business_context"):
        container = current_turn.get(container_name) or {}
        if not isinstance(container, dict):
            continue
        active_report = container.get("active_report")
        if isinstance(active_report, dict) and active_report.get("sections"):
            return active_report
        loaded = container.get("loaded_report_sections")
        if isinstance(loaded, dict) and loaded.get("sections"):
            return loaded
        fallback_ref = _trusted_report_ref(container.get("active_report_ref"))
        if fallback_ref is not None:
            return fallback_ref
    for container in (
        state.get("prompt_context"),
        current_turn.get("query_rewrite_context"),
        state.get("query_rewrite_context"),
        context_bundle_from_state(state),
    ):
        if isinstance(container, dict):
            fallback_ref = _trusted_report_ref(container.get("active_report_ref"))
            if fallback_ref is not None:
                return fallback_ref
    return None


def _trusted_report_ref(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or value.get("report_id") is None:
        return None
    if value.get("trusted") is not True and value.get("is_current_active_report") is not True:
        return None
    if not (value.get("summary") or value.get("feature_summary") or value.get("featureSummary")):
        return None
    return value


def report_context_error_from_state(state: AgentState) -> dict[str, Any] | None:
    current_turn = current_turn_from_state(state)
    business_context = current_turn.get("business_context") or {}
    if not isinstance(business_context, dict):
        return None
    error = business_context.get("report_context_error")
    return error if isinstance(error, dict) else None


def recent_messages_from_state(state: AgentState) -> list[dict[str, Any]]:
    context_bundle = context_bundle_from_state(state)
    recent_messages = context_bundle.get("recent_messages") or []
    if isinstance(recent_messages, list) and recent_messages:
        return recent_messages

    client_context = state.get("client_context") or {}
    client_extra = client_context.get("extra") or {}
    recent_messages = client_extra.get("recent_messages") or []
    return recent_messages if isinstance(recent_messages, list) else []


def last_assistant_content(recent_messages: Any) -> str:
    if not isinstance(recent_messages, list):
        return ""

    for item in reversed(recent_messages):
        if not isinstance(item, dict):
            continue
        if item.get("role") != "assistant":
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()[:1200]
    return ""


def is_short_followup_request(text: str) -> bool:
    compact = "".join(text.split())
    if not compact:
        return False
    followup_patterns = [
        "回答详细一点",
        "详细一点",
        "再详细一点",
        "说详细点",
        "展开说说",
        "继续说",
        "继续",
        "具体一点",
        "再具体点",
        "多说一点",
        "不够详细",
        "太简单",
    ]
    return any(pattern in compact for pattern in followup_patterns)


def is_report_followup_request(text: str) -> bool:
    compact = "".join(text.split())
    if not compact:
        return False
    patterns = [
        r"(前面|刚才|上面|上次|上一份|最近).*?(报告|舌象分析|分析).*?(不够详细|太简单|详细|具体|展开|补充)",
        r"(我的|这份|这个|刚才|前面|上面|上次).*?(舌象分析|报告|分析).*?(详细|具体|展开|补充|再说)",
        r"(舌象分析|报告|分析).*?(不够详细|太简单|详细一点|详细点|具体一点|具体点|展开说)",
        r"(讲|说|回答|分析).*?(详细一点|详细点|具体一点|具体点|展开说说)",
        r"(更详细|详细|完整|重新|生成).*?(报告|舌象报告|分析报告)",
        r"(报告|舌象报告|分析报告).*?(更详细|详细|完整|重新生成)",
        r"(饮食|吃什么|怎么吃|忌口|食物|推荐|建议|调理|注意).*?(推荐|建议|详细|具体|怎么|哪些|一点)?",
    ]
    return any(re.search(pattern, compact) for pattern in patterns) or is_short_followup_request(text)


def has_report_followup_context(state: AgentState) -> bool:
    return bool(
        latest_report_from_state(state)
        or last_assistant_content(recent_messages_from_state(state))
    )


def needs_report_followup_rag(text: str) -> bool:
    compact = "".join(text.split())
    if not compact:
        return False
    if is_detailed_report_request(text):
        return True
    keywords = [
        "饮食",
        "吃什么",
        "怎么吃",
        "忌口",
        "食物",
        "推荐",
        "建议",
        "调理",
        "运动",
        "锻炼",
        "快走",
        "跑步",
        "健身",
        "注意",
        "怎么办",
    ]
    return any(keyword in compact for keyword in keywords)


def is_detailed_report_request(text: str) -> bool:
    compact = "".join(text.split())
    if not compact:
        return False
    if is_diet_or_care_followup(text) and not any(
        keyword in compact
        for keyword in ["详细报告", "完整报告", "更详细的报告", "重新生成报告"]
    ):
        return False
    patterns = [
        r"(给我|生成|重新生成|写|出).*?(更详细|详细|完整).*?(报告|舌象报告|分析报告)",
        r"(更详细|详细|完整).*?(报告|舌象报告|分析报告)",
        r"(报告|舌象报告|分析报告).*?(更详细|详细|完整)",
    ]
    return any(re.search(pattern, compact) for pattern in patterns)


def is_diet_or_care_followup(text: str) -> bool:
    compact = "".join(text.split())
    if not compact:
        return False
    keywords = [
        "饮食",
        "吃什么",
        "怎么吃",
        "忌口",
        "食物",
        "推荐",
        "建议",
        "调理",
        "运动",
        "锻炼",
        "快走",
        "跑步",
        "健身",
    ]
    return any(keyword in compact for keyword in keywords)


def wants_exercise_advice(text: str) -> bool:
    compact = "".join(text.split())
    return any(keyword in compact for keyword in ("运动", "锻炼", "快走", "跑步", "健身", "活动"))


def _exercise_section() -> dict[str, Any]:
    return {
        "title": "每日运动建议",
        "items": [
            "每天安排 20 到 30 分钟轻到中等强度运动，如快走、八段锦、拉伸或轻力量训练。",
            "以微微出汗、呼吸略加快但还能说话为度，不追求一次性高强度。",
            "饭后不要立刻剧烈运动，可在饭后 30 分钟左右散步 10 到 15 分钟。",
            "每周保持 3 到 5 次规律运动；如果当天疲乏、睡眠差，就降低强度或改为拉伸。",
            "运动后观察睡眠、疲劳、腹胀和食欲变化，连续记录比单次感受更有参考价值。",
        ],
    }


def build_followup_rag_query(
    *,
    user_text: str,
    latest_report: dict[str, Any] | None,
    context_bundle: dict[str, Any],
) -> str:
    parts = [user_text]
    if latest_report:
        feature_summary = str(
            latest_report.get("feature_summary")
            or latest_report.get("featureSummary")
            or ""
        ).strip()
        summary = str(latest_report.get("summary") or "").strip()
        if feature_summary:
            parts.append(feature_summary)
        if summary:
            parts.append(summary[:600])

    conversation_summary = context_bundle.get("conversation_summary")
    if isinstance(conversation_summary, dict):
        summary_text = str(conversation_summary.get("text") or "").strip()
        if summary_text:
            parts.append(summary_text[:500])

    if is_detailed_report_request(user_text):
        parts.append("中医 舌象 详细报告 舌苔 舌质 健康管理 继续观察")
    else:
        parts.append("中医 舌象 饮食 运动 健康管理 一般建议")
    return " ".join(part for part in parts if part).strip()


def _build_detailed_report_sections(
    *,
    latest_report: dict[str, Any] | None,
    report_summary: str,
    feature_summary: str,
    rag_context: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rag_answer = clean_text((rag_context or {}).get("answer"), max_length=380)
    feature_text = feature_summary or "前面报告已生成舌象识别结果，但当前上下文没有拿到完整特征摘要。"
    summary_text = report_summary or "当前可基于前面报告继续展开，但详细程度仍受图像模型返回特征范围限制。"

    return [
        {
            "title": "识别结果",
            "items": [
                feature_text,
                "当前图像结果适合作为健康管理参考，需要结合舌苔厚薄、润燥、质地和近期身体感受一起理解。",
            ],
        },
        {
            "title": "舌象说明",
            "content": (
                rag_answer
                or "白苔是常见舌苔表现。薄白而润可见于正常舌象；如果明显偏厚、偏腻或偏滑，通常需要结合饮食、消化、睡眠、口腔清洁和身体感受继续观察。"
            ),
        },
        {
            "title": "结合当前报告",
            "content": summary_text,
        },
        {
            "title": "健康管理建议",
            "items": [
                "近期饮食保持清淡规律，少吃生冷、油腻、甜腻、酒精和夜宵。",
                "保持作息稳定，避免连续熬夜，观察疲乏、睡眠和精神状态是否改善。",
                "饭后不要马上久坐或躺下，可以进行轻度活动，观察腹胀和食欲变化。",
                "保持口腔清洁，尽量在自然光下、同一角度复拍，方便后续对比。",
                "不要仅凭舌象自行用药、停药或调整治疗方案。",
            ],
        },
        _exercise_section(),
        {
            "title": "继续观察",
            "items": [
                "舌苔是薄白还是厚白，是否有腻、滑、干、剥落等变化。",
                "食欲、腹胀、大便、口腻、疲乏和睡眠是否同步变化。",
                "近期饮食油腻、生冷、饮酒、熬夜后舌苔是否更明显。",
                "如果舌象变化持续存在，或伴随明显不适，应咨询专业医生。",
            ],
        },
    ]


def compose_followup_fallback(
    *,
    user_text: str,
    latest_report: dict[str, Any] | None,
    recent_messages: list[dict[str, Any]],
    rag_context: dict[str, Any] | None,
) -> str:
    last_answer = last_assistant_content(recent_messages)
    report_summary = ""
    feature_summary = ""
    if latest_report:
        report_summary = str(latest_report.get("summary") or "").strip()
        feature_summary = str(
            latest_report.get("feature_summary")
            or latest_report.get("featureSummary")
            or ""
        ).strip()

    if is_detailed_report_request(user_text):
        sections = _build_detailed_report_sections(
            latest_report=latest_report,
            report_summary=report_summary,
            feature_summary=feature_summary,
            rag_context=rag_context,
        )
        content_parts = [
            "可以，我基于前面那份舌象分析整理成一份更详细的健康参考报告。",
        ]
        for section in sections:
            content_parts.append("")
            content_parts.append(section["title"])
            if section.get("content"):
                content_parts.append(str(section["content"]))
            for index, item in enumerate(section.get("items") or [], start=1):
                content_parts.append(f"{index}. {item}")
        content_parts.extend(
            [
                "",
                "提醒",
                "以上内容用于一般健康知识说明和健康管理参考，不能替代医生诊断。",
            ]
        )
        return "\n".join(content_parts)

    if is_diet_or_care_followup(user_text):
        feature_text = f"前面报告主要提到{feature_summary}。" if feature_summary else ""
        rag_answer = clean_text((rag_context or {}).get("answer"), max_length=420)
        knowledge_text = f"\n\n{rag_answer}" if rag_answer else ""
        return (
            f"可以，结合前面那份舌象分析，饮食上先以清淡、规律、少生冷甜腻为主。{feature_text}"
            f"{knowledge_text}\n\n"
            "饮食建议\n"
            "1. 近期先减少生冷、油腻、甜腻、酒精和夜宵，观察腹胀、口腻、大便黏等感受是否减轻。\n"
            "2. 三餐尽量规律，晚餐不要过饱，饭后避免马上久坐或躺下。\n"
            "3. 可以选择相对温和、容易消化的食物，例如粥、面、山药、白扁豆、南瓜等日常食材。\n"
            "4. 如果你本身有基础疾病、孕期、正在用药，或不适持续加重，不建议自行调理，应咨询医生。\n\n"
            "每日运动建议\n"
            "1. 每天安排 20 到 30 分钟轻到中等强度运动，如快走、八段锦、拉伸或轻力量训练。\n"
            "2. 以微微出汗、呼吸略加快但还能说话为度，不追求一次性高强度。\n"
            "3. 饭后不要立刻剧烈运动，可在饭后 30 分钟左右散步 10 到 15 分钟。\n"
            "4. 每周保持 3 到 5 次规律运动；如果当天疲乏、睡眠差，就降低强度或改为拉伸。\n"
            "5. 运动后观察睡眠、疲劳、腹胀和食欲变化，连续记录比单次感受更有参考价值。\n\n"
            "继续观察\n"
            "1. 舌苔是薄白还是厚白，是否发腻或发滑。\n"
            "2. 饭后腹胀、食欲、大便状态和疲乏感是否随饮食变化而改善。\n"
            "3. 同一光线下复拍舌象，观察白苔范围和厚薄是否持续变化。\n\n"
            "以上内容用于一般健康知识说明和健康管理参考，不能替代医生诊断。"
        )

    base = report_summary or last_answer or feature_summary
    if not base:
        return (
            "可以继续展开前面的舌象分析。不过当前请求没有带上可追溯的报告摘要，"
            "需要后端补充当前会话的 active_report 或最近消息后再继续。"
        )

    return (
        "可以，我基于前面那份舌象分析再说细一点。\n\n"
        f"{base}\n\n"
        "你可以重点观察三件事：\n"
        "1. 舌苔是薄白还是厚白，是否有发腻、发滑或发干。\n"
        "2. 近期睡眠、疲乏、食欲、腹胀和大便状态是否同步变化。\n"
        "3. 在自然光下复拍时，舌苔范围和颜色是否持续相似。\n\n"
        "如果这些变化持续存在，或者伴随明显不适，建议咨询专业医生。以上内容只作为一般健康知识参考，不能替代医生诊断。"
    )


def clean_text(value: Any, *, max_length: int = 600) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = text.replace("---", "")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:max_length].strip()


def clean_items(value: Any, *, max_items: int = 4) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = clean_text(item, max_length=180)
        if text and text not in items:
            items.append(text)
        if len(items) >= max_items:
            break
    return items


def normalize_structured_followup(
    value: Any,
    *,
    user_text: str,
    fallback_content: str,
    latest_report: dict[str, Any] | None,
    rag_context: dict[str, Any] | None,
) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    detailed_report = is_detailed_report_request(user_text)
    summary = clean_text(source.get("summary"), max_length=700)
    if not summary:
        if detailed_report:
            summary = "这是基于前面舌象分析整理出的更详细健康参考报告，重点包括识别结果、舌象说明、健康管理建议和后续观察方向。"
        else:
            summary = clean_text(fallback_content.split("\n\n", 1)[0], max_length=700)

    raw_sections = source.get("sections")
    sections: list[dict[str, Any]] = []
    if isinstance(raw_sections, list):
        for raw_section in raw_sections:
            if not isinstance(raw_section, dict):
                continue
            title = clean_text(raw_section.get("title"), max_length=30)
            items = clean_items(raw_section.get("items"), max_items=5 if detailed_report else 4)
            section_content = clean_text(raw_section.get("content"), max_length=800 if detailed_report else 500)
            if not title:
                continue
            if items or section_content:
                section: dict[str, Any] = {"title": title}
                if items:
                    section["items"] = items
                if section_content:
                    section["content"] = section_content
                sections.append(section)
            if len(sections) >= (6 if detailed_report else 3):
                break

    if sections and wants_exercise_advice(user_text) and not any("运动" in str(section.get("title") or "") for section in sections):
        sections.append(_exercise_section())

    if not sections and detailed_report:
        report_summary = ""
        feature_summary = ""
        if latest_report:
            report_summary = str(latest_report.get("summary") or "").strip()
            feature_summary = str(
                latest_report.get("feature_summary")
                or latest_report.get("featureSummary")
                or ""
            ).strip()
        sections = _build_detailed_report_sections(
            latest_report=latest_report,
            report_summary=report_summary,
            feature_summary=feature_summary,
            rag_context=rag_context,
        )

    if not sections and is_diet_or_care_followup(user_text):
        sections = [
            {
                "title": "饮食建议",
                "items": [
                    "饮食先保持清淡、规律，避免一上来做很激烈的调整。",
                    "近期少吃生冷、油腻、甜腻、酒精和夜宵。",
                    "优先选择温和、容易消化的日常食物，观察腹胀、口腻、大便黏是否改善。",
                    "如果不适持续加重，或有基础疾病、孕期、正在用药，应咨询医生。",
                ],
            },
        ]
        if wants_exercise_advice(user_text):
            sections.append(_exercise_section())
        sections.append(
            {
                "title": "继续观察",
                "items": [
                    "舌苔是薄白还是厚白，是否发腻、发滑或发干。",
                    "食欲、饭后腹胀、大便状态和疲乏感是否同步变化。",
                    "同一光线下复拍舌象，看白苔范围和厚薄是否持续。",
                ],
            },
        )

    if not sections:
        sections = [
            {
                "title": "可以重点看",
                "items": [
                    "舌苔厚薄、润燥和是否发腻。",
                    "睡眠、食欲、腹胀、大便和疲乏感是否同步变化。",
                    "同一光线下复拍，观察变化是否持续。",
                ],
            }
        ]

    disclaimer = clean_text(source.get("disclaimer"), max_length=220)
    if not disclaimer:
        disclaimer = "以上内容用于一般健康知识说明和健康管理参考，不能替代医生诊断。"
    if "不能替代医生诊断" not in disclaimer and "不构成医疗建议" not in disclaimer:
        disclaimer += " 以上内容不能替代医生诊断。"

    highlights = clean_items(source.get("highlights"), max_items=3)
    if not highlights and latest_report:
        feature_summary = clean_text(
            latest_report.get("feature_summary") or latest_report.get("featureSummary"),
            max_length=80,
        )
        if feature_summary:
            highlights.append(feature_summary)
    if rag_context is not None and not any("知识库" in item for item in highlights):
        highlights.append("已结合知识库资料")

    source_title = clean_text(source.get("title"), max_length=30)
    if detailed_report and (not source_title or source_title == "报告追问"):
        source_title = "详细舌象报告"

    return {
        "schema_version": "1.0",
        "answer_type": "DETAILED_TONGUE_REPORT" if detailed_report else "REPORT_FOLLOWUP",
        "title": source_title
        or ("饮食建议" if is_diet_or_care_followup(user_text) else "报告追问"),
        "summary": summary,
        "highlights": highlights,
        "sections": sections,
        "disclaimer": disclaimer,
    }


async def generate_report_followup_reply(
    state: AgentState,
) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
    settings = get_settings()
    raw_user_text = extract_user_text(state)
    user_text = effective_user_query(state) or raw_user_text
    latest_report = latest_report_from_state(state)
    recent_messages = recent_messages_from_state(state)
    context_bundle = context_bundle_from_state(state)
    rag_context: dict[str, Any] | None = None

    if latest_report is None:
        content = "暂时无法读取当前舌象报告内容，请稍后重试或重新打开报告后再提问。"
        structured = {
            "schema_version": "1.0",
            "answer_type": "REPORT_CONTEXT_UNAVAILABLE",
            "title": "报告暂不可用",
            "summary": content,
            "highlights": [],
            "sections": [],
            "disclaimer": "未读取到可信报告内容时，我不会基于旧报告生成个性化建议。",
        }
        return content, None, structured

    if needs_report_followup_rag(raw_user_text) or needs_report_followup_rag(user_text):
        rag_context = await answer_with_rag(
            build_followup_rag_query(
                user_text=user_text,
                latest_report=latest_report,
                context_bundle=context_bundle,
            )
        )

    context = {
        "raw_user_input": raw_user_text,
        "standalone_query": user_text,
        "prompt_context": state.get("prompt_context") or {},
        "active_report": latest_report,
        "conversation_summary": context_bundle.get("conversation_summary"),
        "traceback_context": context_bundle.get("traceback_context"),
        "recent_messages": recent_messages[-8:],
        "rag_context": rag_context,
        "instruction": (
            "用户是在追问前面已经生成的舌象报告，请基于上下文展开。"
            "如果 active_report 有 summary，就优先基于它。"
            "如果 rag_context 有命中内容，请结合知识库结果，但不要把回答写成检索过程。"
        ),
    }

    try:
        raw_content = await get_chat_model_client().generate(
            messages=[
                {"role": "system", "content": FOLLOWUP_REPORT_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(context, ensure_ascii=False),
                },
            ],
            temperature=settings.chat_model_temperature,
            max_tokens=settings.chat_model_max_tokens,
        )
        payload = extract_json_object(raw_content)
    except Exception:
        payload = None

    if isinstance(payload, dict):
        content = payload.get("content")
        if isinstance(content, str) and content.strip():
            normalized_content = clean_text(content, max_length=1800)
            return (
                normalized_content,
                rag_context,
                normalize_structured_followup(
                    payload.get("structured_content"),
                    user_text=user_text,
                    fallback_content=normalized_content,
                    latest_report=latest_report,
                    rag_context=rag_context,
                ),
            )

    fallback_content = compose_followup_fallback(
        user_text=user_text,
        latest_report=latest_report,
        recent_messages=recent_messages,
        rag_context=rag_context,
    )
    return (
        fallback_content,
        rag_context,
        normalize_structured_followup(
            None,
            user_text=user_text,
            fallback_content=fallback_content,
            latest_report=latest_report,
            rag_context=rag_context,
        ),
    )


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    return data if isinstance(data, dict) else None


async def report_followup_node(state: AgentState) -> AgentState:
    state = with_prompt_context(
        state,
        mode="FULL_FOR_NODE",
        system_prompt=FOLLOWUP_REPORT_PROMPT,
        node_name="report_followup_node",
        include_long_term_memory=True,
    )
    intent_result = state.get("intent_result") or {}
    content, rag_context, structured_content = await generate_report_followup_reply(state)
    tool_decision = {
        "need_rag": rag_context is not None,
        "need_web_search": False,
        "need_tongue_analysis": False,
        "suggested_next_node": "report_followup_node",
        "reason": "report_followup_with_context",
    }
    quality_review = {
        "answerable_without_tool": True,
        "needs_user_choice": False,
        "risk_note": intent_result.get("risk_level", "LOW"),
    }

    return {
        **state,
        "current_node": "report_followup_node",
        "rag_context": rag_context or state.get("rag_context"),
        "tool_decision": tool_decision,
        "quality_review": quality_review,
        "response_message": {
            "role": "assistant",
            "content_type": "text",
            "content": content,
            "structured_content": structured_content,
        },
        "next_action": {
            "type": "RESPOND_TO_USER",
            "payload": {
                "status": "COMPLETED",
                "route_target": "report_followup_subgraph",
                "answer_type": structured_content.get("answer_type", "REPORT_FOLLOWUP"),
                "rag_query": effective_user_query(state) or extract_user_text(state),
                "tool_decision": tool_decision,
                "quality_review": quality_review,
            },
        },
    }
