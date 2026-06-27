import json
import re
from typing import Any

from app.agent.nodes.rag_node_utils import answer_with_rag, fallback_rag_response
from app.agent.state import AgentState
from app.core.config import get_settings
from app.schemas.report import (
    ReportImageInfo,
    ReportRagEvidence,
    TongueAnalysisReport,
)


def get_chat_model_client():
    from app.integrations.model_gateway import get_chat_model_client as _get_chat_model_client

    return _get_chat_model_client()
from app.schemas.tongue import TongueStandardFeatures
from app.tongue.feature_mapping import FEATURE_MAPPINGS, build_tongue_feature_rag_query


def _detected_feature_names(tongue_features: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for group_key in ["overall", "tongue_body", "coating", "regions"]:
        group = tongue_features.get(group_key)
        if not isinstance(group, dict):
            continue

        if group_key == "overall":
            slots = [group]
        else:
            slots = [value for value in group.values() if isinstance(value, dict)]

        for slot in slots:
            for item in slot.get("items") or []:
                name = item.get("name")
                if name and name not in names:
                    names.append(str(name))

    return names


def _fallback_feature_answer(feature_names: list[str]) -> str:
    if not feature_names:
        return (
            "当前图像模型已完成舌象特征识别，但没有返回明确的标准特征。"
            "你可以继续补充近期饮食、睡眠、口腔清洁和身体感受，"
            "我会帮助整理成一般健康管理参考。"
        )

    feature_text = "、".join(feature_names)
    return (
        f"当前图像模型主要识别到：{feature_text}。"
        "这些特征适合作为一般健康管理参考，需要结合舌苔厚薄、润燥、质地和近期身体感受进一步理解。"
    )


def _extract_image_info(state: AgentState) -> ReportImageInfo:
    client_context = state.get("client_context") or {}
    extra = client_context.get("extra") or {}
    if not isinstance(extra, dict):
        extra = {}

    message = state.get("message") or {}
    attachments = message.get("attachments") or []
    first_image = None
    for attachment in attachments:
        if isinstance(attachment, dict) and attachment.get("file_type") == "image":
            first_image = attachment
            break

    if first_image:
        return ReportImageInfo(
            file_id=first_image.get("file_id"),
            source_type="upload",
            source_uri=extra.get("image_url") or extra.get("tongue_image_url"),
        )

    image_path = extra.get("tongue_image_path") or extra.get("image_path")
    if isinstance(image_path, str) and image_path.strip():
        return ReportImageInfo(
            source_type="path",
            source_uri=image_path.strip(),
            filename=image_path.replace("\\", "/").rstrip("/").split("/")[-1],
        )

    image_url = extra.get("tongue_image_url") or extra.get("image_url")
    if isinstance(image_url, str) and image_url.strip():
        return ReportImageInfo(
            source_type="url",
            source_uri=image_url.strip(),
            filename=image_url.rstrip("/").split("/")[-1],
        )

    return ReportImageInfo()


def _extract_user_description(state: AgentState) -> str:
    client_context = state.get("client_context") or {}
    extra = client_context.get("extra") or {}
    if isinstance(extra, dict):
        for key in ["user_description", "symptom_description", "userDescription"]:
            value = extra.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:500]

    message = state.get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and "用户补充描述：" in content:
        return content.split("用户补充描述：", 1)[1].strip()[:500]

    return ""


def _build_augmented_rag_query(feature_query: str, user_description: str) -> str:
    feature_query = feature_query.strip()
    user_description = user_description.strip()
    inferred_terms = _infer_query_terms_from_description(user_description)
    inferred_text = " ".join(inferred_terms)
    if not user_description:
        return f"{feature_query} {inferred_text}".strip()

    user_query = user_description[:160]
    if not feature_query:
        return f"{user_query} {inferred_text} 舌象观察 一般健康知识".strip()

    return f"{feature_query} {user_query} {inferred_text} 舌象观察 一般健康知识".strip()


def _build_feature_name_rag_query(feature_names: list[str]) -> str:
    terms: list[str] = []
    for feature_name in feature_names:
        normalized_name = feature_name.strip()
        if not normalized_name:
            continue
        for mapping in FEATURE_MAPPINGS.values():
            if mapping.name == normalized_name:
                for term in mapping.rag_terms:
                    if term not in terms:
                        terms.append(term)
                break
        if normalized_name not in terms:
            terms.append(normalized_name)

    return build_tongue_feature_rag_query(terms)


def _infer_query_terms_from_description(user_description: str) -> list[str]:
    text = "".join(user_description.split())
    if not text:
        return []

    term_groups: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
        (
            ("腹胀", "肚子胀", "胃胀", "饭后胀", "脘腹胀", "嗳气"),
            ("脾胃", "运化", "食积", "湿浊", "胃脘痞满", "舌苔厚腻"),
        ),
        (
            ("食欲差", "没胃口", "不想吃", "纳差", "吃不下"),
            ("脾胃虚弱", "脾失健运", "湿困脾胃", "食欲不振"),
        ),
        (
            ("大便黏", "大便粘", "便黏", "便粘", "大便不成形", "拉稀", "便溏"),
            ("湿浊", "湿气", "脾虚湿盛", "大便溏", "苔腻"),
        ),
        (
            ("口腻", "嘴巴黏", "嘴巴粘", "口中黏", "口里发黏", "口苦"),
            ("湿浊", "湿热", "痰湿", "苔腻", "脾胃湿热"),
        ),
        (
            ("疲乏", "乏力", "没精神", "犯困", "身重", "困重"),
            ("脾虚", "湿性重浊", "湿困", "气虚", "痰湿"),
        ),
        (
            ("怕冷", "手脚冷", "畏寒", "发冷"),
            ("寒湿", "阳虚", "苔白滑", "脾阳不足"),
        ),
    )

    terms: list[str] = []
    for triggers, expansions in term_groups:
        if any(trigger in text for trigger in triggers):
            for term in expansions:
                if term not in terms:
                    terms.append(term)
    return terms[:12]


def _build_feature_summary(feature_names: list[str]) -> str:
    if not feature_names:
        return "已完成舌象图像特征识别。"

    return f"本次图像识别到的主要舌象特征包括：{'、'.join(feature_names)}。"


def _build_rag_evidence(rag_context: dict[str, Any]) -> list[ReportRagEvidence]:
    evidence: list[ReportRagEvidence] = []
    for hit in rag_context.get("hits") or []:
        if not isinstance(hit, dict):
            continue

        content = hit.get("content")
        if not content:
            continue

        evidence.append(
            ReportRagEvidence(
                chunk_id=str(hit.get("chunk_id") or ""),
                doc_id=hit.get("doc_id"),
                title=hit.get("title"),
                content=str(content),
                source_uri=hit.get("source_uri"),
                tags=hit.get("tags") or [],
                final_score=float(hit.get("final_score") or 0.0),
                metadata=hit.get("metadata") or {},
            )
        )

    return evidence


def _build_health_notes(feature_names: list[str]) -> list[str]:
    notes = [
        "舌象观察容易受到拍摄光线、饮食、口腔清洁和近期作息影响。",
        "建议结合舌苔厚薄、润燥、质地，以及近期饮食、睡眠、胃肠状态综合理解。",
        "若舌象持续明显变化，或伴随发热、明显疼痛、持续胃肠不适等情况，建议咨询专业医生。",
    ]
    if feature_names:
        notes.insert(0, f"本次可重点关注：{'、'.join(feature_names)}。")
    return notes


def _has_feature(feature_names: list[str], keywords: list[str]) -> bool:
    return any(
        keyword in feature_name
        for feature_name in feature_names
        for keyword in keywords
    )


def _build_general_interpretation(feature_names: list[str]) -> str:
    if not feature_names:
        return (
            "本次暂未形成明确特征结论。舌象分析通常需要同时观察舌质颜色、舌体形态、"
            "舌苔颜色、厚薄、润燥和质地，单张图片结果适合作为健康管理参考。"
        )

    if _has_feature(feature_names, ["白苔", "苔白"]):
        return (
            "白苔是常见舌苔表现。薄白而润可以见于正常舌象；如果舌苔明显偏厚、偏腻或偏滑，"
            "一般会结合饮食、消化、睡眠、口腔清洁和身体感受，进一步理解是否存在湿浊、痰湿、食积等方向的可能。"
        )

    return (
        f"{'、'.join(feature_names)}属于舌象观察中的局部特征。"
        "这些特征不能单独等同于疾病判断，需要结合舌质、舌苔厚薄润燥、近期症状和生活状态综合理解。"
    )


def _build_observation_points(feature_names: list[str]) -> list[str]:
    if _has_feature(feature_names, ["白苔", "苔白"]):
        return [
            "舌苔是薄白还是厚白。",
            "舌苔是润、滑、干，还是有腻感。",
            "是否伴随口腻、腹胀、食欲差、大便黏、怕冷或疲乏等感受。",
            "近期是否有熬夜、饮食油腻、饮酒、口腔清洁不足或刚进食后拍摄。"
        ]

    return [
        "舌质颜色是否偏淡、偏红或偏暗。",
        "舌苔厚薄、润燥和质地是否有明显变化。",
        "近期饮食、睡眠、口腔清洁和胃肠状态是否稳定。",
        "同一光线和角度下连续观察，避免单次图片造成误差。"
    ]


def _build_health_suggestions(feature_names: list[str]) -> list[str]:
    suggestions = [
        "保持口腔清洁，尽量在自然光下复拍，便于后续对比。",
        "近期饮食保持清淡规律，减少过油、过甜、过冷和饮酒刺激。",
        "保证睡眠和作息稳定，结合身体感受持续观察 2 到 3 天。"
    ]
    if _has_feature(feature_names, ["白苔", "苔白"]):
        suggestions.insert(
            0,
            "如果只是薄白而润，通常不需要因为“白苔”本身过度紧张。"
        )
    return suggestions


def _build_knowledge_reference_note(rag_context: dict[str, Any]) -> str:
    hit_count = len(rag_context.get("hits") or [])
    if hit_count <= 0:
        return "本次未检索到足够明确的知识库依据，建议将结果作为初步观察参考。"

    return f"本次已检索到 {hit_count} 条相关知识库依据，页面下方可查看来源片段。"


def _format_numbered(items: list[str]) -> str:
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


def _format_rag_hits_for_prompt(rag_context: dict[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for hit in (rag_context.get("hits") or [])[:limit]:
        if not isinstance(hit, dict):
            continue
        content = str(hit.get("content") or "").strip()
        if not content:
            continue
        formatted.append(
            {
                "title": hit.get("title"),
                "content": content[:700],
                "tags": hit.get("tags") or [],
                "final_score": hit.get("final_score"),
            }
        )
    return formatted


def _build_report_generation_context(
    *,
    feature_names: list[str],
    rag_context: dict[str, Any],
    rag_query: str,
    user_description: str,
) -> dict[str, Any]:
    return {
        "image_model_features": feature_names,
        "user_description": user_description,
        "rag_query": rag_query,
        "rag_grounded": bool(rag_context.get("grounded")),
        "rag_answer": rag_context.get("answer") or "",
        "rag_hits": _format_rag_hits_for_prompt(rag_context),
    }


REPORT_SYNTHESIS_SYSTEM_PROMPT = """你是中医舌象健康管理系统的报告生成助手。
你需要综合三类信息生成中文舌象健康参考报告：
1. 图像模型识别到的舌象标准特征。
2. 用户自己补充的身体感受或近期状态。
3. RAG 知识库检索到的资料和答案。

要求：
- 必须把“用户描述”和“舌象特征”关联起来分析，不能只重复图片识别结果。
- 只能做一般健康知识说明和健康管理参考，不能做疾病诊断。
- 药物、方剂、处方类内容可以做一般知识参考，说明常见方向、适用边界和禁忌；不要声称已经为用户确诊。
- 不要建议用户自行用药、停药、换药、加药或减药；涉及具体剂量、孕期、儿童、老人、慢病或正在用药时提醒医生或药师确认。
- 如果图像模型只返回少量特征，也要围绕颜色、厚薄、润燥、质地、拍摄干扰、用户描述给出完整观察方向，不要只写一段话后结束。
- RAG 资料不足时，可以保守表达，但仍要给出下一步观察建议。
- 不要解释你是如何推理的，不要使用“综合理解”“核心关联”“证据显示”“根据知识库”等过程型表达。
- 不要输出 Markdown，不要输出标题符号，不要输出分隔线，不要输出项目符号。
- 必须只返回 JSON，不要返回额外文字。

JSON 格式：
{
  "result_summary": "直接给用户看的结果，4 到 6 句话。要说明已识别特征、可能相关的生活/身体状态、仍需补充的信息和观察边界，但不能诊断。",
  "daily_care": ["用户接下来可以做的生活方式建议，最多 6 条，必要时可包含药物/方剂一般参考、禁忌和就医沟通要点"],
  "observation": ["接下来需要观察或复拍确认的变化，最多 6 条"],
  "risk_reminder": "一句安全提醒"
}
"""


def _extract_json_object(text: str) -> dict[str, Any] | None:
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


def _clean_text(value: Any, *, max_length: int = 500) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = text.replace("---", "")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:max_length].strip()


def _clean_items(value: Any, *, max_items: int = 4) -> list[str]:
    if not isinstance(value, list):
        return []

    items: list[str] = []
    for item in value:
        text = _clean_text(item, max_length=180)
        if text and text not in items:
            items.append(text)
        if len(items) >= max_items:
            break
    return items


def _compose_user_facing_report(
    *,
    payload: dict[str, Any],
    feature_names: list[str],
    user_description: str,
) -> tuple[str, dict[str, Any]] | None:
    result_summary = _clean_text(payload.get("result_summary"), max_length=700)
    daily_care = _clean_items(payload.get("daily_care"), max_items=6)
    observation = _clean_items(payload.get("observation"), max_items=6)
    risk_reminder = _clean_text(payload.get("risk_reminder"), max_length=220)

    if not result_summary:
        return None

    if not daily_care:
        daily_care = _build_health_suggestions(feature_names)[:6]

    if not observation:
        observation = _build_observation_points(feature_names)[:6]

    if not risk_reminder:
        risk_reminder = "以上内容用于一般健康知识说明和健康管理参考，不能替代医生诊断。"

    if "不能替代医生诊断" not in risk_reminder and "不构成医疗建议" not in risk_reminder:
        risk_reminder += " 以上内容不能替代医生诊断。"

    structured_answer = _build_structured_report_answer(
        result_summary=result_summary,
        daily_care=daily_care,
        observation=observation,
        risk_reminder=risk_reminder,
        feature_names=feature_names,
        user_description=user_description,
    )

    sections = [
        "本次结果",
        result_summary,
    ]

    if user_description.strip():
        sections.extend(["", "你可以先这样做", _format_numbered(daily_care)])
    else:
        sections.extend(["", "建议", _format_numbered(daily_care)])

    sections.extend(["", "接下来观察", _format_numbered(observation)])
    sections.extend(["", "提醒", risk_reminder])

    return "\n".join(sections).strip(), structured_answer


def _build_structured_report_answer(
    *,
    result_summary: str,
    daily_care: list[str],
    observation: list[str],
    risk_reminder: str,
    feature_names: list[str],
    user_description: str,
) -> dict[str, Any]:
    highlights = []
    if feature_names:
        highlights.append("识别特征：" + "、".join(feature_names))
    if user_description.strip():
        highlights.append("已结合你的补充描述")

    return {
        "schema_version": "1.0",
        "answer_type": "TONGUE_REPORT",
        "title": "舌象健康参考",
        "summary": result_summary,
        "highlights": highlights,
        "sections": [
            {
                "title": "识别结果",
                "items": feature_names or ["图像模型暂未返回明确标准特征"],
            },
            {
                "title": "舌象含义参考",
                "content": _build_general_interpretation(feature_names),
            },
            {
                "title": "你可以先这样做",
                "items": daily_care,
            },
            {
                "title": "接下来观察",
                "items": observation,
            },
            {
                "title": "补充信息建议",
                "items": [
                    "补充近期饮食、睡眠、口腔清洁、胃肠状态和冷热感受，可以让后续分析更贴近实际。",
                    "建议在自然光、未进食染色食物、同一角度下复拍，便于前后对比。",
                ],
            },
        ],
        "disclaimer": risk_reminder,
    }


async def _generate_integrated_report_answer(
    *,
    feature_names: list[str],
    rag_context: dict[str, Any],
    rag_query: str,
    user_description: str,
) -> tuple[str, dict[str, Any]] | None:
    settings = get_settings()
    context = _build_report_generation_context(
        feature_names=feature_names,
        rag_context=rag_context,
        rag_query=rag_query,
        user_description=user_description,
    )

    messages = [
        {
            "role": "system",
            "content": REPORT_SYNTHESIS_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "请根据以下 JSON 上下文生成最终舌象健康参考报告。\n"
                f"{json.dumps(context, ensure_ascii=False)}"
            ),
        },
    ]

    try:
        raw_content = await get_chat_model_client().generate(
            messages=messages,
            temperature=0.25,
            max_tokens=min(settings.chat_model_max_tokens, 1800),
        )
    except Exception:
        return None

    payload = _extract_json_object(raw_content)
    if payload is None:
        return None

    return _compose_user_facing_report(
        payload=payload,
        feature_names=feature_names,
        user_description=user_description,
    )


def _compose_report_answer(
    *,
    feature_names: list[str],
    rag_context: dict[str, Any],
    user_description: str,
) -> str:
    feature_text = "、".join(feature_names) if feature_names else "暂未识别到明确标准特征"
    description = user_description.strip()
    if description:
        result_summary = (
            f"本次图片主要识别到：{feature_text}。结合你补充的“{description}”，"
            "可以先把重点放在饮食消化、腹胀、大便状态、口腔黏腻感和近期作息变化上。"
            f"{_build_general_interpretation(feature_names)}"
        )
    else:
        result_summary = (
            f"本次图片主要识别到：{feature_text}。"
            f"{_build_general_interpretation(feature_names)}"
        )

    return (
        "本次结果\n"
        f"{result_summary}\n\n"
        "建议\n"
        f"{_format_numbered(_build_health_suggestions(feature_names))}\n\n"
        "接下来观察\n"
        f"{_format_numbered(_build_observation_points(feature_names))}\n\n"
        "提醒\n"
        "以上内容用于一般健康知识说明和健康管理参考，不能替代医生诊断。"
        "如果舌象持续明显异常，或伴随明显不适，请及时咨询专业医生。"
    )


def _build_template_structured_report_answer(
    *,
    feature_names: list[str],
    user_description: str,
) -> dict[str, Any]:
    feature_text = "、".join(feature_names) if feature_names else "暂未识别到明确标准特征"
    if user_description.strip():
        summary = (
            f"本次图片主要识别到：{feature_text}。结合你补充的“{user_description.strip()}”，"
            "可以先把重点放在饮食消化、睡眠、精神状态和舌苔变化上。"
        )
    else:
        summary = f"本次图片主要识别到：{feature_text}。这些内容适合作为一般健康管理参考。"

    return _build_structured_report_answer(
        result_summary=summary,
        daily_care=_build_health_suggestions(feature_names)[:4],
        observation=_build_observation_points(feature_names)[:4],
        risk_reminder="以上内容用于一般健康知识说明和健康管理参考，不能替代医生诊断。",
        feature_names=feature_names,
        user_description=user_description,
    )


def _build_tongue_analysis_report(
    *,
    state: AgentState,
    tongue_features: dict[str, Any],
    feature_names: list[str],
    rag_context: dict[str, Any],
    rag_query: str,
    user_description: str,
    report_generation_mode: str,
    content: str,
    structured_answer: dict[str, Any] | None,
) -> dict[str, Any]:
    report = TongueAnalysisReport(
        report_status="FINAL",
        report_id=state.get("report_id"),
        user_id=state.get("user_id"),
        thread_id=str(state.get("thread_id") or ""),
        task_id=state.get("task_id"),
        task_version=state.get("task_version"),
        image_info=_extract_image_info(state),
        standard_features=TongueStandardFeatures.model_validate(tongue_features),
        feature_summary=_build_feature_summary(feature_names),
        rag_query=rag_query,
        rag_grounded=bool(rag_context.get("grounded")),
        rag_evidence=_build_rag_evidence(rag_context),
        summary=content,
        health_notes=_build_health_notes(feature_names),
        versions={
            "report_schema": "tongue_analysis_report_v1.0",
            "feature_schema": "tongue_standard_features_v1.0",
            "rag_engine": rag_context.get("retrieval_engine"),
            "answer_engine": rag_context.get("answer_engine"),
        },
        metadata={
            "user_description": user_description,
            "report_generation": {
                "mode": report_generation_mode,
                "inputs": [
                    "image_model_features",
                    "user_description",
                    "rag_answer",
                    "rag_hits",
                ],
            },
            "rag_debug": rag_context.get("debug") or {},
            "rag_answer": rag_context.get("answer"),
            "hit_count": len(rag_context.get("hits") or []),
            "structured_sections": {
                "user_description": user_description,
                "general_interpretation": _build_general_interpretation(
                    feature_names
                ),
                "observation_points": _build_observation_points(feature_names),
                "health_suggestions": _build_health_suggestions(feature_names),
                "knowledge_reference_note": _build_knowledge_reference_note(
                    rag_context
                ),
            },
            "structured_answer": structured_answer,
        },
    )
    return report.model_dump(mode="json")


async def tongue_report_node(state: AgentState) -> AgentState:
    tongue_features = state.get("tongue_features") or {}
    if not isinstance(tongue_features, dict):
        tongue_features = {}

    feature_names = _detected_feature_names(tongue_features)
    user_description = _extract_user_description(state)
    feature_rag_query = str(tongue_features.get("rag_query") or "").strip()
    if not feature_rag_query:
        feature_rag_query = _build_feature_name_rag_query(feature_names)
    rag_query = _build_augmented_rag_query(feature_rag_query, user_description)

    if rag_query:
        rag_context = await answer_with_rag(rag_query)
    else:
        rag_context = fallback_rag_response("empty_tongue_feature_rag_query")

    generated_answer = await _generate_integrated_report_answer(
        feature_names=feature_names,
        rag_context=rag_context,
        rag_query=rag_query,
        user_description=user_description,
    )
    content = None
    structured_answer = None
    if isinstance(generated_answer, tuple):
        content, structured_answer = generated_answer
    elif isinstance(generated_answer, str):
        content = generated_answer

    report_generation_mode = "llm_integrated_synthesis"
    if not content:
        report_generation_mode = "template_fallback"
        content = _compose_report_answer(
            feature_names=feature_names,
            rag_context=rag_context,
            user_description=user_description,
        )
        structured_answer = _build_template_structured_report_answer(
            feature_names=feature_names,
            user_description=user_description,
        )
    elif structured_answer is None:
        structured_answer = _build_template_structured_report_answer(
            feature_names=feature_names,
            user_description=user_description,
        )

    draft_report = _build_tongue_analysis_report(
        state=state,
        tongue_features=tongue_features,
        feature_names=feature_names,
        rag_context=rag_context,
        rag_query=rag_query,
        user_description=user_description,
        report_generation_mode=report_generation_mode,
        content=content,
        structured_answer=structured_answer,
    )

    return {
        **state,
        "current_node": "tongue_report_node",
        "rag_context": rag_context,
        "draft_report": draft_report,
        "response_message": {
            "role": "assistant",
            "content_type": "text",
            "content": content,
            "structured_content": structured_answer,
        },
        "next_action": {
            "type": "RESPOND_TO_USER",
            "payload": {
                "status": "COMPLETED",
                "route_target": "tongue_analysis_subgraph",
                "answer_type": "TONGUE_REPORT",
                "grounded": rag_context.get("grounded", False),
                "hit_count": len(rag_context.get("hits") or []),
                "detected_feature_codes": tongue_features.get(
                    "detected_feature_codes", []
                ),
                "rag_query": rag_query,
                "draft_report": draft_report,
            },
        },
    }
