from typing import Any

from app.agent.nodes.rag_node_utils import answer_with_rag, fallback_rag_response
from app.agent.state import AgentState
from app.schemas.report import (
    ReportImageInfo,
    ReportRagEvidence,
    TongueAnalysisReport,
)
from app.schemas.tongue import TongueStandardFeatures


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
    if not user_description:
        return feature_query

    user_query = user_description[:160]
    if not feature_query:
        return f"{user_query} 舌象观察 一般健康知识"

    return f"{feature_query} {user_query} 舌象观察 一般健康知识"


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


def _compose_report_answer(
    *,
    feature_names: list[str],
    rag_context: dict[str, Any],
    user_description: str,
) -> str:
    feature_text = "、".join(feature_names) if feature_names else "暂未识别到明确标准特征"
    user_description_section = (
        f"\n\n用户补充情况\n{user_description.strip()}"
        if user_description.strip()
        else ""
    )
    model_scope_note = (
        "当前图像模型主要返回已识别的标准特征；厚薄、润燥、腻腐、舌质颜色等维度如果未被模型返回，"
        "需要后续模型能力或补充信息进一步完善。"
    )

    return (
        "识别结果\n"
        f"当前图像模型主要识别到：{feature_text}。\n"
        f"{model_scope_note}"
        f"{user_description_section}\n\n"
        "一般解释\n"
        f"{_build_general_interpretation(feature_names)}\n\n"
        "继续观察\n"
        f"{_format_numbered(_build_observation_points(feature_names))}\n\n"
        "健康管理建议\n"
        f"{_format_numbered(_build_health_suggestions(feature_names))}\n\n"
        "知识库依据\n"
        f"{_build_knowledge_reference_note(rag_context)}\n\n"
        "风险提醒\n"
        "以上内容用于一般健康知识说明和健康管理参考，不能替代医生诊断。"
        "如果舌象持续明显异常，或伴随明显不适，请及时咨询专业医生。"
    )


def _build_tongue_analysis_report(
    *,
    state: AgentState,
    tongue_features: dict[str, Any],
    feature_names: list[str],
    rag_context: dict[str, Any],
    rag_query: str,
    user_description: str,
    content: str,
) -> dict[str, Any]:
    report = TongueAnalysisReport(
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
            "rag_debug": rag_context.get("debug") or {},
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
    rag_query = _build_augmented_rag_query(feature_rag_query, user_description)

    if rag_query:
        rag_context = await answer_with_rag(rag_query)
    else:
        rag_context = fallback_rag_response("empty_tongue_feature_rag_query")

    content = _compose_report_answer(
        feature_names=feature_names,
        rag_context=rag_context,
        user_description=user_description,
    )

    draft_report = _build_tongue_analysis_report(
        state=state,
        tongue_features=tongue_features,
        feature_names=feature_names,
        rag_context=rag_context,
        rag_query=rag_query,
        user_description=user_description,
        content=content,
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
        },
        "next_action": {
            "type": "RESPOND_TO_USER",
            "payload": {
                "status": "COMPLETED",
                "route_target": "tongue_analysis_subgraph",
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
