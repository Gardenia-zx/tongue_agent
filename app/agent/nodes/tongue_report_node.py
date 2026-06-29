import json
import logging
import re
from typing import Any

from app.agent.nodes.rag_node_utils import answer_with_rag, fallback_rag_response
from app.agent.state import AgentState
from app.core.config import get_settings
from app.schemas.report import (
    ReportEvidenceRef,
    ReportImageInfo,
    ReportRagEvidence,
    TongueAnalysisReport,
)


def get_chat_model_client():
    from app.integrations.model_gateway import get_chat_model_client as _get_chat_model_client

    return _get_chat_model_client()
from app.schemas.tongue import TongueStandardFeatures
from app.tongue.feature_mapping import FEATURE_MAPPINGS, build_tongue_feature_rag_query


logger = logging.getLogger(__name__)
REPORT_SCHEMA_VERSION = "2.0"
SUMMARY_MIN_LENGTH = 40
PLAN_KEYS = ("diet_plan", "sleep_plan", "exercise_plan")
REPORT_DIMENSIONS = {
    "tongue_body.color": "舌质颜色",
    "tongue_body.shape": "舌体形态",
    "tongue_body.texture": "舌体质地",
    "coating.color": "舌苔颜色",
    "coating.thickness": "舌苔厚薄",
    "coating.moisture": "舌苔润燥",
    "coating.texture": "舌苔质地",
    "regions": "局部分区",
}
DEFAULT_DIMENSION_VALUES = {
    "tongue_body.color": "淡红",
    "tongue_body.shape": "正常",
    "tongue_body.texture": "未见明显异常",
    "coating.color": "白色",
    "coating.thickness": "薄",
    "coating.moisture": "润",
    "coating.texture": "不腻",
    "regions": "未见明显局部异常",
}
STATE_SNAPSHOT_LABELS = {
    "sleep_status": {
        "NORMAL": "睡眠正常，醒后精神较好",
        "SHORT": "睡眠时间偏短",
        "DIFFICULT_OR_WAKE": "入睡困难或容易夜醒",
        "IRREGULAR_LATE": "经常熬夜，作息不规律",
    },
    "digestion_status": {
        "NORMAL": "胃口和消化基本正常",
        "POOR_APPETITE": "食欲偏差",
        "BLOATING": "饭后容易腹胀",
        "GREASY_REFLUX_DISCOMFORT": "容易口腻、反酸或不舒服",
    },
    "bowel_status": {
        "NORMAL": "排便基本正常",
        "DRY_CONSTIPATION": "偏干或排便困难",
        "LOOSE_DIARRHEA": "偏稀或容易腹泻",
        "STICKY_INCOMPLETE": "大便黏滞或感觉排不干净",
    },
    "current_states": {
        "NORMAL": "精神状态正常",
        "FATIGUE": "容易疲乏",
        "STRESS_ANXIETY": "压力较大或容易焦虑",
        "COLD_SENSITIVE": "容易怕冷",
        "HEAT_DRY_MOUTH": "容易燥热或口干",
    },
    "health_goals": {
        "DIET_DIGESTION": "饮食和消化",
        "SLEEP_ROUTINE": "睡眠和作息",
        "FITNESS": "运动和体能",
        "FATIGUE_ENERGY": "疲劳和精神状态",
        "WEIGHT_MANAGEMENT": "体重管理",
        "UNDERSTAND_TONGUE": "了解本次舌象",
    },
}
FEATURE_VALUE_BY_CODE = {
    "coating.color.white": "白色",
    "coating.color.yellow": "黄色",
    "coating.color.black": "黑色",
    "tongue_body.color.red": "红",
    "tongue_body.color.purple": "紫",
    "coating.moisture.slippery": "滑润",
}


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


def _extract_state_snapshot(state: AgentState) -> dict[str, Any]:
    client_context = state.get("client_context") or {}
    extra = client_context.get("extra") or {}
    if not isinstance(extra, dict):
        return {}
    snapshot = extra.get("state_snapshot")
    if not isinstance(snapshot, dict):
        return {}

    def _string(key: str) -> str:
        value = snapshot.get(key)
        return value.strip() if isinstance(value, str) else ""

    def _list(key: str) -> list[str]:
        value = snapshot.get(key)
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    return {
        "observation_window": _string("observation_window") or "LAST_3_DAYS",
        "sleep_status": _string("sleep_status"),
        "digestion_status": _string("digestion_status"),
        "bowel_status": _string("bowel_status"),
        "current_states": _list("current_states"),
        "health_goals": _list("health_goals"),
        "free_description": _string("free_description")[:500],
        "skipped": bool(snapshot.get("skipped")),
    }


def _state_label(group: str, code: str) -> str:
    return STATE_SNAPSHOT_LABELS.get(group, {}).get(code, code)


def _personalization_signals(state_snapshot: dict[str, Any]) -> list[str]:
    if not state_snapshot:
        return []
    if state_snapshot.get("skipped"):
        return ["用户跳过了近3天状态补充，个性化信息不足"]

    signals: list[str] = []
    for key in ("sleep_status", "digestion_status", "bowel_status"):
        code = state_snapshot.get(key)
        if isinstance(code, str) and code:
            signals.append(_state_label(key, code))

    for key in ("current_states", "health_goals"):
        values = state_snapshot.get(key)
        if isinstance(values, list):
            for code in values:
                if isinstance(code, str) and code:
                    signals.append(_state_label(key, code))

    free_description = state_snapshot.get("free_description")
    if isinstance(free_description, str) and free_description.strip():
        signals.append(f"自由补充：{free_description.strip()[:160]}")

    deduped: list[str] = []
    for signal in signals:
        if signal not in deduped:
            deduped.append(signal)
    return deduped


def _state_snapshot_rag_terms(state_snapshot: dict[str, Any]) -> str:
    if not state_snapshot or state_snapshot.get("skipped"):
        return ""
    terms: list[str] = []
    for signal in _personalization_signals(state_snapshot):
        if "正常" not in signal:
            terms.append(signal)
    return " ".join(terms)[:260]


def _build_augmented_rag_query(
    feature_query: str,
    user_description: str,
    state_snapshot: dict[str, Any] | None = None,
) -> str:
    feature_query = feature_query.strip()
    user_description = user_description.strip()
    inferred_terms = _infer_query_terms_from_description(user_description)
    inferred_text = " ".join(inferred_terms)
    state_terms = _state_snapshot_rag_terms(state_snapshot or {})
    if not user_description:
        return f"{feature_query} {inferred_text} {state_terms}".strip()

    user_query = user_description[:160]
    if not feature_query:
        return f"{user_query} {inferred_text} {state_terms} 舌象观察 一般健康知识".strip()

    return f"{feature_query} {user_query} {inferred_text} {state_terms} 舌象观察 一般健康知识".strip()


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


def _build_evidence_refs(rag_context: dict[str, Any]) -> list[ReportEvidenceRef]:
    refs: list[ReportEvidenceRef] = []
    for hit in rag_context.get("hits") or []:
        if not isinstance(hit, dict):
            continue
        refs.append(
            ReportEvidenceRef(
                doc_id=hit.get("doc_id"),
                chunk_id=str(hit.get("chunk_id") or ""),
                title=hit.get("title"),
                final_score=float(hit.get("final_score") or 0.0),
            )
        )
    return refs


def _build_tongue_feature_dicts(tongue_features: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            code = value.get("code")
            if isinstance(code, str) and code not in seen:
                items.append(
                    {
                        "code": code,
                        "name": value.get("name"),
                        "status": value.get("status") or "DETECTED",
                        "confidence": value.get("confidence"),
                    }
                )
                seen.add(code)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(tongue_features)
    return items


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


def _build_dietary_advice(feature_names: list[str]) -> list[str]:
    suggestions = [
        "饮食保持清淡规律，少吃过油、过甜、辛辣、生冷和饮酒刺激。",
        "观察进食后腹胀、口腻、大便、食欲变化，记录与舌象变化的关系。",
    ]
    if _has_feature(feature_names, ["白苔", "苔白"]):
        suggestions.insert(0, "如果只是薄白而润，饮食上保持规律即可，不需要因为“白苔”本身过度紧张。")
    return suggestions


def _build_exercise_advice(feature_names: list[str]) -> list[str]:
    return [
        "选择中等强度、可长期坚持的运动，如快走、八段锦、拉伸或轻力量训练。",
        "运动后关注疲劳、睡眠和恢复情况，避免短期突然增加强度。",
    ]


def _build_lifestyle_advice(feature_names: list[str]) -> list[str]:
    return [
        "保持规律作息，尽量避免连续熬夜。",
        "保持口腔清洁，复拍时尽量使用自然光、相似角度和相似时间。",
    ]


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
    tongue_features: dict[str, Any],
    rag_context: dict[str, Any],
    rag_query: str,
    user_description: str,
    state_snapshot: dict[str, Any] | None = None,
    personalization_signals: list[str] | None = None,
) -> dict[str, Any]:
    feature_context = _extract_feature_context(tongue_features)
    return {
        "detected_features": feature_context["detected_features"],
        "not_evaluated_dimensions": feature_context["not_evaluated_dimensions"],
        "unsupported_dimensions": feature_context["unsupported_dimensions"],
        "recognition_limits": feature_context["recognition_limits"],
        "dimension_values": feature_context["dimension_values"],
        "image_quality": tongue_features.get("image_quality")
        or tongue_features.get("quality_metrics")
        or {},
        "user_description": user_description,
        "state_snapshot": state_snapshot or {},
        "personalization_signals": personalization_signals or [],
        "rag_summary": {
            "query": rag_query,
            "grounded": bool(rag_context.get("grounded")),
            "answer": rag_context.get("answer") or "",
            "hit_count": len(rag_context.get("hits") or []),
            "hits": _format_rag_hits_for_prompt(rag_context),
        },
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
- 必须一次性返回完整结构化 JSON，不要返回额外文字，不要省略任何字段。
- dietary_advice、exercise_advice、lifestyle_advice 每个字段必须返回 2 到 4 条。
- 饮食建议只写饮食、饮水、进食习惯；运动建议只写运动方式、强度、频率和恢复；生活方式只写作息、口腔清洁、复拍习惯。
- 舌苔厚薄、润滑干腻、腹胀食欲等“需要继续观察的问题”只能放入 observation，不要放入 dietary_advice、exercise_advice 或 lifestyle_advice。
- 三类建议都必须是可执行建议，不要写成疑问句。

JSON 格式：
{
  "result_summary": "直接给用户看的结果，4 到 6 句话。要说明已识别特征、可能相关的生活/身体状态、仍需补充的信息和观察边界，但不能诊断。",
  "health_interpretation": "对本次舌象特征的健康管理解释，不能诊断。",
  "dietary_advice": ["饮食建议，最多 4 条，只写饮食相关内容"],
  "exercise_advice": ["运动建议，最多 4 条，只写运动强度、方式和恢复观察"],
  "lifestyle_advice": ["生活方式建议，最多 4 条，只写作息、口腔清洁、复拍习惯等内容"],
  "observation": ["接下来需要观察或复拍确认的变化，最多 6 条"],
  "risk_reminder": "一句安全提醒"
}
"""

REPORT_SYNTHESIS_SYSTEM_PROMPT_V2 = """你是舌象健康管理报告生成助手，只能生成一般健康管理参考，不能诊断疾病。

必须一次性返回完整 JSON，不要输出 Markdown、标题符号或额外解释。

强约束：
1. 顶层 schema_version 必须是 "2.0"。
2. recognition_evidence 只能使用用户上下文 detected_features 中已有的 code/name/confidence/status，不得新增、改名或把未评估维度写成识别事实。
3. recognition_limits 只能来自上下文的 recognition_limits、not_evaluated_dimensions、unsupported_dimensions。
4. dimension_values 可用于生成建议；其中 DEFAULT 是系统默认值，不要写成模型识别事实。
5. 对默认值或未识别但需要讨论的情况，只能写在 conditional_analysis，使用“如果……”结构。
6. diet_plan、sleep_plan、exercise_plan 必须是对象，且包含 goal、actions、frequency、duration、observation_metrics。
7. 报告要详细，饮食、睡眠、运动都要给出可执行计划；exercise_plan 必须包含每日运动安排。
8. sections 不要包含“识别证据”或“识别边界”板块，用户正文不展示模型识别过程。
9. comprehensive_summary 不要写“图像模型识别到”“识别证据”“识别边界”，直接写舌象健康管理建议。
10. risk_tips 必须提醒内容不能替代医生诊断。
11. 如果上下文 state_snapshot.skipped 为 true，必须说明个性化信息不足，不能猜测用户近期状态。
12. 如果提供了 state_snapshot，至少使用 personalization_signals 中两项用户选择；不足两项则使用全部。
13. 建议必须出现“因为你反馈/选择了……所以建议……”这种因果表达。
14. 饮食建议必须包含具体日常食物示例。
15. 运动建议必须包含运动项目、时长和强度；如用户提到损伤或不适，先避开冲突运动。
16. 用户选择 NORMAL 时，不要把正常项写成异常。
17. 自由描述中的过敏、忌口、运动损伤和正在执行的计划优先级最高。

返回 JSON 结构：
{
  "schema_version": "2.0",
  "comprehensive_summary": "4到8句话，说明已识别事实、可信边界、结合用户描述的健康管理方向和后续观察重点",
  "recognition_evidence": [{"code": "来自上下文", "name": "来自上下文", "confidence": 0.9, "status": "DETECTED"}],
  "recognition_limits": [{"dimension": "来自上下文", "status": "NOT_EVALUATED 或 UNSUPPORTED_BY_MODEL", "reason": "边界说明"}],
  "dimension_values": [{"dimension": "coating.color", "name": "舌苔颜色", "value": "白色", "status": "DETECTED 或 DEFAULT"}],
  "tongue_feature_explanation": "详细解释已识别特征的健康管理含义，不把未识别维度当事实",
  "conditional_analysis": [{"condition": "如果……", "interpretation": "对应观察解释"}],
  "diet_plan": {"goal": "目标", "actions": ["具体行动"], "frequency": "每天", "duration": "连续3天", "observation_metrics": ["观察指标"]},
  "sleep_plan": {"goal": "目标", "actions": ["具体行动"], "frequency": "每天", "duration": "连续3天", "observation_metrics": ["观察指标"]},
  "exercise_plan": {"goal": "目标", "actions": ["具体行动"], "frequency": "每天", "duration": "连续3天", "observation_metrics": ["观察指标"]},
  "three_day_observation": ["至少3条"],
  "followup_questions": ["至少3条"],
  "risk_tips": ["至少1条"]
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


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _dimension_from_feature_code(code: str) -> str:
    parts = code.split(".")
    if len(parts) >= 2 and parts[0] in {"tongue_body", "coating"}:
        return f"{parts[0]}.{parts[1]}"
    if parts and parts[0] == "regions":
        return "regions"
    return parts[0] if parts else "unknown"


def _dimension_label(dimension: str) -> str:
    return REPORT_DIMENSIONS.get(dimension, dimension)


def _feature_value(item: dict[str, Any]) -> str:
    code = str(item.get("code") or "")
    return FEATURE_VALUE_BY_CODE.get(code) or str(item.get("name") or code)


def _extract_feature_context(tongue_features: dict[str, Any]) -> dict[str, Any]:
    feature_items = _build_tongue_feature_dicts(tongue_features)
    detected: list[dict[str, Any]] = []
    detected_dimensions: set[str] = set()
    unsupported_dimensions: set[str] = set()
    dimension_values: dict[str, dict[str, Any]] = {
        dimension: {
            "dimension": dimension,
            "name": _dimension_label(dimension),
            "value": value,
            "status": "DEFAULT",
        }
        for dimension, value in DEFAULT_DIMENSION_VALUES.items()
    }

    for item in feature_items:
        code = str(item.get("code") or "").strip()
        if not code:
            continue
        status = str(item.get("status") or "DETECTED").upper()
        dimension = _dimension_from_feature_code(code)
        if status == "DETECTED":
            detected.append(
                {
                    "code": code,
                    "name": str(item.get("name") or code),
                    "confidence": _safe_float(item.get("confidence")),
                    "status": "DETECTED",
                }
            )
            detected_dimensions.add(dimension)
            dimension_values[dimension] = {
                "dimension": dimension,
                "name": _dimension_label(dimension),
                "value": _feature_value(item),
                "status": "DETECTED",
                "confidence": _safe_float(item.get("confidence")),
            }
        elif status == "UNSUPPORTED_BY_MODEL":
            unsupported_dimensions.add(dimension)

    for code in tongue_features.get("unsupported_feature_codes") or []:
        if isinstance(code, str) and code.strip():
            unsupported_dimensions.add(_dimension_from_feature_code(code.strip()))

    explicit_not_evaluated: set[str] = set()
    for code in tongue_features.get("not_evaluated_feature_codes") or []:
        if isinstance(code, str) and code.strip():
            explicit_not_evaluated.add(_dimension_from_feature_code(code.strip()))

    known_dimensions = set(REPORT_DIMENSIONS.keys())
    not_evaluated_dimensions = (
        known_dimensions - detected_dimensions - unsupported_dimensions
    ) | explicit_not_evaluated

    limits: list[dict[str, Any]] = []
    for dimension in sorted(not_evaluated_dimensions):
        limits.append(
            {
                "dimension": dimension,
                "name": _dimension_label(dimension),
                "status": "NOT_EVALUATED",
                "reason": "本次图像模型未给出该维度的稳定识别结果，不能作为已识别事实。",
            }
        )
    for dimension in sorted(unsupported_dimensions):
        limits.append(
            {
                "dimension": dimension,
                "name": _dimension_label(dimension),
                "status": "UNSUPPORTED_BY_MODEL",
                "reason": "当前模型暂不支持该维度，不参与本次舌象结论。",
            }
        )

    return {
        "detected_features": detected,
        "recognition_limits": limits,
        "dimension_values": list(dimension_values.values()),
        "detected_dimensions": sorted(detected_dimensions),
        "unsupported_dimensions": sorted(unsupported_dimensions),
        "not_evaluated_dimensions": sorted(not_evaluated_dimensions),
    }


def _default_summary(feature_names: list[str], user_description: str) -> str:
    feature_text = "、".join(feature_names) if feature_names else "当前舌象信息"
    description = user_description.strip()
    if description:
        return (
            f"结合当前舌象信息和你补充的“{description}”，可以先把重点放在饮食规律、胃肠感受、睡眠恢复和运动节奏上。"
            f"{feature_text}适合作为近期健康管理的观察线索，但不能单独等同于疾病判断。"
            "接下来建议用连续三天的饮食、睡眠、运动和大便状态来验证调整是否有效。"
        )
    return (
        f"结合当前舌象信息，可以先围绕饮食清淡规律、睡眠恢复、适度运动和胃肠状态做连续观察。{feature_text}只能作为一般健康管理参考，"
        "不能直接等同于疾病判断。建议近三天记录饭后腹胀、口中黏腻感、大便状态、睡眠质量和运动后恢复情况。"
    )


def _default_tongue_feature_explanation(feature_names: list[str]) -> str:
    if feature_names:
        return (
            f"当前舌象信息中可重点参考：{'、'.join(feature_names)}。这些表现需要和饮食作息、胃肠状态、睡眠恢复及复拍变化一起理解。"
            "单次舌象只能作为健康管理线索，后续更适合看连续变化。"
        )
    return (
        "本次图像未形成足够明确的标准特征结论。建议优先补充近期饮食、睡眠、胃肠状态和复拍图像，再进行更具体的健康管理分析。"
    )


def _default_conditional_analysis(feature_names: list[str]) -> list[dict[str, str]]:
    feature_text = "、".join(feature_names) if feature_names else "当前特征"
    return [
        {
            "condition": f"如果后续复拍仍稳定出现{feature_text}",
            "interpretation": "可结合饮食、睡眠、口腔清洁和胃肠状态观察是否存在持续性变化，不要只凭单次图片下结论。",
        },
        {
            "condition": "如果实际舌苔偏厚、发腻或口中黏腻感明显",
            "interpretation": "建议重点记录油腻甜食、夜宵、饮酒、腹胀、大便黏滞和食欲变化。",
        },
        {
            "condition": "如果只是薄白且无明显不适",
            "interpretation": "通常以规律饮食、稳定作息和连续观察为主，不需要做激烈调整。",
        },
    ]


def _default_plan(plan_key: str, feature_names: list[str]) -> dict[str, Any]:
    if plan_key == "diet_plan":
        return {
            "goal": "减轻近期饮食对舌象和胃肠状态的干扰",
            "actions": [
                "三餐尽量规律，晚餐七八分饱，减少夜宵。",
                "连续三天少吃生冷、油腻、甜腻、辛辣和酒精。",
                "优先选择温热、清淡、容易消化的日常食物，例如粥、面、山药、南瓜和熟蔬菜。",
                "饭后记录腹胀、口腻、食欲和大便状态，观察是否随饮食调整改善。",
            ],
            "frequency": "每天",
            "duration": "连续3天",
            "observation_metrics": ["饭后腹胀", "口中黏腻感", "食欲", "大便状态"],
        }
    if plan_key == "sleep_plan":
        return {
            "goal": "稳定作息，减少熬夜对舌象和恢复状态的影响",
            "actions": [
                "尽量固定入睡和起床时间，睡前一小时减少高强度工作和刷屏。",
                "晚餐后避免大量咖啡因、酒精和过饱进食。",
                "记录入睡时间、夜醒次数、醒后疲乏感和晨起口干口苦情况。",
            ],
            "frequency": "每天",
            "duration": "连续3天",
            "observation_metrics": ["入睡时间", "夜醒次数", "醒后疲乏感", "晨起口干"],
        }
    return {
        "goal": "用低风险、可持续的活动促进循环和恢复",
        "actions": [
            "每天安排20到30分钟中等强度快走，能说话但略微出汗即可。",
            "久坐超过一小时后起身活动3到5分钟，做肩颈、髋部和小腿拉伸。",
            "如果近期疲乏明显，先选择散步、八段锦或轻柔拉伸，避免突然增加高强度训练。",
            "运动后观察疲劳、睡眠、胃口和第二天精神状态。",
        ],
        "frequency": "每天",
        "duration": "连续3天",
        "observation_metrics": ["运动后疲劳", "睡眠质量", "食欲", "第二天精神状态"],
    }


def _clean_plan(value: Any, *, plan_key: str) -> dict[str, Any]:
    fallback = _default_plan(plan_key, [])
    source = value if isinstance(value, dict) else {}
    actions = _clean_items(source.get("actions") if source else value, max_items=6)
    metrics = _clean_items(source.get("observation_metrics"), max_items=5)
    plan = {
        "goal": _clean_text(source.get("goal"), max_length=120) or fallback["goal"],
        "actions": actions or fallback["actions"],
        "frequency": _clean_text(source.get("frequency"), max_length=40) or fallback["frequency"],
        "duration": _clean_text(source.get("duration"), max_length=40) or fallback["duration"],
        "observation_metrics": metrics or fallback["observation_metrics"],
    }
    return plan


def _clean_conditional_analysis(value: Any, feature_names: list[str]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            condition = _clean_text(item.get("condition"), max_length=120)
            interpretation = _clean_text(item.get("interpretation"), max_length=260)
            if condition and interpretation:
                items.append({"condition": condition, "interpretation": interpretation})
            if len(items) >= 5:
                break
    return items or _default_conditional_analysis(feature_names)


def _ensure_min_items(items: list[str], fallback: list[str], min_items: int, max_items: int) -> list[str]:
    result = list(items[:max_items])
    for item in fallback:
        if len(result) >= min_items:
            break
        if item not in result:
            result.append(item)
    return result[:max_items]


def _default_followup_questions() -> list[str]:
    return [
        "最近三天是否有腹胀、口中黏腻、食欲下降或大便黏滞？",
        "最近是否熬夜、饮酒、吃夜宵，或明显增加生冷甜腻食物？",
        "复拍时舌苔厚薄、润燥、颜色范围是否和本次相近？",
    ]


def _normalize_schema2_payload(
    *,
    payload: dict[str, Any],
    context: dict[str, Any],
    feature_names: list[str],
    user_description: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    missing_fields: list[str] = []
    result_summary = _clean_text(
        payload.get("comprehensive_summary")
        or payload.get("result_summary")
        or payload.get("summary"),
        max_length=1000,
    )
    if not result_summary:
        return None, ["comprehensive_summary"]
    if len(result_summary) < SUMMARY_MIN_LENGTH:
        result_summary = f"{result_summary} {_default_summary(feature_names, user_description)}".strip()
        missing_fields.append("comprehensive_summary")

    feature_explanation = _clean_text(
        payload.get("tongue_feature_explanation") or payload.get("health_interpretation"),
        max_length=800,
    ) or _default_tongue_feature_explanation(feature_names)

    three_day_observation = _clean_items(
        payload.get("three_day_observation")
        or payload.get("observation")
        or payload.get("observation_points"),
        max_items=6,
    )
    three_day_observation = _ensure_min_items(
        three_day_observation,
        _build_observation_points(feature_names),
        3,
        6,
    )

    followup_questions = _clean_items(payload.get("followup_questions"), max_items=5)
    followup_questions = _ensure_min_items(
        followup_questions,
        _default_followup_questions(),
        3,
        5,
    )

    risk_tips = _clean_items(
        payload.get("risk_tips") or payload.get("risk_reminder"),
        max_items=3,
    )
    if not risk_tips:
        risk_tips = [
            "以上内容用于一般健康知识说明和健康管理参考，不能替代医生诊断；如果不适明显或持续加重，请及时咨询医生。"
        ]

    diet_source = payload.get("diet_plan") or payload.get("dietary_advice")
    sleep_source = payload.get("sleep_plan") or payload.get("lifestyle_advice")
    exercise_source = payload.get("exercise_plan") or payload.get("exercise_advice")
    structured = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "answer_type": "TONGUE_REPORT",
        "title": "舌象健康参考",
        "summary": result_summary,
        "comprehensive_summary": result_summary,
        "recognition_evidence": context["detected_features"],
        "recognition_limits": context["recognition_limits"],
        "dimension_values": context.get("dimension_values") or [],
        "conditional_analysis": _clean_conditional_analysis(
            payload.get("conditional_analysis"), feature_names
        ),
        "tongue_feature_explanation": feature_explanation,
        "diet_plan": _clean_plan(diet_source, plan_key="diet_plan"),
        "sleep_plan": _clean_plan(sleep_source, plan_key="sleep_plan"),
        "exercise_plan": _clean_plan(exercise_source, plan_key="exercise_plan"),
        "three_day_observation": three_day_observation,
        "followup_questions": followup_questions,
        "risk_tips": risk_tips,
        "disclaimer": risk_tips[0],
    }

    legacy = _derive_legacy_fields(structured)
    structured.update(legacy)
    structured["sections"] = _build_schema2_sections(structured)
    structured["highlights"] = [
        "已生成饮食、睡眠和运动计划",
    ]

    validation_errors = _validate_schema2_report(structured)
    missing_fields.extend(validation_errors)
    return structured, sorted(set(missing_fields))


def _derive_legacy_fields(structured: dict[str, Any]) -> dict[str, Any]:
    diet_plan = structured.get("diet_plan") if isinstance(structured.get("diet_plan"), dict) else {}
    sleep_plan = structured.get("sleep_plan") if isinstance(structured.get("sleep_plan"), dict) else {}
    exercise_plan = structured.get("exercise_plan") if isinstance(structured.get("exercise_plan"), dict) else {}
    health_interpretation = (
        _clean_text(structured.get("tongue_feature_explanation"), max_length=800)
        or _clean_text(structured.get("comprehensive_summary"), max_length=800)
    )
    lifestyle_actions = list(sleep_plan.get("actions") or [])
    return {
        "health_interpretation": health_interpretation,
        "dietary_advice": list(diet_plan.get("actions") or [])[:4],
        "exercise_advice": list(exercise_plan.get("actions") or [])[:4],
        "lifestyle_advice": lifestyle_actions[:4],
        "observation_points": list(structured.get("three_day_observation") or [])[:6],
    }


def _build_schema2_sections(structured: dict[str, Any]) -> list[dict[str, Any]]:
    conditional_items = [
        f"{item.get('condition')}：{item.get('interpretation')}"
        for item in structured.get("conditional_analysis") or []
        if isinstance(item, dict)
    ]

    def plan_section(key: str, title: str) -> dict[str, Any]:
        plan = structured.get(key) if isinstance(structured.get(key), dict) else {}
        return {
            "section_key": key,
            "title": title,
            "content": plan.get("goal"),
            "items": list(plan.get("actions") or []),
            "metadata": {
                "frequency": plan.get("frequency"),
                "duration": plan.get("duration"),
                "observation_metrics": plan.get("observation_metrics") or [],
            },
        }

    return [
        {
            "section_key": "tongue_feature_explanation",
            "title": "舌象特征解释",
            "content": structured.get("tongue_feature_explanation"),
        },
        {"section_key": "conditional_analysis", "title": "条件性分析", "items": conditional_items},
        plan_section("diet_plan", "饮食计划"),
        plan_section("sleep_plan", "睡眠计划"),
        plan_section("exercise_plan", "运动计划"),
        {
            "section_key": "three_day_observation",
            "title": "未来三天观察",
            "items": structured.get("three_day_observation") or [],
        },
        {
            "section_key": "followup_questions",
            "title": "后续追问",
            "items": structured.get("followup_questions") or [],
        },
    ]


def _validate_schema2_report(structured: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if len(_clean_text(structured.get("comprehensive_summary"), max_length=2000)) < SUMMARY_MIN_LENGTH:
        errors.append("comprehensive_summary")
    evidence = structured.get("recognition_evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append("recognition_evidence")
    else:
        for item in evidence:
            if not isinstance(item, dict) or item.get("status") != "DETECTED":
                errors.append("recognition_evidence")
                break
    limits = structured.get("recognition_limits")
    if not isinstance(limits, list) or not limits:
        errors.append("recognition_limits")
    for key in PLAN_KEYS:
        plan = structured.get(key)
        if not isinstance(plan, dict) or not _clean_items(plan.get("actions"), max_items=6):
            errors.append(key)
    if len(_clean_items(structured.get("three_day_observation"), max_items=6)) < 3:
        errors.append("three_day_observation")
    if len(_clean_items(structured.get("followup_questions"), max_items=5)) < 3:
        errors.append("followup_questions")
    return sorted(set(errors))


def _compose_schema2_text(structured: dict[str, Any]) -> str:
    lines = ["本次结果", str(structured.get("comprehensive_summary") or "").strip()]
    for section in structured.get("sections") or []:
        if not isinstance(section, dict):
            continue
        title = str(section.get("title") or "").strip()
        content = str(section.get("content") or "").strip()
        items = [str(item).strip() for item in section.get("items") or [] if str(item).strip()]
        if not title or (not content and not items):
            continue
        lines.extend(["", title])
        if content:
            lines.append(content)
        if items:
            lines.append(_format_numbered(items))
    risk_tips = structured.get("risk_tips") or []
    if risk_tips:
        lines.extend(["", "提醒", str(risk_tips[0])])
    return "\n".join(lines).strip()


def _json_complete(text: str) -> bool:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return False
    try:
        json.loads(text[start : end + 1])
        return True
    except json.JSONDecodeError:
        return False


def _compose_schema2_report(
    *,
    payload: dict[str, Any],
    context: dict[str, Any],
    feature_names: list[str],
    user_description: str,
) -> tuple[str, dict[str, Any], str, list[str]] | None:
    structured, missing_fields = _normalize_schema2_payload(
        payload=payload,
        context=context,
        feature_names=feature_names,
        user_description=user_description,
    )
    if structured is None:
        return None
    mode = "llm_partial_repaired" if missing_fields else "llm_full"
    return _compose_schema2_text(structured), structured, mode, missing_fields


def _compose_user_facing_report(
    *,
    payload: dict[str, Any],
    feature_names: list[str],
    user_description: str,
) -> tuple[str, dict[str, Any]] | None:
    result_summary = _clean_text(payload.get("result_summary"), max_length=700)
    observation = _clean_items(payload.get("observation"), max_items=6)
    risk_reminder = _clean_text(payload.get("risk_reminder"), max_length=220)
    health_interpretation = _clean_text(payload.get("health_interpretation"), max_length=600)
    dietary_advice = _clean_items(payload.get("dietary_advice"), max_items=4)
    exercise_advice = _clean_items(payload.get("exercise_advice"), max_items=4)
    lifestyle_advice = _clean_items(payload.get("lifestyle_advice"), max_items=4)

    if not result_summary or not dietary_advice or not exercise_advice or not lifestyle_advice:
        return None

    if not health_interpretation:
        health_interpretation = _build_general_interpretation(feature_names)

    if not dietary_advice:
        dietary_advice = _build_dietary_advice(feature_names)

    if not exercise_advice:
        exercise_advice = _build_exercise_advice(feature_names)

    if not lifestyle_advice:
        lifestyle_advice = _build_lifestyle_advice(feature_names)

    daily_care = (dietary_advice + exercise_advice + lifestyle_advice)[:6]

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
    structured_answer.update(
        {
            "comprehensive_summary": result_summary,
            "health_interpretation": health_interpretation,
            "dietary_advice": dietary_advice,
            "exercise_advice": exercise_advice,
            "lifestyle_advice": lifestyle_advice,
            "risk_tips": [risk_reminder],
            "observation_points": observation,
        }
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
    tongue_features: dict[str, Any],
    rag_context: dict[str, Any],
    rag_query: str,
    user_description: str,
    state_snapshot: dict[str, Any] | None = None,
    personalization_signals: list[str] | None = None,
) -> tuple[str, dict[str, Any], str, list[str]] | None:
    settings = get_settings()
    context = _build_report_generation_context(
        feature_names=feature_names,
        tongue_features=tongue_features,
        rag_context=rag_context,
        rag_query=rag_query,
        user_description=user_description,
        state_snapshot=state_snapshot,
        personalization_signals=personalization_signals,
    )

    messages = [
        {
            "role": "system",
            "content": REPORT_SYNTHESIS_SYSTEM_PROMPT_V2,
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
        result = await get_chat_model_client().generate_with_metadata(
            messages=messages,
            temperature=settings.report_model_temperature,
            max_tokens=settings.report_model_max_tokens,
        )
        raw_content = result.content
        finish_reason = result.finish_reason
    except Exception as exc:
        logger.warning(
            "tongue_report_model_request_failed",
            extra={"rag_hit_count": len(rag_context.get("hits") or []), "error": str(exc)},
        )
        return None

    logger.info(
        "tongue_report_model_finished",
        extra={
            "finish_reason": finish_reason,
            "rag_hit_count": len(rag_context.get("hits") or []),
        },
    )
    payload = _extract_json_object(raw_content)
    if payload is None:
        suspected_truncated = finish_reason == "length" or not _json_complete(raw_content)
        logger.warning(
            "tongue_report_json_parse_failed",
            extra={
                "finish_reason": finish_reason,
                "suspected_truncated": suspected_truncated,
                "rag_hit_count": len(rag_context.get("hits") or []),
            },
        )
        return None

    composed = _compose_schema2_report(
        payload=payload,
        context=context,
        feature_names=feature_names,
        user_description=user_description,
    )
    if composed is None:
        logger.warning(
            "tongue_report_core_summary_missing",
            extra={
                "finish_reason": finish_reason,
                "rag_hit_count": len(rag_context.get("hits") or []),
            },
        )
        return None

    content, structured, mode, missing_fields = composed
    logger.info(
        "tongue_report_generation_mode",
        extra={
            "mode": mode,
            "missing_fields": missing_fields,
            "finish_reason": finish_reason,
            "rag_hit_count": len(rag_context.get("hits") or []),
        },
    )
    return content, structured, mode, missing_fields


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
        f"{_format_numbered((_build_dietary_advice(feature_names) + _build_exercise_advice(feature_names) + _build_lifestyle_advice(feature_names))[:6])}\n\n"
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

    dietary_advice = _build_dietary_advice(feature_names)
    exercise_advice = _build_exercise_advice(feature_names)
    lifestyle_advice = _build_lifestyle_advice(feature_names)
    risk_reminder = "以上内容用于一般健康知识说明和健康管理参考，不能替代医生诊断。"
    answer = _build_structured_report_answer(
        result_summary=summary,
        daily_care=(dietary_advice + exercise_advice + lifestyle_advice)[:6],
        observation=_build_observation_points(feature_names)[:4],
        risk_reminder=risk_reminder,
        feature_names=feature_names,
        user_description=user_description,
    )
    answer.update(
        {
            "comprehensive_summary": summary,
            "health_interpretation": _build_general_interpretation(feature_names),
            "dietary_advice": dietary_advice,
            "exercise_advice": exercise_advice,
            "lifestyle_advice": lifestyle_advice,
            "risk_tips": [risk_reminder],
            "observation_points": _build_observation_points(feature_names)[:4],
        }
    )
    return answer


def _build_template_schema2_structured_report_answer(
    *,
    tongue_features: dict[str, Any],
    feature_names: list[str],
    user_description: str,
    state_snapshot: dict[str, Any] | None = None,
    personalization_signals: list[str] | None = None,
) -> dict[str, Any]:
    context = _extract_feature_context(tongue_features)
    signals = personalization_signals or _personalization_signals(state_snapshot or {})
    summary = _default_summary(feature_names, user_description)
    if state_snapshot and state_snapshot.get("skipped"):
        summary += "你本次跳过了近3天状态补充，因此个性化信息不足，建议先按保守方式执行并继续记录。"
    elif signals:
        summary += f"本次已结合你反馈的{ '、'.join(signals[:3]) }，建议优先做轻量、可观察的调整。"

    diet_plan = _default_plan("diet_plan", feature_names)
    sleep_plan = _default_plan("sleep_plan", feature_names)
    exercise_plan = _default_plan("exercise_plan", feature_names)
    snapshot = state_snapshot or {}
    if snapshot.get("digestion_status") in {"BLOATING", "GREASY_REFLUX_DISCOMFORT"}:
        diet_plan["actions"] = [
            "因为你反馈饭后容易不舒服，所以这三天晚餐先控制在七分饱，可选小米粥、山药、南瓜、鸡蛋或豆腐等温和食物。"
        ] + diet_plan["actions"]
    if snapshot.get("sleep_status") in {"SHORT", "DIFFICULT_OR_WAKE", "IRREGULAR_LATE"}:
        sleep_plan["actions"] = [
            "因为你反馈近期睡眠不够稳定，所以先把入睡时间前移15到30分钟，睡前一小时减少刷屏和夜宵。"
        ] + sleep_plan["actions"]
    if "FATIGUE" in (snapshot.get("current_states") or []):
        exercise_plan["actions"] = [
            "因为你反馈容易疲乏，所以运动先选饭后15到20分钟舒缓步行或轻柔拉伸，以微微发热、不明显气喘为度。"
        ] + exercise_plan["actions"]

    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "comprehensive_summary": summary,
        "tongue_feature_explanation": _default_tongue_feature_explanation(feature_names),
        "conditional_analysis": _default_conditional_analysis(feature_names),
        "diet_plan": diet_plan,
        "sleep_plan": sleep_plan,
        "exercise_plan": exercise_plan,
        "three_day_observation": _build_observation_points(feature_names)[:6],
        "followup_questions": _default_followup_questions(),
        "risk_tips": [
            "以上内容用于一般健康知识说明和健康管理参考，不能替代医生诊断；如果不适明显或持续加重，请及时咨询医生。"
        ],
    }
    structured, _missing = _normalize_schema2_payload(
        payload=payload,
        context={
            "detected_features": context["detected_features"],
            "recognition_limits": context["recognition_limits"],
            "dimension_values": context["dimension_values"],
        },
        feature_names=feature_names,
        user_description=user_description,
    )
    if structured is None:
        return _build_structured_report_answer(
            result_summary=_default_summary(feature_names, user_description),
            daily_care=(
                _build_dietary_advice(feature_names)
                + _build_exercise_advice(feature_names)
                + _build_lifestyle_advice(feature_names)
            )[:6],
            observation=_build_observation_points(feature_names)[:4],
            risk_reminder=payload["risk_tips"][0],
            feature_names=feature_names,
            user_description=user_description,
        )
    return structured


def _build_tongue_analysis_report(
    *,
    state: AgentState,
    tongue_features: dict[str, Any],
    feature_names: list[str],
    rag_context: dict[str, Any],
    rag_query: str,
    user_description: str,
    state_snapshot: dict[str, Any] | None,
    personalization_signals: list[str],
    report_generation_mode: str,
    report_missing_fields: list[str],
    content: str,
    structured_answer: dict[str, Any] | None,
) -> dict[str, Any]:
    structured = structured_answer or {}
    comprehensive_summary = (
        _clean_text(structured.get("comprehensive_summary"), max_length=900)
        or _clean_text(structured.get("summary"), max_length=900)
        or content
    )
    health_interpretation = (
        _clean_text(structured.get("health_interpretation"), max_length=600)
        or _build_general_interpretation(feature_names)
    )
    dietary_advice = _clean_items(structured.get("dietary_advice"), max_items=4) or _build_dietary_advice(feature_names)
    exercise_advice = _clean_items(structured.get("exercise_advice"), max_items=4) or _build_exercise_advice(feature_names)
    lifestyle_advice = _clean_items(structured.get("lifestyle_advice"), max_items=4) or _build_lifestyle_advice(feature_names)
    risk_tips = _clean_items(structured.get("risk_tips"), max_items=2)
    if not risk_tips:
        risk_tip = (
            _clean_text(structured.get("risk_reminder"), max_length=220)
            or _clean_text(structured.get("disclaimer"), max_length=220)
            or "以上内容用于一般健康知识说明和健康管理参考，不能替代医生诊断。"
        )
        risk_tips = [risk_tip]

    report = TongueAnalysisReport(
        schema_version=REPORT_SCHEMA_VERSION,
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
        evidence_refs=_build_evidence_refs(rag_context),
        comprehensive_summary=comprehensive_summary,
        recognition_evidence=structured.get("recognition_evidence") or [],
        recognition_limits=structured.get("recognition_limits") or [],
        dimension_values=structured.get("dimension_values") or [],
        conditional_analysis=structured.get("conditional_analysis") or [],
        tongue_feature_explanation=_clean_text(
            structured.get("tongue_feature_explanation"),
            max_length=900,
        ),
        diet_plan=structured.get("diet_plan") or _default_plan("diet_plan", feature_names),
        sleep_plan=structured.get("sleep_plan") or _default_plan("sleep_plan", feature_names),
        exercise_plan=structured.get("exercise_plan") or _default_plan("exercise_plan", feature_names),
        three_day_observation=_clean_items(
            structured.get("three_day_observation"),
            max_items=6,
        ) or _build_observation_points(feature_names)[:6],
        followup_questions=_clean_items(
            structured.get("followup_questions"),
            max_items=5,
        ) or _default_followup_questions(),
        tongue_features=_build_tongue_feature_dicts(tongue_features),
        health_interpretation=health_interpretation,
        dietary_advice=dietary_advice,
        exercise_advice=exercise_advice,
        lifestyle_advice=lifestyle_advice,
        risk_tips=risk_tips,
        summary=content,
        health_notes=_build_health_notes(feature_names),
        versions={
            "report_schema": "tongue_analysis_report_v2.0",
            "feature_schema": "tongue_standard_features_v1.0",
            "rag_engine": rag_context.get("retrieval_engine"),
            "answer_engine": rag_context.get("answer_engine"),
        },
        metadata={
            "user_description": user_description,
            "state_snapshot": state_snapshot or {},
            "personalization_signals": personalization_signals,
            "report_generation": {
                "mode": report_generation_mode,
                "missing_fields": report_missing_fields,
                "mode_counts": {report_generation_mode: 1},
                "inputs": [
                    "detected_features",
                    "not_evaluated_dimensions",
                    "unsupported_dimensions",
                    "user_description",
                    "state_snapshot",
                    "personalization_signals",
                    "rag_summary",
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
                "dietary_advice": dietary_advice,
                "exercise_advice": exercise_advice,
                "lifestyle_advice": lifestyle_advice,
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
    state_snapshot = _extract_state_snapshot(state)
    personalization_signals = _personalization_signals(state_snapshot)
    feature_rag_query = str(tongue_features.get("rag_query") or "").strip()
    if not feature_rag_query:
        feature_rag_query = _build_feature_name_rag_query(feature_names)
    rag_query = _build_augmented_rag_query(feature_rag_query, user_description, state_snapshot)

    if rag_query:
        rag_context = await answer_with_rag(rag_query)
    else:
        rag_context = fallback_rag_response("empty_tongue_feature_rag_query")

    generated_answer = await _generate_integrated_report_answer(
        feature_names=feature_names,
        tongue_features=tongue_features,
        rag_context=rag_context,
        rag_query=rag_query,
        user_description=user_description,
        state_snapshot=state_snapshot,
        personalization_signals=personalization_signals,
    )
    content = None
    structured_answer = None
    report_generation_mode = "llm_full"
    report_missing_fields: list[str] = []
    if isinstance(generated_answer, tuple):
        if len(generated_answer) == 4:
            content, structured_answer, report_generation_mode, report_missing_fields = generated_answer
        elif len(generated_answer) == 2:
            content, structured_answer = generated_answer
    elif isinstance(generated_answer, str):
        content = generated_answer

    if not content:
        report_generation_mode = "template_fallback"
        structured_answer = _build_template_schema2_structured_report_answer(
            tongue_features=tongue_features,
            feature_names=feature_names,
            user_description=user_description,
            state_snapshot=state_snapshot,
            personalization_signals=personalization_signals,
        )
        report_missing_fields = _validate_schema2_report(structured_answer)
        content = _compose_schema2_text(structured_answer)
    elif structured_answer is None:
        structured_answer = _build_template_schema2_structured_report_answer(
            tongue_features=tongue_features,
            feature_names=feature_names,
            user_description=user_description,
            state_snapshot=state_snapshot,
            personalization_signals=personalization_signals,
        )

    draft_report = _build_tongue_analysis_report(
        state=state,
        tongue_features=tongue_features,
        feature_names=feature_names,
        rag_context=rag_context,
        rag_query=rag_query,
        user_description=user_description,
        state_snapshot=state_snapshot,
        personalization_signals=personalization_signals,
        report_generation_mode=report_generation_mode,
        report_missing_fields=report_missing_fields,
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
