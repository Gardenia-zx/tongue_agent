from typing import Any

from app.agent.state import AgentState
from app.integrations.tongue_model_client import (
    TongueModelError,
    get_tongue_model_client,
)
from app.tongue.feature_mapping import normalize_tongue_model_result


def _extract_extra(state: AgentState) -> dict[str, Any]:
    client_context = state.get("client_context") or {}
    extra = client_context.get("extra") or {}
    if isinstance(extra, dict):
        return extra
    return {}


def _extract_model_result(state: AgentState) -> dict[str, Any] | None:
    tongue_features = state.get("tongue_features")
    if isinstance(tongue_features, dict) and (
        tongue_features.get("features")
        or tongue_features.get("raw_detections")
        or tongue_features.get("data")
    ):
        return tongue_features

    extra = _extract_extra(state)
    model_result = extra.get("tongue_model_result")
    if isinstance(model_result, dict):
        return model_result

    return None


def _extract_image_input(state: AgentState) -> tuple[str, str] | None:
    extra = _extract_extra(state)

    for key in ["tongue_image_path", "image_path"]:
        value = extra.get(key)
        if isinstance(value, str) and value.strip():
            return "path", value.strip()

    for key in ["tongue_image_url", "image_url"]:
        value = extra.get(key)
        if isinstance(value, str) and value.strip():
            return "url", value.strip()

    return None


async def _predict_model_result(state: AgentState) -> dict[str, Any] | None:
    model_result = _extract_model_result(state)
    if model_result is not None:
        return model_result

    image_input = _extract_image_input(state)
    if image_input is None:
        return None

    source_type, value = image_input
    client = get_tongue_model_client()
    if source_type == "path":
        return await client.predict_image_path(value)

    return await client.predict_image_url(value)


def _detected_feature_names(standard_features: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for group_key in ["overall", "tongue_body", "coating", "regions"]:
        group = standard_features.get(group_key)
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


async def tongue_analysis_node(state: AgentState) -> AgentState:
    try:
        model_result = await _predict_model_result(state)
    except TongueModelError as exc:
        return {
            **state,
            "current_node": "tongue_analysis_node",
            "response_message": {
                "role": "assistant",
                "content_type": "text",
                "content": "舌象图片识别服务暂时不可用，请稍后再试，或重新上传一张清晰的舌象图片。",
            },
            "next_action": {
                "type": "TONGUE_MODEL_FAILED",
                "payload": {
                    "status": "FAILED",
                    "reason": str(exc),
                    "thread_id": state.get("thread_id"),
                    "report_id": state.get("report_id"),
                    "task_id": state.get("task_id"),
                },
            },
        }

    if model_result is None:
        return {
            **state,
            "current_node": "tongue_analysis_node",
            "response_message": {
                "role": "assistant",
                "content_type": "text",
                "content": "可以开始舌象分析。请上传清晰的舌象图片，尽量使用自然光，避免滤镜、强反光和刚吃过有色食物后拍摄。",
            },
            "next_action": {
                "type": "START_OR_RESUME_TONGUE_ANALYSIS",
                "payload": {
                    "status": "WAIT_USER_ANSWER",
                    "report_id": state.get("report_id"),
                    "task_id": state.get("task_id"),
                    "thread_id": state.get("thread_id"),
                    "required_input": "tongue_image",
                },
            },
        }

    standard_features = normalize_tongue_model_result(model_result)
    detected_names = _detected_feature_names(standard_features)
    rag_query = standard_features.get("rag_query") or ""
    feature_text = "、".join(detected_names) if detected_names else "舌象图像特征"

    draft_report = {
        "schema_version": "1.0",
        "report_type": "tongue_analysis_mvp",
        "standard_features": standard_features,
        "feature_rag_query": rag_query,
    }

    return {
        **state,
        "current_node": "tongue_analysis_node",
        "tongue_features": standard_features,
        "draft_report": draft_report,
        "response_message": {
            "role": "assistant",
            "content_type": "text",
            "content": (
                f"本次图像识别到的主要舌象特征包括：{feature_text}。"
                "接下来可以结合知识库资料，给出这些特征的一般健康知识说明。"
            ),
        },
        "next_action": {
            "type": "TONGUE_FEATURES_READY",
            "payload": {
                "status": "COMPLETED",
                "report_id": state.get("report_id"),
                "task_id": state.get("task_id"),
                "thread_id": state.get("thread_id"),
                "detected_feature_codes": standard_features.get(
                    "detected_feature_codes", []
                ),
                "rag_query": rag_query,
            },
        },
    }
