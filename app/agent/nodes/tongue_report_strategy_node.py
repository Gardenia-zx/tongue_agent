"""Report strategy for the dedicated tongue-image workflow."""

from typing import Any

from app.agent.nodes import tongue_report_node as report_impl
from app.agent.state import AgentState


REPORT_PROMPT = """你是舌象健康管理报告生成助手。必须返回完整 JSON，不要输出 Markdown 或额外文字。

规则：
1. schema_version 固定为 2.0。
2. recognition_evidence 只能包含 detected_features 中 status=DETECTED 的项目。
3. recognition_limits 仅记录未评估和模型暂不支持的维度。
4. dimension_values 是报告实际采用的完整特征集合。DETECTED 是模型结果，DEFAULT 是系统基线值。
5. DETECTED 优先级高于 DEFAULT，不允许默认值覆盖模型结果。
6. comprehensive_summary、tongue_feature_explanation、饮食、睡眠和运动计划可以直接结合 DETECTED 与 DEFAULT 进行完整分析。
7. 用户正文不要反复说明字段来源，也不要写成模型调试说明。
8. conditional_analysis 用于补充其他情况，不要求默认值只能出现在条件句中。
9. diet_plan、sleep_plan、exercise_plan 必须包含 goal、actions、frequency、duration、observation_metrics。
10. 建议必须具体可执行，并说明观察周期；risk_tips 需要包含必要的健康提示。

返回字段：schema_version、comprehensive_summary、recognition_evidence、recognition_limits、dimension_values、tongue_feature_explanation、conditional_analysis、diet_plan、sleep_plan、exercise_plan、three_day_observation、followup_questions、risk_tips。
"""


_original_template_builder = report_impl._build_template_schema2_structured_report_answer


def _adopted_feature_text(tongue_features: dict[str, Any]) -> str:
    context = report_impl._extract_feature_context(tongue_features)
    values = {
        str(item.get("dimension")): str(item.get("value") or "")
        for item in context.get("dimension_values") or []
        if isinstance(item, dict)
    }
    coating_color = values.get("coating.color", "").removesuffix("色")
    coating_thickness = values.get("coating.thickness", "")
    parts = [
        f"舌质{values.get('tongue_body.color', '')}" if values.get("tongue_body.color") else "",
        f"舌体{values.get('tongue_body.shape', '')}" if values.get("tongue_body.shape") else "",
        (
            f"舌体质地{values.get('tongue_body.texture', '')}"
            if values.get("tongue_body.texture")
            else ""
        ),
        f"舌苔{coating_color}{coating_thickness}" if coating_color or coating_thickness else "",
        f"舌苔{values.get('coating.moisture', '')}" if values.get("coating.moisture") else "",
        (
            f"舌苔质地{values.get('coating.texture', '')}"
            if values.get("coating.texture")
            else ""
        ),
        values.get("regions", ""),
    ]
    return "、".join(part for part in parts if part)


def _template_builder_with_defaults(
    *,
    tongue_features: dict[str, Any],
    feature_names: list[str],
    user_description: str,
    state_snapshot: dict[str, Any] | None = None,
    personalization_signals: list[str] | None = None,
) -> dict[str, Any]:
    """
    在原始 Schema 2.0 模板的基础上补充完整采用的舌象维度。

    必须完整转发原模板函数的参数，避免新增报告上下文后，
    策略包装器与原始函数签名不一致。
    """
    structured = _original_template_builder(
        tongue_features=tongue_features,
        feature_names=feature_names,
        user_description=user_description,
        state_snapshot=state_snapshot,
        personalization_signals=personalization_signals,
    )

    adopted = _adopted_feature_text(tongue_features)
    if not adopted:
        return structured

    # 保留原始模板根据近期状态生成的个性化内容，
    # 只在前面补充本次实际采用的舌象特征，不再整段覆盖。
    existing_summary = str(
        structured.get("comprehensive_summary")
        or structured.get("summary")
        or ""
    ).strip()

    feature_summary = f"本次舌象整体可概括为：{adopted}。"

    if feature_summary not in existing_summary:
        summary = f"{feature_summary}{existing_summary}"
    else:
        summary = existing_summary

    existing_explanation = str(
        structured.get("tongue_feature_explanation")
        or structured.get("health_interpretation")
        or ""
    ).strip()

    feature_explanation = f"综合采用的舌象特征为{adopted}。"

    if feature_explanation not in existing_explanation:
        explanation = f"{feature_explanation}{existing_explanation}"
    else:
        explanation = existing_explanation

    structured["comprehensive_summary"] = summary
    structured["summary"] = summary
    structured["tongue_feature_explanation"] = explanation
    structured["health_interpretation"] = explanation

    for section in structured.get("sections") or []:
        if not isinstance(section, dict):
            continue

        section_key = section.get("section_key") or section.get("sectionKey")
        if section_key == "tongue_feature_explanation":
            section["content"] = explanation

    return structured


report_impl.REPORT_SYNTHESIS_SYSTEM_PROMPT_V2 = REPORT_PROMPT
report_impl._build_template_schema2_structured_report_answer = _template_builder_with_defaults


async def tongue_report_strategy_node(state: AgentState) -> AgentState:
    return await report_impl.tongue_report_node(state)
