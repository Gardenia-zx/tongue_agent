"""Report strategy for the dedicated tongue-image workflow."""

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


report_impl.REPORT_SYNTHESIS_SYSTEM_PROMPT_V2 = REPORT_PROMPT


async def tongue_report_strategy_node(state: AgentState) -> AgentState:
    return await report_impl.tongue_report_node(state)
