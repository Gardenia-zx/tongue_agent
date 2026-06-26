from dataclasses import dataclass
from typing import Awaitable, Callable

from app.agent.context_builder import effective_user_query, query_context_from_state
from app.agent.nodes.general_chat_node import general_chat_node
from app.agent.nodes.health_qa_node import health_qa_node
from app.agent.nodes.report_followup_node import report_followup_node
from app.agent.nodes.tongue_analysis_node import tongue_analysis_node
from app.agent.nodes.tongue_report_node import tongue_report_node
from app.agent.state import AgentState
from app.agent.tool_policy import can_use_report_followup

ToolHandler = Callable[[AgentState], Awaitable[AgentState]]


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    handler: ToolHandler


GENERAL_CHAT_TOOL = "general_chat_tool"
HEALTH_QA_TOOL = "health_qa_tool"
REPORT_FOLLOWUP_TOOL = "report_followup_tool"
TONGUE_IMAGE_ANALYSIS_TOOL = "tongue_image_analysis_tool"
TONGUE_REPORT_GENERATE_TOOL = "tongue_report_generate_tool"

TOOLS: dict[str, AgentTool] = {
    GENERAL_CHAT_TOOL: AgentTool(GENERAL_CHAT_TOOL, "回答系统能力和普通聊天问题。", general_chat_node),
    HEALTH_QA_TOOL: AgentTool(HEALTH_QA_TOOL, "基于知识库回答一般健康知识问题。", health_qa_node),
    REPORT_FOLLOWUP_TOOL: AgentTool(REPORT_FOLLOWUP_TOOL, "基于当前活动报告回答明确的报告追问。", report_followup_node),
    TONGUE_IMAGE_ANALYSIS_TOOL: AgentTool(TONGUE_IMAGE_ANALYSIS_TOOL, "调用图像识别模型提取标准化舌象特征。", tongue_analysis_node),
    TONGUE_REPORT_GENERATE_TOOL: AgentTool(TONGUE_REPORT_GENERATE_TOOL, "根据本轮识别特征生成结构化健康参考报告。", tongue_report_node),
}


def _intent_route(state: AgentState) -> str:
    intent_result = state.get("intent_result") or {}
    return str(intent_result.get("route_target") or "")


def _route_hint(state: AgentState) -> str:
    return str(query_context_from_state(state).get("route_hint") or "")


def _has_tongue_image_input(state: AgentState) -> bool:
    client_context = state.get("client_context") or {}
    extra = client_context.get("extra") or {}
    if isinstance(extra, dict):
        for key in ("tongue_model_result", "tongue_image_path", "image_path", "tongue_image_url", "image_url"):
            if extra.get(key):
                return True
    message = state.get("message") or {}
    attachments = message.get("attachments") or []
    if isinstance(attachments, list):
        for attachment in attachments:
            if isinstance(attachment, dict) and (
                attachment.get("file_type") == "image"
                or attachment.get("purpose") == "tongue_image"
            ):
                return True
    return False


def _query_contains_health_terms(state: AgentState) -> bool:
    compact = "".join(effective_user_query(state).split())
    return any(
        keyword in compact
        for keyword in (
            "湿气", "脾虚", "白苔", "黄苔", "舌苔", "齿痕", "裂纹",
            "食欲", "腹胀", "大便", "睡眠", "疲乏", "饮食", "忌口",
            "调理", "运动", "作息", "是什么意思", "怎么办",
        )
    )


def choose_agent_tool(state: AgentState) -> tuple[AgentTool, str]:
    route_hint = _route_hint(state)
    route_target = _intent_route(state)

    if _has_tongue_image_input(state):
        return TOOLS[TONGUE_IMAGE_ANALYSIS_TOOL], "tongue_image_input_available"
    if route_hint == "tongue_analysis_subgraph":
        return TOOLS[TONGUE_IMAGE_ANALYSIS_TOOL], "query_context_route_tongue_analysis"
    if route_target == "tongue_analysis_subgraph":
        return TOOLS[TONGUE_IMAGE_ANALYSIS_TOOL], "intent_route_tongue_analysis"

    if can_use_report_followup(state):
        return TOOLS[REPORT_FOLLOWUP_TOOL], "explicit_report_binding"

    if route_hint == "health_qa_subgraph":
        return TOOLS[HEALTH_QA_TOOL], "query_context_route_health_qa"
    if route_target == "health_qa_subgraph":
        return TOOLS[HEALTH_QA_TOOL], "intent_route_health_qa"

    if route_hint == "general_chat_subgraph":
        if _query_contains_health_terms(state):
            return TOOLS[HEALTH_QA_TOOL], "general_hint_but_health_query"
        return TOOLS[GENERAL_CHAT_TOOL], "query_context_route_general_chat"
    if route_target == "general_chat_subgraph":
        if _query_contains_health_terms(state):
            return TOOLS[HEALTH_QA_TOOL], "general_route_but_health_query"
        return TOOLS[GENERAL_CHAT_TOOL], "intent_route_general_chat"

    if route_target == "report_explanation_subgraph":
        return TOOLS[GENERAL_CHAT_TOOL], "report_context_not_explicit_or_missing"
    if _query_contains_health_terms(state):
        return TOOLS[HEALTH_QA_TOOL], "health_terms_without_report_binding"
    return TOOLS[GENERAL_CHAT_TOOL], "default_general_chat"
