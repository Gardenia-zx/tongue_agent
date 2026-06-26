from dataclasses import dataclass
from typing import Awaitable, Callable

from app.agent.context_builder import active_report_from_state, effective_user_query, query_context_from_state
from app.agent.nodes.general_chat_node import general_chat_node
from app.agent.nodes.health_qa_node import health_qa_node
from app.agent.nodes.report_followup_node import report_followup_node
from app.agent.nodes.tongue_analysis_node import tongue_analysis_node
from app.agent.nodes.tongue_report_node import tongue_report_node
from app.agent.state import AgentState


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
    GENERAL_CHAT_TOOL: AgentTool(
        name=GENERAL_CHAT_TOOL,
        description="回答系统能力、普通聊天和不需要外部检索的开放式问题。",
        handler=general_chat_node,
    ),
    HEALTH_QA_TOOL: AgentTool(
        name=HEALTH_QA_TOOL,
        description="调用 RAG 知识库回答一般健康知识问题。",
        handler=health_qa_node,
    ),
    REPORT_FOLLOWUP_TOOL: AgentTool(
        name=REPORT_FOLLOWUP_TOOL,
        description="基于当前活动舌象报告、最近对话和必要 RAG 结果回答报告追问。",
        handler=report_followup_node,
    ),
    TONGUE_IMAGE_ANALYSIS_TOOL: AgentTool(
        name=TONGUE_IMAGE_ANALYSIS_TOOL,
        description="调用舌象图像识别模型，提取标准化舌象特征。",
        handler=tongue_analysis_node,
    ),
    TONGUE_REPORT_GENERATE_TOOL: AgentTool(
        name=TONGUE_REPORT_GENERATE_TOOL,
        description="基于舌象识别特征、用户描述和 RAG 资料生成结构化舌象健康参考报告。",
        handler=tongue_report_node,
    ),
}


def _intent_route(state: AgentState) -> str:
    intent_result = state.get("intent_result") or {}
    return str(intent_result.get("route_target") or "")


def _target_type(state: AgentState) -> str:
    query_context = query_context_from_state(state)
    reference = query_context.get("reference_resolution") or {}
    return str(reference.get("target_type") or "")


def _has_report_context(state: AgentState) -> bool:
    return bool(active_report_from_state(state))


def _has_tongue_image_input(state: AgentState) -> bool:
    client_context = state.get("client_context") or {}
    extra = client_context.get("extra") or {}
    if isinstance(extra, dict):
        for key in [
            "tongue_model_result",
            "tongue_image_path",
            "image_path",
            "tongue_image_url",
            "image_url",
        ]:
            value = extra.get(key)
            if value:
                return True

    message = state.get("message") or {}
    attachments = message.get("attachments") or []
    if isinstance(attachments, list):
        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            if attachment.get("file_type") == "image" or attachment.get("purpose") == "tongue_image":
                return True

    return False


def _query_contains_health_terms(state: AgentState) -> bool:
    query = effective_user_query(state)
    compact = "".join(query.split())
    return any(
        keyword in compact
        for keyword in [
            "湿气",
            "脾虚",
            "白苔",
            "黄苔",
            "舌苔",
            "饮食",
            "忌口",
            "调理",
            "是什么意思",
        ]
    )


def choose_agent_tool(state: AgentState) -> tuple[AgentTool, str]:
    target_type = _target_type(state)
    route_target = _intent_route(state)
    has_report = _has_report_context(state)

    if _has_tongue_image_input(state):
        return TOOLS[TONGUE_IMAGE_ANALYSIS_TOOL], "tongue_image_input_available"

    if target_type == "TONGUE_ANALYSIS":
        return TOOLS[TONGUE_IMAGE_ANALYSIS_TOOL], "query_context_target_tongue_analysis"

    if route_target == "tongue_analysis_subgraph":
        return TOOLS[TONGUE_IMAGE_ANALYSIS_TOOL], "intent_route_tongue_analysis"

    if target_type == "REPORT" and has_report:
        return TOOLS[REPORT_FOLLOWUP_TOOL], "query_context_target_report"

    if target_type == "HEALTH_QA":
        return TOOLS[HEALTH_QA_TOOL], "query_context_target_health_qa"

    if target_type == "GENERAL_CHAT":
        return TOOLS[GENERAL_CHAT_TOOL], "query_context_target_general_chat"

    if route_target == "report_explanation_subgraph" and has_report:
        return TOOLS[REPORT_FOLLOWUP_TOOL], "intent_route_report_with_active_report"

    if route_target == "health_qa_subgraph":
        return TOOLS[HEALTH_QA_TOOL], "intent_route_health_qa"

    if route_target == "general_chat_subgraph":
        if has_report and _query_contains_health_terms(state):
            return TOOLS[REPORT_FOLLOWUP_TOOL], "general_route_but_report_health_context"
        return TOOLS[GENERAL_CHAT_TOOL], "intent_route_general_chat"

    if has_report and _query_contains_health_terms(state):
        return TOOLS[REPORT_FOLLOWUP_TOOL], "active_report_with_health_terms"

    return TOOLS[GENERAL_CHAT_TOOL], "default_general_chat"
