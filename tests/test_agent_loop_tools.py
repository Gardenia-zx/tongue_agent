import json
import unittest
from unittest.mock import AsyncMock, patch

from app.agent.nodes.agent_gate_node import agent_gate_node, select_agent_gate_next
from app.agent.nodes.agent_loop_node import agent_loop_node
from app.agent.nodes.query_rewrite_node import query_rewrite_node
from app.agent.tooling import (
    TOOLS,
    AgentTool,
    TONGUE_IMAGE_ANALYSIS_TOOL,
    TONGUE_REPORT_GENERATE_TOOL,
)


class FakeToolCallingClient:
    def __init__(self, messages: list[dict]) -> None:
        self.messages = messages
        self.calls: list[dict] = []

    async def chat(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        if not self.messages:
            raise AssertionError("FakeToolCallingClient has no more messages")
        return self.messages.pop(0)


def tool_call_message(tool_name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def final_answer_message(
    *,
    content: str,
    answer_type: str,
    title: str,
) -> dict:
    return {
        "role": "assistant",
        "content": json.dumps(
            {
                "type": "final_answer",
                "content": content,
                "structured_content": {
                    "schema_version": "1.0",
                    "answer_type": answer_type,
                    "title": title,
                    "summary": content,
                    "sections": [{"title": title, "items": [content]}],
                    "disclaimer": "以上内容用于一般健康知识说明和健康管理参考，不能替代医生诊断。",
                },
            },
            ensure_ascii=False,
        ),
    }


class TestAgentLoopTools(unittest.IsolatedAsyncioTestCase):
    def _state_after_tongue_report(self) -> dict:
        active_report = {
            "report_id": 6,
            "summary": "本次图片主要识别到白苔。饮食上建议清淡规律，少生冷甜腻。",
            "feature_summary": "白苔",
            "detected_feature_codes": ["coating.color.white"],
            "rag_evidence": [],
        }
        last_answer = {
            "message_id": 200,
            "role": "assistant",
            "content_type": "text",
            "content": "本次图片主要识别到白苔。健康管理建议包括饮食清淡、规律作息和继续观察。",
            "report_id": 6,
            "metadata": {
                "node_name": "tongue_report_node",
                "route_target": "tongue_analysis_subgraph",
                "answer_type": "TONGUE_REPORT",
                "structured_content": {
                    "answer_type": "TONGUE_REPORT",
                    "title": "舌象健康参考报告",
                    "sections": [
                        {"title": "识别结果", "items": ["白苔"]},
                        {"title": "健康管理建议", "items": ["饮食清淡", "规律作息"]},
                    ],
                },
            },
        }
        context_bundle = {
            "conversation_id": "90001",
            "active_report": active_report,
            "last_final_answer": last_answer,
            "recent_messages": [last_answer],
        }
        return {
            "schema_version": "1.0",
            "request_id": "agent_loop_report_test",
            "trace_id": "trace_agent_loop_report_test",
            "user_id": 10001,
            "thread_id": "thread_agent_loop_report_test",
            "conversation_id": "90001",
            "message": {
                "role": "user",
                "content_type": "text",
                "content": "更详细一点",
                "attachments": [],
            },
            "context_bundle": context_bundle,
            "client_context": {
                "page": "ai_chat",
                "extra": {"context_bundle": context_bundle},
            },
            "intent_result": {
                "primary_intent": "GENERAL_CHAT",
                "detected_intent": "GENERAL_CHAT",
                "route_target": "general_chat_subgraph",
                "decision": "ROUTE",
                "risk_level": "LOW",
            },
        }

    def _state_after_diet_answer(self) -> dict:
        active_report = {
            "report_id": 6,
            "summary": "本次图片主要识别到白苔，用户描述饭后腹胀、大便黏、食欲不振。",
            "feature_summary": "白苔",
            "detected_feature_codes": ["coating.color.white"],
            "rag_evidence": [],
        }
        last_answer = {
            "message_id": 201,
            "role": "assistant",
            "content_type": "text",
            "content": "饮食建议：近期先减少生冷、油腻、甜腻，三餐规律，观察腹胀和大便状态。",
            "report_id": 6,
            "metadata": {
                "node_name": "report_followup_node",
                "route_target": "report_followup_subgraph",
                "answer_type": "REPORT_FOLLOWUP",
                "structured_content": {
                    "answer_type": "REPORT_FOLLOWUP",
                    "title": "饮食建议",
                    "sections": [
                        {
                            "title": "饮食建议",
                            "items": ["少生冷甜腻", "三餐规律"],
                        }
                    ],
                },
            },
        }
        context_bundle = {
            "conversation_id": "90001",
            "active_report": active_report,
            "last_final_answer": last_answer,
            "recent_messages": [last_answer],
        }
        return {
            "schema_version": "1.0",
            "request_id": "agent_loop_test",
            "trace_id": "trace_agent_loop_test",
            "user_id": 10001,
            "thread_id": "thread_agent_loop_test",
            "conversation_id": "90001",
            "message": {
                "role": "user",
                "content_type": "text",
                "content": "还是不够详细",
                "attachments": [],
            },
            "context_bundle": context_bundle,
            "client_context": {
                "page": "ai_chat",
                "extra": {"context_bundle": context_bundle},
            },
            "intent_result": {
                "primary_intent": "GENERAL_CHAT",
                "detected_intent": "GENERAL_CHAT",
                "route_target": "general_chat_subgraph",
                "decision": "ROUTE",
                "risk_level": "LOW",
            },
        }

    def _image_analysis_state(self) -> dict:
        return {
            "schema_version": "1.0",
            "request_id": "agent_loop_image_test",
            "trace_id": "trace_agent_loop_image_test",
            "user_id": 10001,
            "thread_id": "thread_agent_loop_image_test",
            "conversation_id": "90002",
            "report_id": 7,
            "task_id": 8,
            "message": {
                "role": "user",
                "content_type": "mixed",
                "content": "我想做一次舌象分析",
                "attachments": [
                    {
                        "file_id": 1,
                        "file_type": "image",
                        "purpose": "tongue_image",
                    }
                ],
            },
            "client_context": {
                "page": "tongue_analyze",
                "extra": {
                    "image_path": "D:\\tongue\\storage\\uploads\\tongue.jpg",
                },
            },
            "intent_result": {
                "primary_intent": "TONGUE_ANALYSIS_START",
                "detected_intent": "TONGUE_ANALYSIS_START",
                "route_target": "tongue_analysis_subgraph",
                "decision": "ROUTE",
                "risk_level": "LOW",
            },
        }

    async def test_agent_loop_inherits_diet_focus_and_calls_report_tool(self) -> None:
        state = await query_rewrite_node(self._state_after_diet_answer())
        gated = await agent_gate_node(state)

        self.assertEqual("agent_loop_node", select_agent_gate_next(gated))
        self.assertEqual(
            "DIET_ADVICE",
            gated["query_context"]["reference_resolution"]["target_focus"],
        )
        self.assertIn("饮食", gated["query_context"]["standalone_query"])

        fake_client = FakeToolCallingClient(
            [
                tool_call_message(
                    "report_followup_tool",
                    {"reason": "上一轮是饮食建议，当前用户要求继续展开"},
                ),
                final_answer_message(
                    content="可以，我继续把饮食建议展开。",
                    answer_type="REPORT_FOLLOWUP",
                    title="饮食建议",
                ),
            ]
        )

        with patch(
            "app.agent.nodes.agent_loop_node.get_chat_model_client",
            return_value=fake_client,
        ), patch(
            "app.agent.nodes.report_followup_node.generate_report_followup_reply",
            AsyncMock(
                return_value=(
                    "可以，我继续把饮食建议展开。",
                    None,
                    {
                        "answer_type": "REPORT_FOLLOWUP",
                        "title": "饮食建议",
                        "sections": [{"title": "饮食建议", "items": ["少生冷甜腻"]}],
                    },
                )
            ),
        ) as mocked_followup:
            result = await agent_loop_node(gated)

        mocked_followup.assert_awaited_once()
        self.assertEqual("report_followup_tool", result["agent_loop"]["selected_tool"])
        self.assertEqual("agent_loop_node", result["current_node"])
        self.assertEqual("饮食建议", result["response_message"]["structured_content"]["title"])
        self.assertEqual(2, len(fake_client.calls))

    async def test_short_followup_after_initial_report_does_not_become_diet_advice(self) -> None:
        state = await query_rewrite_node(self._state_after_tongue_report())
        gated = await agent_gate_node(state)

        self.assertEqual("agent_loop_node", select_agent_gate_next(gated))
        self.assertEqual(
            "DETAILED_REPORT",
            gated["query_context"]["reference_resolution"]["target_focus"],
        )
        self.assertIn("健康参考报告", gated["query_context"]["standalone_query"])

        fake_client = FakeToolCallingClient(
            [
                tool_call_message(
                    "report_followup_tool",
                    {"reason": "上一轮是初始舌象报告，用户要求更详细"},
                ),
                final_answer_message(
                    content="可以，我整理成更详细的舌象健康参考报告。",
                    answer_type="DETAILED_TONGUE_REPORT",
                    title="详细舌象报告",
                ),
            ]
        )

        with patch(
            "app.agent.nodes.agent_loop_node.get_chat_model_client",
            return_value=fake_client,
        ), patch(
            "app.agent.nodes.report_followup_node.generate_report_followup_reply",
            AsyncMock(
                return_value=(
                    "可以，我整理成更详细的舌象健康参考报告。",
                    None,
                    {
                        "answer_type": "DETAILED_TONGUE_REPORT",
                        "title": "详细舌象报告",
                        "sections": [{"title": "识别结果", "items": ["白苔"]}],
                    },
                )
            ),
        ):
            result = await agent_loop_node(gated)

        self.assertEqual("report_followup_tool", result["agent_loop"]["selected_tool"])
        self.assertEqual("详细舌象报告", result["response_message"]["structured_content"]["title"])
        self.assertEqual(2, len(fake_client.calls))

    async def test_image_analysis_runs_inside_agent_loop_with_report_generation(self) -> None:
        async def fake_image_tool(state: dict) -> dict:
            return {
                **state,
                "current_node": "tongue_analysis_node",
                "tongue_features": {
                    "detected_feature_codes": ["coating.color.white"],
                    "rag_query": "白苔 舌象观察",
                },
                "next_action": {
                    "type": "TONGUE_FEATURES_READY",
                    "payload": {
                        "status": "COMPLETED",
                        "detected_feature_codes": ["coating.color.white"],
                        "rag_query": "白苔 舌象观察",
                    },
                },
            }

        async def fake_report_tool(state: dict) -> dict:
            return {
                **state,
                "current_node": "tongue_report_node",
                "draft_report": {"report_status": "COMPLETED"},
                "response_message": {
                    "role": "assistant",
                    "content_type": "text",
                    "content": "本次图片主要识别到白苔。",
                    "structured_content": {
                        "answer_type": "TONGUE_REPORT",
                        "title": "舌象健康参考报告",
                    },
                },
                "next_action": {
                    "type": "RESPOND_TO_USER",
                    "payload": {
                        "status": "COMPLETED",
                        "answer_type": "TONGUE_REPORT",
                        "route_target": "tongue_analysis_subgraph",
                        "draft_report": {"report_status": "COMPLETED"},
                    },
                },
            }

        original_image_tool = TOOLS[TONGUE_IMAGE_ANALYSIS_TOOL]
        original_report_tool = TOOLS[TONGUE_REPORT_GENERATE_TOOL]
        fake_client = FakeToolCallingClient(
            [
                tool_call_message(
                    TONGUE_IMAGE_ANALYSIS_TOOL,
                    {"reason": "用户上传了舌象图片"},
                    call_id="call_image",
                ),
                tool_call_message(
                    TONGUE_REPORT_GENERATE_TOOL,
                    {"reason": "舌象特征已识别，需要生成报告"},
                    call_id="call_report",
                ),
                final_answer_message(
                    content="本次图片主要识别到白苔。",
                    answer_type="TONGUE_REPORT",
                    title="舌象健康参考报告",
                ),
            ]
        )
        try:
            TOOLS[TONGUE_IMAGE_ANALYSIS_TOOL] = AgentTool(
                name=TONGUE_IMAGE_ANALYSIS_TOOL,
                description="fake image tool",
                handler=fake_image_tool,
            )
            TOOLS[TONGUE_REPORT_GENERATE_TOOL] = AgentTool(
                name=TONGUE_REPORT_GENERATE_TOOL,
                description="fake report tool",
                handler=fake_report_tool,
            )

            state = await query_rewrite_node(self._image_analysis_state())
            gated = await agent_gate_node(state)
            with patch(
                "app.agent.nodes.agent_loop_node.get_chat_model_client",
                return_value=fake_client,
            ):
                result = await agent_loop_node(gated)
        finally:
            TOOLS[TONGUE_IMAGE_ANALYSIS_TOOL] = original_image_tool
            TOOLS[TONGUE_REPORT_GENERATE_TOOL] = original_report_tool

        self.assertEqual("agent_loop_node", select_agent_gate_next(gated))
        self.assertEqual("agent_loop_node", result["current_node"])
        self.assertEqual("TONGUE_REPORT", result["next_action"]["payload"]["answer_type"])
        self.assertEqual(
            [call["tool_name"] for call in result["agent_loop"]["tool_calls"]],
            [TONGUE_IMAGE_ANALYSIS_TOOL, TONGUE_REPORT_GENERATE_TOOL],
        )
        self.assertEqual("model_tool_calling_loop", result["agent_loop"]["mode"])
        self.assertEqual(3, len(fake_client.calls))

    async def test_prefixed_fenced_final_answer_is_parsed_as_structured_content(self) -> None:
        state = await query_rewrite_node(self._state_after_tongue_report())
        gated = await agent_gate_node(state)
        final_payload = {
            "type": "final_answer",
            "content": "## Detailed report\nThis should be normalized.",
            "structured_content": {
                "schema_version": "1.0",
                "answer_type": "DETAILED_TONGUE_REPORT",
                "title": "Detailed tongue report",
                "summary": "Clean summary for display.",
                "sections": [
                    {
                        "title": "Observation",
                        "items": ["White coating"],
                    }
                ],
                "disclaimer": "General health reference only.",
            },
        }
        fake_client = FakeToolCallingClient(
            [
                {
                    "role": "assistant",
                    "content": (
                        "Tool results are enough.\n```json\n"
                        f"{json.dumps(final_payload, ensure_ascii=False)}\n```"
                    ),
                }
            ]
        )

        with patch(
            "app.agent.nodes.agent_loop_node.get_chat_model_client",
            return_value=fake_client,
        ):
            result = await agent_loop_node(gated)

        self.assertEqual("Detailed report\nThis should be normalized.", result["response_message"]["content"])
        self.assertEqual(
            "Detailed tongue report",
            result["response_message"]["structured_content"]["title"],
        )
        self.assertNotIn("```json", result["response_message"]["content"])
        self.assertNotIn("final_answer", result["response_message"]["content"])

    async def test_malformed_final_answer_falls_back_to_last_tool_result(self) -> None:
        state = await query_rewrite_node(self._state_after_tongue_report())
        gated = await agent_gate_node(state)
        fake_client = FakeToolCallingClient(
            [
                tool_call_message(
                    "report_followup_tool",
                    {"reason": "Need a more detailed report"},
                ),
                {
                    "role": "assistant",
                    "content": (
                        "Tool results are enough.\n```json\n"
                        '{"type":"final_answer","content":"## Broken raw report",'
                        '"structured_content":{"answer_type":"DETAILED_TONGUE_REPORT"'
                    ),
                },
            ]
        )

        with patch(
            "app.agent.nodes.agent_loop_node.get_chat_model_client",
            return_value=fake_client,
        ), patch(
            "app.agent.nodes.report_followup_node.generate_report_followup_reply",
            AsyncMock(
                return_value=(
                    "Clean tool report summary.",
                    None,
                    {
                        "answer_type": "DETAILED_TONGUE_REPORT",
                        "title": "Clean detailed report",
                        "summary": "Clean tool report summary.",
                        "sections": [{"title": "Observation", "items": ["White coating"]}],
                    },
                )
            ),
        ):
            result = await agent_loop_node(gated)

        self.assertEqual("Clean tool report summary.", result["response_message"]["content"])
        self.assertEqual(
            "Clean detailed report",
            result["response_message"]["structured_content"]["title"],
        )
        self.assertNotIn("```json", result["response_message"]["content"])
        self.assertNotIn("final_answer", result["response_message"]["content"])


if __name__ == "__main__":
    unittest.main()
