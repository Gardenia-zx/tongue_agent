import unittest
from unittest.mock import AsyncMock, patch

from app.agent.graph import select_tongue_analysis_next
from app.agent.nodes.tongue_report_node import tongue_report_node


class TestTongueReportNode(unittest.IsolatedAsyncioTestCase):
    async def test_tongue_report_node_uses_feature_rag_query(self) -> None:
        state = {
            "thread_id": "tongue_report_thread",
            "user_id": 10001,
            "report_id": 20001,
            "task_id": 30001,
            "client_context": {
                "extra": {
                    "image_path": "D:\\tongue\\tongue_delivery\\tongue.jpg",
                }
            },
            "tongue_features": {
                "detected_feature_codes": ["coating.color.white"],
                "rag_query": "白苔 苔白 舌苔发白 舌苔颜色 舌象观察 一般健康知识",
                "coating": {
                    "color": {
                        "status": "DETECTED",
                        "items": [
                            {
                                "code": "coating.color.white",
                                "name": "白苔",
                                "confidence": 0.93,
                            }
                        ],
                    }
                },
            },
            "draft_report": {
                "schema_version": "1.0",
                "report_type": "tongue_analysis_mvp",
            },
        }

        rag_context = {
            "answer": "白苔是舌苔颜色的一种常见描述。",
            "query": state["tongue_features"]["rag_query"],
            "hits": [
                {
                    "chunk_id": "c1",
                    "doc_id": "d1",
                    "title": "白苔和黄苔的一般理解",
                    "content": "白苔和黄苔是舌苔颜色的常见描述。",
                    "source_uri": "seed://tongue",
                    "tags": ["白苔", "舌苔"],
                    "final_score": 0.91,
                    "metadata": {"source": "test"},
                }
            ],
            "retrieval_engine": "ES_BM25_TEXT2VEC_HYBRID",
            "answer_engine": "deepseek_openai_compatible",
            "grounded": True,
            "debug": {},
        }

        mocked_answer = AsyncMock(return_value=rag_context)
        with patch(
            "app.agent.nodes.tongue_report_node.answer_with_rag",
            mocked_answer,
        ):
            result = await tongue_report_node(state)

        mocked_answer.assert_awaited_once_with(state["tongue_features"]["rag_query"])
        self.assertEqual(result["current_node"], "tongue_report_node")
        self.assertEqual(result["next_action"]["type"], "RESPOND_TO_USER")
        self.assertTrue(result["next_action"]["payload"]["grounded"])
        self.assertEqual(result["next_action"]["payload"]["hit_count"], 1)
        self.assertIn("识别结果", result["response_message"]["content"])
        self.assertIn("当前图像模型主要识别到：白苔", result["response_message"]["content"])
        self.assertIn("一般解释", result["response_message"]["content"])
        self.assertIn("继续观察", result["response_message"]["content"])
        self.assertIn("健康管理建议", result["response_message"]["content"])
        self.assertIn("知识库依据", result["response_message"]["content"])
        self.assertIn("页面下方可查看来源片段", result["response_message"]["content"])
        draft_report = result["draft_report"]
        self.assertEqual(draft_report["schema_version"], "1.0")
        self.assertEqual(draft_report["report_type"], "tongue_analysis_mvp")
        self.assertEqual(draft_report["report_status"], "DRAFT")
        self.assertEqual(draft_report["report_id"], 20001)
        self.assertEqual(draft_report["user_id"], 10001)
        self.assertEqual(draft_report["thread_id"], "tongue_report_thread")
        self.assertEqual(draft_report["image_info"]["source_type"], "path")
        self.assertEqual(draft_report["image_info"]["filename"], "tongue.jpg")
        self.assertEqual(
            draft_report["standard_features"]["detected_feature_codes"],
            ["coating.color.white"],
        )
        self.assertEqual(draft_report["rag_query"], state["tongue_features"]["rag_query"])
        self.assertTrue(draft_report["rag_grounded"])
        self.assertEqual(len(draft_report["rag_evidence"]), 1)
        self.assertEqual(draft_report["rag_evidence"][0]["chunk_id"], "c1")
        self.assertIn("白苔", draft_report["feature_summary"])
        self.assertIn("白苔", draft_report["health_notes"][0])
        self.assertIn("structured_sections", draft_report["metadata"])
        self.assertIn(
            "白苔是常见舌苔表现",
            draft_report["metadata"]["structured_sections"]["general_interpretation"],
        )
        self.assertIn("draft_report", result["next_action"]["payload"])


class TestTongueGraphRouting(unittest.TestCase):
    def test_tongue_features_ready_routes_to_report_node(self) -> None:
        state = {"next_action": {"type": "TONGUE_FEATURES_READY"}}
        self.assertEqual(select_tongue_analysis_next(state), "tongue_report_node")

    def test_tongue_model_wait_or_failed_routes_to_memory_commit(self) -> None:
        for action_type in ["START_OR_RESUME_TONGUE_ANALYSIS", "TONGUE_MODEL_FAILED"]:
            state = {"next_action": {"type": action_type}}
            self.assertEqual(select_tongue_analysis_next(state), "memory_commit_node")


if __name__ == "__main__":
    unittest.main()
