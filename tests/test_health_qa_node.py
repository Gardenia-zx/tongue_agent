import unittest
from unittest.mock import AsyncMock, patch

from app.agent.nodes.health_qa_node import health_qa_node


class HealthQANodeTests(unittest.IsolatedAsyncioTestCase):
    def _state(self, text: str) -> dict:
        return {
            "turn_id": "turn-health",
            "request_id": "request-health",
            "user_id": 7,
            "thread_id": "thread-health",
            "message": {"role": "user", "content": text},
            "current_turn": {
                "turn_id": "turn-health",
                "query_context": {
                    "standalone_query": text,
                    "reference_resolution": {"target_type": "GENERAL_TOPIC"},
                },
            },
        }

    def _structured_texts(self, structured: dict) -> list[str]:
        values = [
            str(structured.get("summary") or ""),
            str(structured.get("title") or ""),
            str(structured.get("disclaimer") or ""),
        ]
        for section in structured.get("sections") or []:
            values.append(str(section.get("title") or ""))
            values.append(str(section.get("content") or ""))
            values.extend(str(item) for item in section.get("items") or [])
        return values

    async def test_markdown_rag_answer_is_returned_as_structured_plain_text(self) -> None:
        rag_answer = """
根据知识库片段，肾虚需要区分是**阴虚**还是**阳虚**。

**肾虚的主要分类：**

1. **肾阳虚**：具体可以表现为：
* 肾气虚弱
* 肾阳不振
2. **肾阴虚**：主要指肾阴亏虚。

**调理注意事项：**
* **阴阳互损**：调理需要统筹兼顾。
* **影响其他脏腑**：需要结合表现观察。

以上内容仅供健康管理参考，不能替代医生诊断。
""".strip()

        with patch(
            "app.agent.nodes.health_qa_node.answer_with_rag",
            AsyncMock(return_value={"answer": rag_answer, "hits": [{"chunk_id": "1"}], "grounded": True}),
        ):
            result = await health_qa_node(self._state("肾虚是什么意思"))

        response = result["response_message"]
        structured = response["structured_content"]
        self.assertNotIn("**", response["content"])
        self.assertNotIn("*", response["content"])
        self.assertTrue(structured["sections"])
        for text in self._structured_texts(structured):
            self.assertNotIn("**", text)
            self.assertNotIn("#", text)
            self.assertFalse(text.strip().startswith("*"))

    async def test_plain_rag_answer_still_gets_a_stable_section(self) -> None:
        rag_answer = "肾虚是中医概念，通常需要结合睡眠、怕冷、腰膝状态等表现综合理解。"

        with patch(
            "app.agent.nodes.health_qa_node.answer_with_rag",
            AsyncMock(return_value={"answer": rag_answer, "hits": [], "grounded": False}),
        ):
            result = await health_qa_node(self._state("肾虚是什么意思"))

        structured = result["response_message"]["structured_content"]
        self.assertEqual(rag_answer, result["response_message"]["content"])
        self.assertEqual("参考说明", structured["sections"][0]["title"])
        self.assertEqual(rag_answer, structured["sections"][0]["content"])


if __name__ == "__main__":
    unittest.main()
