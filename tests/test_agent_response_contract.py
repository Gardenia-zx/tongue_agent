import unittest

from app.agent.response_contract import (
    final_answer_from_text,
    is_final_answer_candidate,
    looks_like_internal_json,
    normalize_structured_content,
    structured_content_to_text,
)


class AgentResponseContractTest(unittest.TestCase):
    def test_final_answer_without_type_is_allowed_when_no_internal_fields(self) -> None:
        payload = {
            "content": "白苔通常需要结合厚薄、润燥和近期饮食睡眠一起观察。",
            "structured_content": {"sections": [{"title": "建议", "items": ["规律作息"]}]},
        }

        self.assertTrue(is_final_answer_candidate(payload))

    def test_tool_payload_with_content_is_not_final_answer(self) -> None:
        payload = {
            "content": "工具结果",
            "tool_decision": {"need_rag": True},
        }

        self.assertFalse(is_final_answer_candidate(payload))

    def test_markdown_final_answer_is_parsed(self) -> None:
        content, structured = final_answer_from_text(
            '```json\n{"type":"final_answer","content":"可以继续观察舌苔变化。","structured_content":{"title":"观察建议"}}\n```'
        )

        self.assertEqual("可以继续观察舌苔变化。", content)
        self.assertEqual("观察建议", structured["title"])

    def test_truncated_internal_json_is_blocked(self) -> None:
        self.assertTrue(looks_like_internal_json('{"type":"final_answer","structured_content":{'))
        self.assertIsNone(final_answer_from_text('{"type":"final_answer","structured_content":{'))

    def test_prefixed_truncated_internal_json_is_blocked(self) -> None:
        raw = '工具结果足够。\n```json\n{"type":"final_answer","structured_content":{'
        self.assertTrue(looks_like_internal_json(raw))
        self.assertIsNone(final_answer_from_text(raw))

    def test_structured_content_falls_back_to_text(self) -> None:
        structured = normalize_structured_content(
            {
                "summary": "这是摘要。",
                "sections": [{"title": "饮食", "items": ["少生冷", {"bad": "object"}]}],
            }
        )

        text = structured_content_to_text(structured)

        self.assertIn("这是摘要。", text)
        self.assertIn("饮食：", text)
        self.assertIn("少生冷", text)
        self.assertNotIn("bad", text)


if __name__ == "__main__":
    unittest.main()
