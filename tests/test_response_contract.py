import unittest

from app.agent.response_contract import (
    final_answer_from_text,
    looks_like_internal_json,
)


class ResponseContractTests(unittest.TestCase):
    def test_prefixed_valid_final_answer_json_is_unwrapped(self) -> None:
        raw = (
            '根据已有报告记录，我可以直接回答。\n'
            '{"type":"final_answer","content":"上次报告整体偏正常。",'
            '"structured_content":{"answer_type":"REPORT_FOLLOWUP",'
            '"title":"报告回顾","summary":"整体偏正常","sections":[]}}'
        )

        parsed = final_answer_from_text(raw)

        self.assertIsNotNone(parsed)
        content, structured = parsed or ("", None)
        self.assertEqual("上次报告整体偏正常。", content)
        self.assertEqual("REPORT_FOLLOWUP", (structured or {}).get("answer_type"))
        self.assertNotIn('"type"', content)

    def test_prefixed_malformed_final_answer_json_recovers_content(self) -> None:
        raw = '''根据已有的活动报告记录，我可以直接为你回顾。不需要再调用工具了。

{
  "type": "final_answer",
  "content": "当然记得！您上一次的舌象健康报告重点如下：

舌象基本情况：舌质淡红、舌苔薄白，重点关注特征是"白苔"。

核心建议方向：保持规律饮食、稳定作息和适量运动。",
  "structured_content": {
    "schema_version": "1.0",
    "answer_type": "REPORT_FOLLOWUP",
    "title": "上一次舌象报告回顾"
  }
}'''

        self.assertTrue(looks_like_internal_json(raw))
        parsed = final_answer_from_text(raw)

        self.assertIsNotNone(parsed)
        content, structured = parsed or ("", None)
        self.assertIn("舌质淡红、舌苔薄白", content)
        self.assertIn('重点关注特征是"白苔"', content)
        self.assertIn("规律饮食", content)
        self.assertNotIn("structured_content", content)
        self.assertNotIn('"type"', content)
        self.assertIsNone(structured)

    def test_prefixed_internal_wrapper_is_not_treated_as_plain_text(self) -> None:
        raw = (
            '我已经整理好了。\n'
            '{"type":"final_answer","content":"回答",'
            '"structured_content":{"summary":"摘要"}}'
        )

        self.assertTrue(looks_like_internal_json(raw))

    def test_normal_natural_language_is_unchanged(self) -> None:
        raw = "上一次报告显示舌质淡红、舌苔薄白，整体偏正常。"

        parsed = final_answer_from_text(raw)

        self.assertEqual((raw, None), parsed)


if __name__ == "__main__":
    unittest.main()
