import unittest
from unittest.mock import patch

from app.agent.nodes.tongue_analysis_node import tongue_analysis_node
from app.tongue.feature_mapping import normalize_tongue_model_result


class TestTongueFeatureMapping(unittest.TestCase):
    def test_normalize_model_features_to_agent_standard_features(self) -> None:
        model_result = {
            "features": [
                {
                    "class_id": 8,
                    "feature": "齿痕舌",
                    "confidence": 0.88,
                    "bbox_xyxy": [1, 2, 3, 4],
                },
                {
                    "class_id": 9,
                    "feature": "白苔",
                    "confidence": 0.93,
                    "bbox_xyxy": [5, 6, 7, 8],
                },
                {
                    "class_id": 12,
                    "feature": "滑苔",
                    "confidence": 0.81,
                    "bbox_xyxy": [9, 10, 11, 12],
                },
            ]
        }

        standard = normalize_tongue_model_result(model_result)

        self.assertIn(
            "tongue_body.texture.tooth_marked",
            standard["detected_feature_codes"],
        )
        self.assertIn("coating.color.white", standard["detected_feature_codes"])
        self.assertIn("coating.moisture.slippery", standard["detected_feature_codes"])
        self.assertEqual(
            standard["tongue_body"]["texture"]["items"][0]["name"],
            "齿痕舌",
        )
        self.assertEqual(standard["coating"]["color"]["items"][0]["name"], "白苔")
        self.assertIn("齿痕舌", standard["rag_query"])
        self.assertIn("白苔", standard["rag_query"])
        self.assertNotIn("UNSUPPORTED_BY_MODEL", standard["rag_query"])

    def test_duplicate_class_keeps_highest_confidence(self) -> None:
        standard = normalize_tongue_model_result(
            {
                "features": [
                    {
                        "class_id": 9,
                        "feature": "白苔",
                        "confidence": 0.55,
                        "bbox_xyxy": [],
                    },
                    {
                        "class_id": 9,
                        "feature": "白苔",
                        "confidence": 0.91,
                        "bbox_xyxy": [],
                    },
                ]
            }
        )

        item = standard["coating"]["color"]["items"][0]
        self.assertEqual(item["confidence"], 0.91)
        self.assertEqual(len(standard["coating"]["color"]["items"]), 1)


class TestTongueAnalysisNode(unittest.IsolatedAsyncioTestCase):
    async def test_node_builds_standard_features_without_negative_wording(self) -> None:
        state = {
            "thread_id": "tongue_feature_thread",
            "task_id": 1,
            "report_id": 2,
            "client_context": {
                "extra": {
                    "tongue_model_result": {
                        "features": [
                            {
                                "class_id": 9,
                                "feature": "白苔",
                                "confidence": 0.9368,
                                "bbox_xyxy": [142.81, 403.45, 1575.72, 2048.74],
                            }
                        ]
                    }
                }
            },
        }

        result = await tongue_analysis_node(state)
        content = result["response_message"]["content"]

        self.assertEqual(result["next_action"]["type"], "TONGUE_FEATURES_READY")
        self.assertIn("coating.color.white", result["tongue_features"]["detected_feature_codes"])
        self.assertIn("白苔", result["next_action"]["payload"]["rag_query"])
        self.assertNotIn("得不出结论", content)
        self.assertNotIn("无法判断", content)
        self.assertNotIn("不做结论", content)

    async def test_node_calls_tongue_model_client_when_image_path_exists(self) -> None:
        class FakeTongueModelClient:
            async def predict_image_path(self, image_path: str) -> dict:
                self.image_path = image_path
                return {
                    "features": [
                        {
                            "class_id": 8,
                            "feature": "齿痕舌",
                            "confidence": 0.88,
                            "bbox_xyxy": [1, 2, 3, 4],
                        }
                    ]
                }

        fake_client = FakeTongueModelClient()
        state = {
            "thread_id": "tongue_model_thread",
            "client_context": {
                "extra": {
                    "image_path": "D:\\tongue\\tongue_delivery\\tongue.jpg",
                }
            },
        }

        with patch(
            "app.agent.nodes.tongue_analysis_node.get_tongue_model_client",
            return_value=fake_client,
        ):
            result = await tongue_analysis_node(state)

        self.assertEqual(fake_client.image_path, "D:\\tongue\\tongue_delivery\\tongue.jpg")
        self.assertEqual(result["next_action"]["type"], "TONGUE_FEATURES_READY")
        self.assertIn(
            "tongue_body.texture.tooth_marked",
            result["tongue_features"]["detected_feature_codes"],
        )


if __name__ == "__main__":
    unittest.main()
