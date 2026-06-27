import unittest

from app.agent.nodes.web_search_tool import _parse_results


class WebSearchToolTests(unittest.TestCase):
    def test_parse_tavily_results(self) -> None:
        payload = {
            "results": [
                {
                    "title": "肾虚概念",
                    "url": "https://example.test/kidney",
                    "content": "肾虚是中医理论中的概念，需要结合具体表现辨析。",
                },
                {
                    "url": "https://example.test/diet",
                    "raw_content": "饮食调理应结合个体情况。",
                },
            ]
        }

        results = _parse_results(payload)

        self.assertEqual("肾虚概念", results[0]["title"])
        self.assertEqual("https://example.test/kidney", results[0]["url"])
        self.assertIn("中医理论", results[0]["snippet"])
        self.assertEqual("https://example.test/diet", results[1]["title"])


if __name__ == "__main__":
    unittest.main()
