import unittest
from unittest.mock import AsyncMock, patch

from app.agent.nodes.rag_node_utils import _with_web_search_fallback


class RagWebFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_rag_no_hits_uses_web_search_fallback(self) -> None:
        rag_result = {
            "answer": "本地知识库没有命中。",
            "query": "肾虚是什么意思",
            "hits": [],
            "grounded": False,
            "debug": {"reason": "no_retrieval_hits"},
        }
        web_result = {
            "query": "肾虚是什么意思",
            "grounded": True,
            "results": [
                {
                    "title": "肾虚概念",
                    "snippet": "肾虚是中医理论中的概念，需要结合具体表现辨析。",
                    "url": "https://example.test/kidney",
                }
            ],
            "debug": {"source": "web_search"},
        }

        with patch("app.agent.nodes.rag_node_utils.search_web", AsyncMock(return_value=web_result)):
            result = await _with_web_search_fallback(rag_result, query="肾虚是什么意思")

        self.assertTrue(result["grounded"])
        self.assertEqual("WEB_SEARCH_FALLBACK", result["retrieval_engine"])
        self.assertIn("公开资料", result["answer"])
        self.assertEqual("web_search", result["hits"][0]["metadata"]["source"])


if __name__ == "__main__":
    unittest.main()
