import unittest
import os

import httpx

from app.core.config import get_settings


class TestDeepSeekOpenAICompatibleAPI(unittest.IsolatedAsyncioTestCase):
    async def test_chat_completions_contract(self) -> None:
        if os.getenv("RUN_LLM_INTEGRATION_TESTS") != "1":
            self.skipTest(
                "Set RUN_LLM_INTEGRATION_TESTS=1 to run the real DeepSeek integration test."
            )

        settings = get_settings()
        api_key = settings.model_gateway_api_key

        if not api_key or api_key == "replace-with-your-local-api-key":
            self.skipTest(
                "DeepSeek API key is not configured. Create a local .env file and set "
                "MODEL_GATEWAY_API_KEY before running this integration test."
            )

        base_url = settings.model_gateway_base_url.rstrip("/")
        payload = {
            "model": settings.chat_model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个接口连通性测试助手，只需要简短回答。",
                },
                {
                    "role": "user",
                    "content": "请只回复：DeepSeek OpenAI compatible API connected",
                },
            ],
            "temperature": 0,
            "max_tokens": 64,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        async with httpx.AsyncClient(
            timeout=settings.model_gateway_timeout_seconds,
        ) as client:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
            )

        self.assertEqual(
            200,
            response.status_code,
            msg=f"DeepSeek request failed: {response.status_code} {response.text}",
        )

        data = response.json()
        self.assertIsInstance(data, dict)
        self.assertIn("choices", data)
        self.assertIsInstance(data["choices"], list)
        self.assertGreater(len(data["choices"]), 0)

        first_choice = data["choices"][0]
        self.assertIsInstance(first_choice, dict)
        self.assertIn("message", first_choice)

        message = first_choice["message"]
        self.assertIsInstance(message, dict)
        self.assertEqual("assistant", message.get("role"))
        self.assertIsInstance(message.get("content"), str)
        self.assertTrue(message["content"].strip())
