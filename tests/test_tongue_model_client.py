import tempfile
import unittest
from pathlib import Path

import httpx

from app.integrations.tongue_model_client import TongueModelClient, TongueModelError


class TestTongueModelClient(unittest.IsolatedAsyncioTestCase):
    async def test_predict_image_path_uploads_file_and_returns_data(self) -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            body = await request.aread()
            self.assertIn(b'name="file"', body)
            self.assertIn(b"image/jpeg", body)
            self.assertEqual(request.headers.get("x-api-key"), "secret")
            self.assertEqual(request.headers.get("authorization"), "Bearer hf_secret")
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "message": "success",
                    "data": {
                        "image": "tongue.jpg",
                        "features": [
                            {
                                "class_id": 9,
                                "feature": "白苔",
                                "confidence": 0.93,
                                "bbox_xyxy": [1, 2, 3, 4],
                            }
                        ],
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        client = TongueModelClient(
            base_url="https://example.test",
            api_key="secret",
            bearer_token="hf_secret",
            timeout_seconds=5,
            max_concurrency=1,
            max_image_size_mb=1,
            transport=transport,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "tongue.jpg"
            image_path.write_bytes(b"fake-jpeg-bytes")
            result = await client.predict_image_path(image_path)

        self.assertEqual(len(requests), 1)
        self.assertEqual(str(requests[0].url), "https://example.test/v1/tongue/predict")
        self.assertEqual(result["features"][0]["feature"], "白苔")

    async def test_predict_image_path_rejects_missing_file(self) -> None:
        client = TongueModelClient(
            base_url="https://example.test",
            api_key=None,
            bearer_token=None,
            timeout_seconds=5,
            max_concurrency=1,
            max_image_size_mb=1,
        )

        with self.assertRaises(TongueModelError):
            await client.predict_image_path("missing.jpg")


if __name__ == "__main__":
    unittest.main()
