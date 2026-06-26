import unittest

from app.core.turn_hash import canonical_agent_request_hash
from app.integrations.turn_records import SnapshotCodec
from app.schemas.agent import AgentRunRequest


def _request(*, trace_id: str = "trace_a", request_id: str = "req_a", content: str = "你好") -> AgentRunRequest:
    return AgentRunRequest(
        request_id=request_id,
        trace_id=trace_id,
        tenant_id="tenant_1",
        user_id=1,
        thread_id="thread_1",
        thread_epoch=1,
        turn_id="conversation_1:user_1:assistant_1",
        user_message_id="user_1",
        assistant_message_id="assistant_1",
        conversation_id="conversation_1",
        message={
            "message_id": "user_1",
            "role": "user",
            "content_type": "text",
            "content": content,
            "attachments": [],
        },
        options={
            "memory": {
                "can_read": True,
                "can_write": False,
            },
            "trusted_claims": {
                "role": "USER",
            },
        },
    )


class TestTurnIdempotencyCore(unittest.TestCase):
    def test_canonical_hash_ignores_dynamic_fields(self) -> None:
        first = _request(trace_id="trace_a", request_id="req_a")
        second = _request(trace_id="trace_b", request_id="req_b")

        self.assertEqual(
            canonical_agent_request_hash(
                first,
                tenant_id="tenant_1",
                turn_id="conversation_1:user_1:assistant_1",
            ),
            canonical_agent_request_hash(
                second,
                tenant_id="tenant_1",
                turn_id="conversation_1:user_1:assistant_1",
            ),
        )

    def test_canonical_hash_changes_for_business_content(self) -> None:
        first = _request(content="你好")
        second = _request(content="我想追问上一轮")

        self.assertNotEqual(
            canonical_agent_request_hash(
                first,
                tenant_id="tenant_1",
                turn_id="conversation_1:user_1:assistant_1",
            ),
            canonical_agent_request_hash(
                second,
                tenant_id="tenant_1",
                turn_id="conversation_1:user_1:assistant_1",
            ),
        )

    def test_snapshot_codec_encrypts_and_round_trips(self) -> None:
        codec = SnapshotCodec("unit-test-secret")
        payload = {
            "status": "COMPLETED",
            "message": {
                "content": "这是一段健康相关响应。",
            },
        }

        encrypted = codec.encrypt(payload)

        self.assertNotIn("健康相关响应", encrypted)
        self.assertEqual(payload, codec.decrypt(encrypted))


if __name__ == "__main__":
    unittest.main()
