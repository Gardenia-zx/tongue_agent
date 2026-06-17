from typing import Any, Literal

from pydantic import BaseModel, Field


class Attachment(BaseModel):
    file_id: int
    file_type: Literal["image", "document", "other"]
    purpose: str | None = None


class AgentMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content_type: Literal["text", "image", "mixed", "tool_result"] = "text"
    content: str | None = None
    attachments: list[Attachment] = Field(default_factory=list)


class AgentClientContext(BaseModel):
    page: str | None = None
    active_report_id: int | None = None
    device_type: str | None = None
    locale: str = "zh-CN"
    extra: dict[str, Any] = Field(default_factory=dict)


class AgentRunRequest(BaseModel):
    schema_version: str = "1.0"
    request_id: str
    trace_id: str
    user_id: int
    thread_id: str
    conversation_id: str | None = None
    report_id: int | None = None
    task_id: int | None = None
    task_version: int | None = None
    message: AgentMessage
    client_context: AgentClientContext = Field(default_factory=AgentClientContext)
    options: dict[str, Any] = Field(default_factory=dict)


class NextAction(BaseModel):
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentRunResponse(BaseModel):
    schema_version: str = "1.0"
    request_id: str
    trace_id: str
    thread_id: str
    conversation_id: str | None = None
    report_id: int | None = None
    task_id: int | None = None
    status: Literal[
        "COMPLETED",
        "WAIT_USER_ANSWER",
        "CLARIFY",
        "REJECTED",
        "FAILED",
    ]
    intent_result: dict[str, Any] | None = None
    message: dict[str, Any] | None = None
    next_action: NextAction | None = None
    state_snapshot: dict[str, Any] = Field(default_factory=dict)

class AgentEvent(BaseModel):
    schema_version: str = "1.0"
    event_id: str
    trace_id: str
    thread_id: str
    user_id: int
    report_id: int | None = None
    task_id: int | None = None
    event_type: str
    node_name: str | None = None
    display_text: str
    progress: float | None = Field(default=None, ge=0.0, le=1.0)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str