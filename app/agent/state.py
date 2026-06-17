from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    schema_version: str

    trace_id: str
    request_id: str
    user_id: int
    thread_id: str

    conversation_id: str
    report_id: int
    task_id: int
    task_version: int

    message: dict[str, Any]
    client_context: dict[str, Any]
    options: dict[str, Any]

    current_node: str

    intent_result: dict[str, Any]
    safety_result: dict[str, Any]

    image_quality: dict[str, Any]
    tongue_features: dict[str, Any]

    rag_context: dict[str, Any]
    questions: list[dict[str, Any]]
    answers: list[dict[str, Any]]

    memory_context: dict[str, Any]

    draft_report: dict[str, Any]
    final_report: dict[str, Any]

    next_action: dict[str, Any]
    response_message: dict[str, Any]

    versions: dict[str, Any]

    retry: dict[str, Any]
    errors: list[dict[str, Any]]

    extensions: dict[str, Any]