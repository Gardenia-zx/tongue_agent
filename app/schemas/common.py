from typing import Any

from pydantic import BaseModel, Field


# 通用响应，错误结构，版本集合
class ErrorDetail(BaseModel):
    error_code: str
    error_stage: str | None = None
    retryable: bool = False
    user_action: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ApiResponse(BaseModel):
    success: bool
    code: str
    message: str
    trace_id: str
    data: dict[str, Any] | None = None
    error: ErrorDetail | None = None


class VersionBundle(BaseModel):
    schema_version: str = "1.0"

    vision_model_version: str | None = None
    label_mapping_version: str | None = None

    intent_engine_version: str | None = None
    intent_model_version: str | None = None

    embedding_model_version: str | None = None
    rerank_model_version: str | None = None

    rag_pipeline_version: str | None = None
    knowledge_version: str | None = None

    report_prompt_version: str | None = None
    safety_rule_version: str | None = None

    memory_policy_version: str | None = None
    workflow_version: str | None = None
    app_version: str | None = None