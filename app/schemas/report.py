from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.tongue import TongueStandardFeatures


class ReportImageInfo(BaseModel):
    image_id: str | None = None
    file_id: int | None = None
    source_type: Literal["path", "url", "upload", "unknown"] = "unknown"
    source_uri: str | None = None
    filename: str | None = None
    content_type: str | None = None


class ReportRagEvidence(BaseModel):
    chunk_id: str
    doc_id: str | None = None
    title: str | None = None
    content: str
    source_uri: str | None = None
    tags: list[str] = Field(default_factory=list)
    final_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class TongueAnalysisReport(BaseModel):
    schema_version: str = "1.0"
    report_type: Literal["tongue_analysis_mvp"] = "tongue_analysis_mvp"
    report_status: Literal["DRAFT", "FINAL"] = "DRAFT"
    report_id: int | None = None
    user_id: int | None = None
    thread_id: str
    task_id: int | None = None
    task_version: int | None = None
    image_info: ReportImageInfo = Field(default_factory=ReportImageInfo)
    standard_features: TongueStandardFeatures
    feature_summary: str
    rag_query: str = ""
    rag_grounded: bool = False
    rag_evidence: list[ReportRagEvidence] = Field(default_factory=list)
    summary: str
    health_notes: list[str] = Field(default_factory=list)
    risk_disclaimer: str = (
        "以上内容用于一般健康知识说明和健康管理参考，不能替代医生诊断。"
    )
    versions: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
