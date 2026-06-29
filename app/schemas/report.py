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


class ReportEvidenceRef(BaseModel):
    doc_id: str | None = None
    chunk_id: str | None = None
    title: str | None = None
    final_score: float | None = None


class ReportRecognitionEvidence(BaseModel):
    code: str
    name: str
    confidence: float | None = None
    status: Literal["DETECTED"] = "DETECTED"


class ReportRecognitionLimit(BaseModel):
    dimension: str
    status: Literal["NOT_EVALUATED", "UNSUPPORTED_BY_MODEL"]
    reason: str = ""


class ReportConditionalAnalysis(BaseModel):
    condition: str
    interpretation: str


class ReportPlan(BaseModel):
    goal: str = ""
    actions: list[str] = Field(default_factory=list)
    frequency: str = "每天"
    duration: str = "连续3天"
    observation_metrics: list[str] = Field(default_factory=list)


class TongueAnalysisReport(BaseModel):
    schema_version: str = "2.0"
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
    evidence_refs: list[ReportEvidenceRef] = Field(default_factory=list)
    comprehensive_summary: str = Field(default="", description="面向用户的综合摘要，不能替代诊断。")
    recognition_evidence: list[ReportRecognitionEvidence] = Field(default_factory=list)
    recognition_limits: list[ReportRecognitionLimit] = Field(default_factory=list)
    dimension_values: list[dict[str, Any]] = Field(default_factory=list)
    conditional_analysis: list[ReportConditionalAnalysis] = Field(default_factory=list)
    tongue_feature_explanation: str = ""
    diet_plan: ReportPlan = Field(default_factory=ReportPlan)
    sleep_plan: ReportPlan = Field(default_factory=ReportPlan)
    exercise_plan: ReportPlan = Field(default_factory=ReportPlan)
    three_day_observation: list[str] = Field(default_factory=list)
    followup_questions: list[str] = Field(default_factory=list)
    tongue_features: list[dict[str, Any]] = Field(default_factory=list)
    health_interpretation: str = Field(default="", description="舌象特征的健康管理解释，不能诊断。")
    dietary_advice: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="饮食建议，只能包含饮食、饮水、进食习惯相关内容。",
    )
    exercise_advice: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="运动建议，只能包含运动方式、强度、频率和恢复观察相关内容。",
    )
    lifestyle_advice: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="生活方式建议，只能包含作息、口腔清洁、拍摄复查习惯等内容，不能写舌苔观察问题。",
    )
    risk_tips: list[str] = Field(default_factory=list, max_length=2)
    summary: str
    health_notes: list[str] = Field(default_factory=list)
    risk_disclaimer: str = (
        "以上内容用于一般健康知识说明和健康管理参考，不能替代医生诊断。"
    )
    versions: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
