from typing import Any, Literal

from pydantic import BaseModel, Field


FeatureStatus = Literal[
    "DETECTED",
    "NOT_DETECTED",
    "UNSUPPORTED_BY_MODEL",
    "NOT_EVALUATED",
]


class TongueRawDetection(BaseModel):
    class_id: int
    feature: str
    confidence: float
    bbox_xyxy: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TongueFeatureItem(BaseModel):
    code: str
    name: str
    status: FeatureStatus = "DETECTED"
    confidence: float | None = None
    evidence: list[TongueRawDetection] = Field(default_factory=list)
    rag_terms: list[str] = Field(default_factory=list)


class TongueFeatureSlot(BaseModel):
    status: FeatureStatus = "NOT_EVALUATED"
    items: list[TongueFeatureItem] = Field(default_factory=list)


class TongueBodyFeatureGroup(BaseModel):
    color: TongueFeatureSlot = Field(default_factory=TongueFeatureSlot)
    shape: TongueFeatureSlot = Field(default_factory=TongueFeatureSlot)
    texture: TongueFeatureSlot = Field(default_factory=TongueFeatureSlot)
    spots: TongueFeatureSlot = Field(default_factory=TongueFeatureSlot)


class TongueCoatingFeatureGroup(BaseModel):
    color: TongueFeatureSlot = Field(default_factory=TongueFeatureSlot)
    quality: TongueFeatureSlot = Field(default_factory=TongueFeatureSlot)
    distribution: TongueFeatureSlot = Field(default_factory=TongueFeatureSlot)
    thickness: TongueFeatureSlot = Field(default_factory=TongueFeatureSlot)
    moisture: TongueFeatureSlot = Field(default_factory=TongueFeatureSlot)
    greasy: TongueFeatureSlot = Field(default_factory=TongueFeatureSlot)


class TongueRegionFeatureGroup(BaseModel):
    kidney: TongueFeatureSlot = Field(default_factory=TongueFeatureSlot)
    liver_gallbladder: TongueFeatureSlot = Field(default_factory=TongueFeatureSlot)
    spleen_stomach: TongueFeatureSlot = Field(default_factory=TongueFeatureSlot)
    heart_lung: TongueFeatureSlot = Field(default_factory=TongueFeatureSlot)


class TongueStandardFeatures(BaseModel):
    schema_version: str = "1.0"
    source: str = "tongue_model"
    overall: TongueFeatureSlot = Field(default_factory=TongueFeatureSlot)
    tongue_body: TongueBodyFeatureGroup = Field(default_factory=TongueBodyFeatureGroup)
    coating: TongueCoatingFeatureGroup = Field(default_factory=TongueCoatingFeatureGroup)
    regions: TongueRegionFeatureGroup = Field(default_factory=TongueRegionFeatureGroup)
    raw_detections: list[TongueRawDetection] = Field(default_factory=list)
    detected_feature_codes: list[str] = Field(default_factory=list)
    supported_feature_codes: list[str] = Field(default_factory=list)
    unsupported_feature_codes: list[str] = Field(default_factory=list)
    rag_query: str = ""
    rag_terms: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
