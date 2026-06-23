from dataclasses import dataclass
from typing import Any

from app.schemas.tongue import (
    TongueFeatureItem,
    TongueFeatureSlot,
    TongueRawDetection,
    TongueStandardFeatures,
)


@dataclass(frozen=True)
class FeatureMapping:
    class_id: int
    code: str
    name: str
    group: str
    slot: str
    rag_terms: tuple[str, ...]


FEATURE_MAPPINGS: dict[int, FeatureMapping] = {
    0: FeatureMapping(
        0,
        "overall.healthy_tongue",
        "健康舌",
        "overall",
        "overall",
        ("健康舌", "正常舌象", "舌象观察一般原则"),
    ),
    1: FeatureMapping(
        1,
        "coating.distribution.peeled",
        "剥苔舌",
        "coating",
        "distribution",
        ("剥苔", "地图舌", "舌苔剥落", "舌苔分布"),
    ),
    2: FeatureMapping(
        2,
        "tongue_body.color.red",
        "红舌",
        "tongue_body",
        "color",
        ("红舌", "舌色偏红", "舌质红", "舌象一般说明"),
    ),
    3: FeatureMapping(
        3,
        "tongue_body.color.purple",
        "紫舌",
        "tongue_body",
        "color",
        ("紫舌", "舌色偏紫", "舌质紫", "舌象一般说明"),
    ),
    4: FeatureMapping(
        4,
        "tongue_body.shape.enlarged",
        "胖大舌",
        "tongue_body",
        "shape",
        ("胖大舌", "舌体胖大", "舌胖", "舌形"),
    ),
    5: FeatureMapping(
        5,
        "tongue_body.shape.thin",
        "瘦舌",
        "tongue_body",
        "shape",
        ("瘦舌", "舌体偏瘦", "舌形"),
    ),
    6: FeatureMapping(
        6,
        "tongue_body.spots.red_spot",
        "红点舌",
        "tongue_body",
        "spots",
        ("红点舌", "舌面红点", "舌尖红点", "舌象一般说明"),
    ),
    7: FeatureMapping(
        7,
        "tongue_body.texture.fissured",
        "裂纹舌",
        "tongue_body",
        "texture",
        ("裂纹舌", "舌面裂纹", "舌纹", "舌象一般说明"),
    ),
    8: FeatureMapping(
        8,
        "tongue_body.texture.tooth_marked",
        "齿痕舌",
        "tongue_body",
        "texture",
        ("齿痕舌", "舌边齿痕", "舌头有齿痕", "舌体齿痕"),
    ),
    9: FeatureMapping(
        9,
        "coating.color.white",
        "白苔",
        "coating",
        "color",
        ("白苔", "苔白", "舌苔发白", "舌苔颜色"),
    ),
    10: FeatureMapping(
        10,
        "coating.color.yellow",
        "黄苔",
        "coating",
        "color",
        ("黄苔", "苔黄", "舌苔发黄", "舌苔颜色"),
    ),
    11: FeatureMapping(
        11,
        "coating.color.black",
        "黑苔",
        "coating",
        "color",
        ("黑苔", "苔黑", "舌苔发黑", "舌苔颜色"),
    ),
    12: FeatureMapping(
        12,
        "coating.moisture.slippery",
        "滑苔",
        "coating",
        "moisture",
        ("滑苔", "舌苔滑", "舌苔湿润", "舌苔润燥"),
    ),
    13: FeatureMapping(
        13,
        "regions.kidney.depression",
        "肾区凹陷",
        "regions",
        "kidney",
        ("肾区凹陷", "舌面区域特征", "舌象分区"),
    ),
    14: FeatureMapping(
        14,
        "regions.kidney.bulge",
        "肾区凸起",
        "regions",
        "kidney",
        ("肾区凸起", "舌面区域特征", "舌象分区"),
    ),
    15: FeatureMapping(
        15,
        "regions.liver_gallbladder.depression",
        "肝胆区凹陷",
        "regions",
        "liver_gallbladder",
        ("肝胆区凹陷", "舌面区域特征", "舌象分区"),
    ),
    16: FeatureMapping(
        16,
        "regions.liver_gallbladder.bulge",
        "肝胆区凸起",
        "regions",
        "liver_gallbladder",
        ("肝胆区凸起", "舌面区域特征", "舌象分区"),
    ),
    17: FeatureMapping(
        17,
        "regions.spleen_stomach.depression",
        "脾胃区凹陷",
        "regions",
        "spleen_stomach",
        ("脾胃区凹陷", "舌面区域特征", "脾胃区", "舌象分区"),
    ),
    18: FeatureMapping(
        18,
        "regions.heart_lung.depression",
        "心肺区凹陷",
        "regions",
        "heart_lung",
        ("心肺区凹陷", "舌面区域特征", "心肺区", "舌象分区"),
    ),
    19: FeatureMapping(
        19,
        "regions.heart_lung.bulge",
        "心肺区凸起",
        "regions",
        "heart_lung",
        ("心肺区凸起", "舌面区域特征", "心肺区", "舌象分区"),
    ),
}

UNSUPPORTED_FEATURE_CODES = [
    "coating.thickness.thick",
    "coating.thickness.thin",
    "coating.greasy.greasy",
    "coating.moisture.dry",
    "tongue_body.color.pale",
    "tongue_body.color.dark_red",
    "tongue_body.color.crimson",
    "sublingual_vein.color",
    "sublingual_vein.shape",
]


def normalize_tongue_model_result(model_result: dict[str, Any]) -> dict[str, Any]:
    raw_features = _extract_raw_features(model_result)
    return normalize_raw_detections(raw_features).model_dump(mode="json")


def normalize_raw_detections(raw_features: list[dict[str, Any]]) -> TongueStandardFeatures:
    standard = TongueStandardFeatures(
        supported_feature_codes=[mapping.code for mapping in FEATURE_MAPPINGS.values()],
        unsupported_feature_codes=UNSUPPORTED_FEATURE_CODES,
    )
    terms: list[str] = []
    detected_codes: list[str] = []
    best_by_code: dict[str, TongueFeatureItem] = {}

    for raw in raw_features:
        detection = _to_raw_detection(raw)
        standard.raw_detections.append(detection)
        mapping = FEATURE_MAPPINGS.get(detection.class_id)
        if mapping is None:
            continue

        item = TongueFeatureItem(
            code=mapping.code,
            name=mapping.name,
            confidence=detection.confidence,
            evidence=[detection],
            rag_terms=list(mapping.rag_terms),
        )

        current = best_by_code.get(mapping.code)
        if current is None or (item.confidence or 0.0) > (current.confidence or 0.0):
            best_by_code[mapping.code] = item

    for item in best_by_code.values():
        mapping = _mapping_by_code(item.code)
        if mapping is None:
            continue

        _append_feature_item(standard, mapping, item)
        detected_codes.append(item.code)
        for term in item.rag_terms:
            if term not in terms:
                terms.append(term)

    standard.detected_feature_codes = sorted(detected_codes)
    standard.rag_terms = terms
    standard.rag_query = build_tongue_feature_rag_query(terms)
    return standard


def build_tongue_feature_rag_query(rag_terms: list[str]) -> str:
    unique_terms = []
    for term in rag_terms:
        term = term.strip()
        if term and term not in unique_terms:
            unique_terms.append(term)

    if not unique_terms:
        return ""

    return " ".join([*unique_terms, "舌象观察", "一般健康知识"])


def _extract_raw_features(model_result: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(model_result, dict):
        return []

    if isinstance(model_result.get("raw_detections"), list):
        return model_result["raw_detections"]

    if isinstance(model_result.get("features"), list):
        return model_result["features"]

    data = model_result.get("data")
    if isinstance(data, dict):
        if isinstance(data.get("raw_detections"), list):
            return data["raw_detections"]
        if isinstance(data.get("features"), list):
            return data["features"]

    return []


def _to_raw_detection(raw: dict[str, Any]) -> TongueRawDetection:
    return TongueRawDetection(
        class_id=int(raw.get("class_id")),
        feature=str(raw.get("feature") or raw.get("name") or ""),
        confidence=float(raw.get("confidence") or 0.0),
        bbox_xyxy=[float(value) for value in raw.get("bbox_xyxy") or []],
        metadata={
            key: value
            for key, value in raw.items()
            if key not in {"class_id", "feature", "name", "confidence", "bbox_xyxy"}
        },
    )


def _mapping_by_code(code: str) -> FeatureMapping | None:
    for mapping in FEATURE_MAPPINGS.values():
        if mapping.code == code:
            return mapping
    return None


def _append_feature_item(
    standard: TongueStandardFeatures,
    mapping: FeatureMapping,
    item: TongueFeatureItem,
) -> None:
    if mapping.group == "overall":
        _append_to_slot(standard.overall, item)
        return

    group = getattr(standard, mapping.group)
    slot = getattr(group, mapping.slot)
    _append_to_slot(slot, item)


def _append_to_slot(slot: TongueFeatureSlot, item: TongueFeatureItem) -> None:
    slot.status = "DETECTED"
    slot.items.append(item)
