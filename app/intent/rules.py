import re
from dataclasses import dataclass, field

from app.intent.domain_terms import DomainNormalizationResult
from app.schemas.intent import IntentDecision, RiskLevel


HIGH_RISK_ROUTE = "high_risk_safety_subgraph"
GENERAL_CHAT_ROUTE = "general_chat_subgraph"
TONGUE_ANALYSIS_ROUTE = "tongue_analysis_subgraph"
HEALTH_QA_ROUTE = "health_qa_subgraph"
REPORT_EXPLANATION_ROUTE = "report_explanation_subgraph"
TREND_ANALYSIS_ROUTE = "trend_analysis_subgraph"
PRIVACY_ROUTE = "privacy_request_subgraph"


@dataclass(frozen=True)
class QueryNormalization:
    original_text: str
    normalized_text: str
    replacements: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class RuleIntentMatch:
    intent_code: str
    route_target: str
    risk_level: RiskLevel
    confidence: float
    matched_rule: str
    matched_texts: list[str] = field(default_factory=list)
    decision: IntentDecision = IntentDecision.ROUTE
    safety_flags: list[str] = field(default_factory=list)


TEXT_REPLACEMENTS = {
    "舌像": "舌象",
    "舌相": "舌象",
    "舌照": "舌象照片",
    "胸疼": "胸痛",
    "胸口疼": "胸痛",
    "胸口痛": "胸痛",
    "心口疼": "胸痛",
    "心口痛": "胸痛",
    "喘不过气": "呼吸困难",
    "喘不上气": "呼吸困难",
    "呼吸不上来": "呼吸困难",
}

EMERGENCY_PATTERNS = [
    r"胸.*?(痛|疼|闷|压迫|憋)",
    r"心口.*?(痛|疼|闷|压迫|憋)",
    r"呼吸困难",
    r"喘.*?气",
    r"昏迷|晕倒|意识不清|休克",
    r"大出血|大量出血|血止不住",
    r"严重过敏|喉咙肿|过敏.*?喘",
    r"抽搐|惊厥",
]

PRESCRIPTION_PATTERNS = [
    r"吃什么药",
    r"用什么药",
    r"开什么药",
    r"开.*?方",
    r"处方",
    r"药量|剂量",
    r"停药|换药|加药|减药",
    r"治疗方案",
]

GENERAL_CHAT_PATTERNS = [
    r"你好",
    r"你能做什么",
    r"你可以.*?什么",
    r"系统.*?怎么用",
    r"功能",
    r"帮助",
]

TONGUE_ANALYSIS_PATTERNS = [
    r"(舌象|舌头|舌苔).*?(分析|看看|看一下|看下|上传|照片|图片|拍|测|报告)",
    r"(分析|看看|看一下|看下|上传|拍|测).*?(舌象|舌头|舌苔|舌象照片)",
    r"舌象健康报告",
]

TONGUE_CONTINUE_PATTERNS = [
    r"继续.*?(分析|报告|追问)",
    r"(刚才|上次|前面).*?(分析|问题|追问|报告)",
    r"(回答|补充).*?(刚才|上次|症状|问题|追问)",
    r"我来回答",
]

HEALTH_QA_PATTERNS = [
    r"(是什么|什么意思|代表什么|说明什么|为什么|区别|有关系吗|怎么调理|怎么办)",
]

REPORT_PATTERNS = [
    r"报告.*?(解释|什么意思|怎么看|建议|证据|来源|为什么)",
    r"(解释|看看|看一下).*?报告",
]

TREND_PATTERNS = [
    r"趋势|历史趋势|最近几次|这几次|和上次相比|相比|变化",
]

PRIVACY_PATTERNS = [
    r"删除|删掉|清空|撤回.*?授权|不要保存|隐私|注销",
]


def normalize_query_text(text: str) -> QueryNormalization:
    original = text.strip()
    normalized = original
    replacements: list[dict[str, str]] = []

    for source, target in TEXT_REPLACEMENTS.items():
        if source in normalized:
            normalized = normalized.replace(source, target)
            replacements.append({"from": source, "to": target})

    return QueryNormalization(
        original_text=original,
        normalized_text=normalized.strip(),
        replacements=replacements,
    )


def _first_pattern_match(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        if re.search(pattern, text):
            return pattern
    return None


def match_safety_intent(text: str) -> RuleIntentMatch | None:
    emergency_pattern = _first_pattern_match(text, EMERGENCY_PATTERNS)
    if emergency_pattern:
        return RuleIntentMatch(
            intent_code="EMERGENCY_MEDICAL",
            route_target=HIGH_RISK_ROUTE,
            risk_level=RiskLevel.EMERGENCY,
            confidence=1.0,
            matched_rule="hard_rule_emergency",
            matched_texts=[emergency_pattern],
            safety_flags=["EMERGENCY_INTENT"],
        )

    prescription_pattern = _first_pattern_match(text, PRESCRIPTION_PATTERNS)
    if prescription_pattern:
        return RuleIntentMatch(
            intent_code="HIGH_RISK_MEDICAL",
            route_target=HIGH_RISK_ROUTE,
            risk_level=RiskLevel.HIGH,
            confidence=1.0,
            matched_rule="hard_rule_prescription",
            matched_texts=[prescription_pattern],
            safety_flags=["HIGH_RISK_INTENT"],
        )

    return None


def match_business_intent(
    text: str,
    domain_result: DomainNormalizationResult,
) -> RuleIntentMatch | None:
    if _first_pattern_match(text, GENERAL_CHAT_PATTERNS):
        return RuleIntentMatch(
            intent_code="GENERAL_CHAT",
            route_target=GENERAL_CHAT_ROUTE,
            risk_level=RiskLevel.LOW,
            confidence=0.95,
            matched_rule="hard_rule_general_chat",
        )

    if _first_pattern_match(text, TONGUE_CONTINUE_PATTERNS):
        return RuleIntentMatch(
            intent_code="TONGUE_ANALYSIS_CONTINUE",
            route_target=TONGUE_ANALYSIS_ROUTE,
            risk_level=RiskLevel.LOW,
            confidence=0.95,
            matched_rule="hard_rule_tongue_analysis_continue",
        )

    if _first_pattern_match(text, PRIVACY_PATTERNS):
        return RuleIntentMatch(
            intent_code="PRIVACY_REQUEST",
            route_target=PRIVACY_ROUTE,
            risk_level=RiskLevel.MEDIUM,
            confidence=0.95,
            matched_rule="hard_rule_privacy_request",
        )

    if _first_pattern_match(text, TREND_PATTERNS):
        return RuleIntentMatch(
            intent_code="TREND_ANALYSIS",
            route_target=TREND_ANALYSIS_ROUTE,
            risk_level=RiskLevel.LOW,
            confidence=0.9,
            matched_rule="hard_rule_trend_analysis",
        )

    if _first_pattern_match(text, REPORT_PATTERNS):
        return RuleIntentMatch(
            intent_code="REPORT_EXPLANATION",
            route_target=REPORT_EXPLANATION_ROUTE,
            risk_level=RiskLevel.LOW,
            confidence=0.92,
            matched_rule="hard_rule_report_explanation",
        )

    if _first_pattern_match(text, TONGUE_ANALYSIS_PATTERNS):
        return RuleIntentMatch(
            intent_code="TONGUE_ANALYSIS_START",
            route_target=TONGUE_ANALYSIS_ROUTE,
            risk_level=RiskLevel.LOW,
            confidence=0.95,
            matched_rule="hard_rule_tongue_analysis",
        )

    related_intents = {
        intent
        for term in domain_result.normalized_terms
        for intent in term.related_intents
    }
    health_question = _first_pattern_match(text, HEALTH_QA_PATTERNS)
    if health_question and "HEALTH_KNOWLEDGE_QA" in related_intents:
        return RuleIntentMatch(
            intent_code="HEALTH_KNOWLEDGE_QA",
            route_target=HEALTH_QA_ROUTE,
            risk_level=RiskLevel.LOW,
            confidence=0.92,
            matched_rule="hard_rule_domain_health_qa",
            matched_texts=[health_question],
        )

    return None
