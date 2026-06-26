import math
from dataclasses import dataclass


# 候选意图各项评分
@dataclass(frozen=True)
class IntentScore:
    # ES召回的bm25评分
    bm25_score: float = 0.0
    # 向量相似度
    vector_score: float = 0.0
    # 关键词直接匹配
    keyword_score: float = 0.0
    # 最终混合得分
    fusion_score: float = 0.0

# 限制分数在[0,1]
def clamp_score(score: float) -> float:
    return min(max(score, 0.0), 1.0)



def normalize_bm25(score: float, max_score: float) -> float:
    if max_score <= 0:
        return 0.0
    return clamp_score(score / max_score)

# 余弦相似度计算
def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0
    # 范围映射[-1,0]->0,后续部分不变
    cosine = dot / (left_norm*right_norm)
    return clamp_score(cosine)

# 关键词匹配得分计算
def calculate_keyword_score(query: str, keywords: list[str]) -> float:
    if not query or not keywords:
        return 0.0

    query_lower = query.lower()
    valid_keywords = list({
        keyword.strip().lower()
        for keyword in keywords
        if isinstance(keyword, str) and keyword.strip()
    })

    if not valid_keywords:
        return 0.0

    matched_count = sum(1 for keyword in valid_keywords if keyword in query_lower)
    return clamp_score(matched_count / len(valid_keywords))


def fuse_intent_score(
    *,
    bm25_score: float,
    max_bm25_score: float,
    vector_score: float,
    keyword_score: float,
    bm25_weight: float = 0.35,
    vector_weight: float = 0.5,
    keyword_weight: float = 0.15,
) -> IntentScore:
    if bm25_weight<=0 or vector_weight<=0 or keyword_weight<=0:
        raise ValueError("score weights must sum to a positive value")
    weight_sum = (
        bm25_weight
        + vector_weight
        + keyword_weight
    )
    normalized_bm25 = normalize_bm25(bm25_score, max_bm25_score)
    normalized_vector = clamp_score(vector_score)
    normalized_keyword = clamp_score(keyword_score)

    fusion_score = (
        normalized_bm25 * bm25_weight
        + normalized_vector * vector_weight
        + normalized_keyword * keyword_weight
    )/weight_sum

    return IntentScore(
        bm25_score=normalized_bm25,
        vector_score=normalized_vector,
        keyword_score=normalized_keyword,
        fusion_score=clamp_score(fusion_score),
    )