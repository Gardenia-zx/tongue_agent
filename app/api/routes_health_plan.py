import json
import re
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.integrations.model_gateway import ModelGatewayError, get_chat_model_client


router = APIRouter(prefix="/agent/health-plan", tags=["health-plan"])


class HealthPlanAgentRequest(BaseModel):
    mode: Literal["review", "generate_detailed"]
    plan: dict[str, Any]
    report: dict[str, Any] = Field(default_factory=dict)
    state_snapshot: dict[str, Any] = Field(default_factory=dict)
    personalization_signals: list[str] = Field(default_factory=list)


class HealthPlanAgentResponse(BaseModel):
    status: Literal["REASONABLE", "NEEDS_IMPROVEMENT", "COMPLETED", "FAILED"]
    summary: str = ""
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    recommended_action: Literal["ACTIVATE", "GENERATE_DETAILED", "NONE"] = "NONE"
    days: list[dict[str, Any]] = Field(default_factory=list)
    error_message: str | None = None


_REVIEW_SYSTEM_PROMPT = """你是健康计划安全评估助手。只做一般健康管理参考，不做疾病诊断，不提供药物处方。
你会收到来源舌象报告、近3天状态、个性化信号和用户编辑后的7天计划。
请评估计划是否具体、可执行、前后均衡，并检查是否违反用户忌口、过敏、运动损伤或自由描述限制。
必须只返回JSON：
{
  "status": "REASONABLE" 或 "NEEDS_IMPROVEMENT",
  "summary": "简洁结论",
  "issues": ["问题点"],
  "suggestions": ["调整建议"],
  "recommended_action": "ACTIVATE" 或 "GENERATE_DETAILED"
}
评估只给结论和修改建议，不覆盖计划。信息不足时采用保守判断，不得虚构用户状态。"""


_GENERATE_SYSTEM_PROMPT = """你是7天健康计划生成助手。只做一般健康管理参考，不做疾病诊断，不提供药物处方。
必须基于来源报告、近3天状态、个性化信号以及用户现有草稿，生成更具体且可执行的7天计划。
用户草稿或自由描述中的忌口、过敏、运动损伤和生活限制优先级最高，必须保留并避免冲突。
必须只返回JSON：
{
  "status": "COMPLETED",
  "summary": "生成说明",
  "days": [
    {
      "dayIndex": 1,
      "date": "YYYY-MM-DD",
      "diet": {
        "breakfast": ["具体食物和份量"],
        "lunch": ["具体食物和份量"],
        "dinner": ["具体食物和份量"],
        "avoid": ["避免项"]
      },
      "exercise": {
        "activity": "具体项目",
        "durationMinutes": 20,
        "intensity": "具体强度",
        "warmup": ["热身动作"],
        "cooldown": ["放松动作"]
      },
      "sleep": {
        "targetBedtime": "23:30",
        "targetWakeTime": "07:30",
        "actions": ["睡前行动"]
      },
      "observations": ["当天观察项"]
    }
  ]
}
必须恰好7天，每天都包含早餐、午餐、晚餐、避免项、运动项目、时长、强度、热身、放松、睡眠时间、睡前行动和观察项。"""


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("model response is not a JSON object")
    return payload


def _valid_day(day: Any) -> bool:
    if not isinstance(day, dict):
        return False
    diet = day.get("diet") or {}
    exercise = day.get("exercise") or {}
    sleep = day.get("sleep") or {}
    required_meals = all(isinstance(diet.get(key), list) and bool(diet.get(key)) for key in ("breakfast", "lunch", "dinner"))
    valid_exercise = (
        isinstance(exercise.get("activity"), str)
        and bool(exercise.get("activity", "").strip())
        and isinstance(exercise.get("durationMinutes"), int)
        and 5 <= exercise.get("durationMinutes") <= 180
        and isinstance(exercise.get("intensity"), str)
        and bool(exercise.get("intensity", "").strip())
        and isinstance(exercise.get("warmup"), list)
        and bool(exercise.get("warmup"))
        and isinstance(exercise.get("cooldown"), list)
        and bool(exercise.get("cooldown"))
    )
    valid_sleep = (
        isinstance(sleep.get("targetBedtime"), str)
        and bool(sleep.get("targetBedtime", "").strip())
        and isinstance(sleep.get("targetWakeTime"), str)
        and bool(sleep.get("targetWakeTime", "").strip())
        and isinstance(sleep.get("actions"), list)
        and bool(sleep.get("actions"))
    )
    return required_meals and valid_exercise and valid_sleep and isinstance(day.get("observations"), list)


def _normalize_review(payload: dict[str, Any]) -> HealthPlanAgentResponse:
    status = str(payload.get("status") or "FAILED").upper()
    if status not in {"REASONABLE", "NEEDS_IMPROVEMENT"}:
        raise ValueError("invalid review status")
    action = "ACTIVATE" if status == "REASONABLE" else "GENERATE_DETAILED"
    return HealthPlanAgentResponse(
        status=status,
        summary=str(payload.get("summary") or ""),
        issues=[str(item) for item in payload.get("issues") or [] if str(item).strip()][:12],
        suggestions=[str(item) for item in payload.get("suggestions") or [] if str(item).strip()][:12],
        recommended_action=action,
    )


def _normalize_generation(payload: dict[str, Any]) -> HealthPlanAgentResponse:
    days = payload.get("days") or []
    if not isinstance(days, list) or len(days) != 7 or not all(_valid_day(day) for day in days):
        raise ValueError("generated plan is incomplete")
    indexes = {int(day.get("dayIndex", 0)) for day in days}
    if indexes != set(range(1, 8)):
        raise ValueError("generated plan day indexes are invalid")
    return HealthPlanAgentResponse(
        status="COMPLETED",
        summary=str(payload.get("summary") or "已生成更具体的7天健康计划。"),
        days=days,
    )


@router.post("/review", response_model=HealthPlanAgentResponse)
async def review_health_plan(request: HealthPlanAgentRequest) -> HealthPlanAgentResponse:
    system_prompt = _REVIEW_SYSTEM_PROMPT if request.mode == "review" else _GENERATE_SYSTEM_PROMPT
    user_payload = {
        "mode": request.mode,
        "source_report": request.report,
        "state_snapshot": request.state_snapshot,
        "personalization_signals": request.personalization_signals,
        "current_plan": request.plan,
    }
    try:
        content = await get_chat_model_client().generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=0.2,
            max_tokens=5000 if request.mode == "generate_detailed" else 1400,
            extra_body={"response_format": {"type": "json_object"}},
        )
        payload = _extract_json(content)
        return _normalize_review(payload) if request.mode == "review" else _normalize_generation(payload)
    except (ModelGatewayError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return HealthPlanAgentResponse(
            status="FAILED",
            summary="AI健康计划处理失败，原草稿未发生变化。",
            recommended_action="NONE",
            error_message=str(exc)[:500],
        )
