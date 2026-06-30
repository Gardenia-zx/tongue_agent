import json
import re
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.integrations.model_gateway import ModelGatewayError, get_chat_model_client


router = APIRouter(prefix="/agent/health-plan", tags=["health-plan-agent"])


class HealthPlanDiet(BaseModel):
    breakfast: list[str] = Field(default_factory=list)
    lunch: list[str] = Field(default_factory=list)
    dinner: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)

    @field_validator("breakfast", "lunch", "dinner", "avoid")
    @classmethod
    def clean_items(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            item = str(value).strip()
            if item and item not in result:
                result.append(item[:100])
        return result[:12]


class HealthPlanExercise(BaseModel):
    activity: str
    durationMinutes: int = Field(ge=5, le=180)
    intensity: str
    warmup: list[str] = Field(default_factory=list)
    cooldown: list[str] = Field(default_factory=list)

    @field_validator("activity", "intensity")
    @classmethod
    def require_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field cannot be empty")
        return cleaned[:120]

    @field_validator("warmup", "cooldown")
    @classmethod
    def clean_actions(cls, values: list[str]) -> list[str]:
        return [str(value).strip()[:100] for value in values if str(value).strip()][:8]

    @model_validator(mode="after")
    def require_warmup_and_cooldown(self) -> "HealthPlanExercise":
        if not self.warmup or not self.cooldown:
            raise ValueError("warmup and cooldown must both be present")
        return self


class HealthPlanSleep(BaseModel):
    targetBedtime: str
    targetWakeTime: str
    actions: list[str] = Field(default_factory=list)

    @field_validator("targetBedtime", "targetWakeTime")
    @classmethod
    def validate_time(cls, value: str) -> str:
        cleaned = value.strip()
        if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", cleaned):
            raise ValueError("time must use HH:mm")
        return cleaned

    @field_validator("actions")
    @classmethod
    def require_actions(cls, values: list[str]) -> list[str]:
        cleaned = [str(value).strip()[:120] for value in values if str(value).strip()]
        if not cleaned:
            raise ValueError("sleep actions cannot be empty")
        return cleaned[:10]


class HealthPlanDay(BaseModel):
    dayIndex: int = Field(ge=1, le=7)
    date: str | None = None
    diet: HealthPlanDiet
    exercise: HealthPlanExercise
    sleep: HealthPlanSleep
    observations: list[str] = Field(default_factory=list)

    @field_validator("observations")
    @classmethod
    def clean_observations(cls, values: list[str]) -> list[str]:
        return [str(value).strip()[:120] for value in values if str(value).strip()][:12]

    @model_validator(mode="after")
    def require_complete_day(self) -> "HealthPlanDay":
        if not self.diet.breakfast or not self.diet.lunch or not self.diet.dinner:
            raise ValueError("breakfast, lunch and dinner must all be present")
        if not self.diet.avoid:
            raise ValueError("avoid items must be present")
        if not self.observations:
            raise ValueError("observations must be present")
        return self


class HealthPlanAgentRequest(BaseModel):
    mode: Literal["review", "generate_detailed"]
    plan_id: int | None = None
    source_report_id: int | None = None
    plan_days: list[dict[str, Any]] = Field(default_factory=list)
    draft_report: dict[str, Any] = Field(default_factory=dict)
    state_snapshot: dict[str, Any] | None = None
    personalization_signals: list[str] = Field(default_factory=list)


class HealthPlanReviewPayload(BaseModel):
    status: Literal["REASONABLE", "NEEDS_IMPROVEMENT"]
    summary: str
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    recommended_action: Literal["ACTIVATE", "GENERATE_DETAILED"]


class HealthPlanGenerationPayload(BaseModel):
    status: Literal["COMPLETED"] = "COMPLETED"
    summary: str = "已生成更具体的 7 天健康计划。"
    days: list[HealthPlanDay]

    @model_validator(mode="after")
    def validate_complete_week(self) -> "HealthPlanGenerationPayload":
        if len(self.days) != 7:
            raise ValueError("generated plan must contain exactly seven days")
        indexes = sorted(day.dayIndex for day in self.days)
        if indexes != list(range(1, 8)):
            raise ValueError("dayIndex must contain 1 through 7 exactly once")
        self.days.sort(key=lambda day: day.dayIndex)
        return self


class HealthPlanAgentResponse(BaseModel):
    status: Literal["REASONABLE", "NEEDS_IMPROVEMENT", "COMPLETED", "FAILED"]
    summary: str
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    recommended_action: Literal["ACTIVATE", "GENERATE_DETAILED"] | None = None
    days: list[HealthPlanDay] | None = None


REVIEW_SYSTEM_PROMPT = """你是健康计划质量评估助手，只做日常健康管理参考，不做疾病诊断，不提供药物或处方建议。

请评估用户的 7 天饮食、运动、睡眠和观察计划是否具体、可执行、内部一致，并与来源舌象报告、近 3 天状态及个性化信号保持一致。

评估原则：
1. 每天必须有具体早餐、午餐、晚餐，不接受只有“清淡饮食”等空泛表述。
2. 运动必须包含项目、时长、强度；存在膝盖不适等限制时，不得推荐冲突项目。
3. 睡眠必须包含入睡时间、起床时间和具体行动。
4. 用户自由描述中的过敏、忌口、损伤和明确限制优先级最高。
5. 不因为一次舌象结果制造疾病结论，也不夸大风险。
6. 计划整体合理且足够具体时返回 REASONABLE；存在明显泛化、冲突、缺项或不可执行内容时返回 NEEDS_IMPROVEMENT。

仅返回 JSON：
{
  "status": "REASONABLE 或 NEEDS_IMPROVEMENT",
  "summary": "简明评估结论",
  "issues": ["具体问题"],
  "suggestions": ["具体修改建议"],
  "recommended_action": "ACTIVATE 或 GENERATE_DETAILED"
}
"""


GENERATION_SYSTEM_PROMPT = """你是 7 天健康管理计划生成助手。只生成日常饮食、运动、睡眠与观察计划，不做疾病诊断，不提供药物、处方或治疗方案。

必须依据来源舌象报告、近 3 天状态、个性化信号和用户当前草稿，生成更具体、可执行的完整 7 天计划。

强制规则：
1. 必须返回完整 7 天，dayIndex 从 1 到 7 且不重复。
2. 每天都包含早餐、午餐、晚餐和避免项；食物必须具体并带合理份量示例。
3. 每天运动必须包含 activity、durationMinutes、intensity、warmup、cooldown，热身和放松都不得为空。
4. 每天睡眠必须包含 targetBedtime、targetWakeTime、actions，时间格式 HH:mm。
5. 每天包含至少一个 observations。
6. 优先保留用户草稿中明确修改过的偏好；自由描述中的过敏、忌口、运动损伤或“不喝牛奶”等限制不得冲突。
7. 内容以常见、容易获得的日常食物和低风险活动为主，不进行极端节食或突然高强度训练。
8. 不自动启用计划，只返回可供用户确认的草稿。

仅返回 JSON：
{
  "status": "COMPLETED",
  "summary": "本次生成重点",
  "days": [
    {
      "dayIndex": 1,
      "date": null,
      "diet": {
        "breakfast": ["具体食物"],
        "lunch": ["具体食物"],
        "dinner": ["具体食物"],
        "avoid": ["具体避免项"]
      },
      "exercise": {
        "activity": "运动项目",
        "durationMinutes": 20,
        "intensity": "强度描述",
        "warmup": ["热身动作"],
        "cooldown": ["放松动作"]
      },
      "sleep": {
        "targetBedtime": "23:30",
        "targetWakeTime": "07:30",
        "actions": ["具体行动"]
      },
      "observations": ["观察项"]
    }
  ]
}
"""


@router.post("/review", response_model=HealthPlanAgentResponse)
async def review_or_generate_health_plan(
    request: HealthPlanAgentRequest,
) -> HealthPlanAgentResponse:
    try:
        context = _build_context(request)
        if request.mode == "review":
            raw = await _call_model(REVIEW_SYSTEM_PROMPT, context, max_tokens=1800)
            payload = HealthPlanReviewPayload.model_validate(_extract_json_object(raw))
            return HealthPlanAgentResponse(**payload.model_dump())

        raw = await _call_model(GENERATION_SYSTEM_PROMPT, context, max_tokens=7000)
        payload = HealthPlanGenerationPayload.model_validate(_extract_json_object(raw))
        return HealthPlanAgentResponse(
            status="COMPLETED",
            summary=payload.summary,
            days=payload.days,
        )
    except (ModelGatewayError, ValidationError, ValueError, json.JSONDecodeError) as exc:
        return HealthPlanAgentResponse(
            status="FAILED",
            summary=f"AI 健康计划处理失败：{type(exc).__name__}",
            issues=[str(exc)[:500]],
            suggestions=["请保留当前草稿，稍后重新提交。"],
        )


async def _call_model(system_prompt: str, context: str, *, max_tokens: int) -> str:
    client = get_chat_model_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context},
    ]
    try:
        return await client.generate(
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
            extra_body={"response_format": {"type": "json_object"}},
        )
    except ModelGatewayError as exc:
        if "response_format" not in str(exc).lower():
            raise
        return await client.generate(
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
        )


def _build_context(request: HealthPlanAgentRequest) -> str:
    payload = {
        "plan_id": request.plan_id,
        "source_report_id": request.source_report_id,
        "source_report": request.draft_report,
        "state_snapshot": request.state_snapshot,
        "personalization_signals": request.personalization_signals,
        "current_plan_days": request.plan_days,
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(serialized) > 60000:
        serialized = serialized[:60000]
    return "请根据以下输入完成任务。输入 JSON：\n" + serialized


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])

    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value
