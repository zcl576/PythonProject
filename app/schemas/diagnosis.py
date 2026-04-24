from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DiagnosisAgentRequest(BaseModel):
    project_id: int = Field(..., description="项目ID")
    question: str | None = Field(default=None, description="自然语言问题")
    person_id: str | None = None
    telephone: str | None = None
    card_no: str | None = None
    device_id: str | None = None


class SuggestedAction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    action: str
    action_name: str = Field(alias="actionName")
    description: str
    risk_level: str = Field(alias="riskLevel")


class DiagnosisAgentResponse(BaseModel):
    normalized_request: dict[str, Any]
    answer: str
    summary: str
    main_cause: str
    main_cause_name: str
    confidence: float | None = None
    agent_mode: str = "rule"
    llm_used: bool = False
    warnings: list[str] = Field(default_factory=list)
    evidences: list[str] = Field(default_factory=list)
    suggested_actions: list[SuggestedAction] = Field(default_factory=list)
    context: dict[str, Any] | None = None
    diagnosis: dict[str, Any] | None = None
