from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agent.trace import AgentTrace


class DiagnosisAgentRequest(BaseModel):
    session_id: str | None = Field(default=None, description="会话ID")
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
    trace_id: str
    session_id: str
    status: str
    normalized_request: dict[str, Any]
    answer: str
    follow_up_question: str | None = None
    needs_input: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    trace: AgentTrace | None = None
    available_actions: list[dict[str, Any]] = Field(default_factory=list)
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
