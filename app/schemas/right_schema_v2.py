from typing import Any

from pydantic import BaseModel, Field


class AgentChoice(BaseModel):
    id: str
    label: str
    description: str | None = None
    data: dict[str, Any] | None = None


class AgentConfirm(BaseModel):
    id: str
    label: str
    description: str | None = None
    risk_level: str | None = None


class AgentRequestV2(BaseModel):
    question: str | None = Field(default=None, description="用户提问内容")
    session_id: str | None = Field(default=None, description="会话 ID")
    choice_id: str | None = Field(default=None, description="用户选择项 ID")
    confirm_id: str | None = Field(default=None, description="用户确认操作 ID")


class AgentResponseV2(BaseModel):
    answer: str
    status: str
    session_id: str
    follow_up_question: str | None = None
    choices: list[AgentChoice] = Field(default_factory=list)
    confirm: AgentConfirm | None = None
    needs_input: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
