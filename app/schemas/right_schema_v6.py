from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentChoiceV6(BaseModel):
    id: str
    label: str
    description: str | None = None
    data: dict[str, Any] | None = None


class AgentConfirmV6(BaseModel):
    id: str
    label: str
    description: str | None = None
    risk_level: str | None = None


class AgentResumeV6(BaseModel):
    type: Literal["choice", "confirm"]
    choice_id: str | None = None
    confirm_id: str | None = None
    confirmed: bool | None = None


class AgentRequestV6(BaseModel):
    question: str | None = Field(default=None, description="User question")
    session_id: str | None = Field(default=None, description="Session ID")
    resume: AgentResumeV6 | dict[str, Any] | None = Field(
        default=None,
        description="LangGraph interrupt resume payload",
    )


class AgentResponseV6(BaseModel):
    answer: str
    status: str
    session_id: str
    follow_up_question: str | None = None
    choices: list[AgentChoiceV6] = Field(default_factory=list)
    confirm: AgentConfirmV6 | None = None
    needs_input: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
