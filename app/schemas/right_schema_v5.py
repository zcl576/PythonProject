from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentChoiceV5(BaseModel):
    id: str
    label: str
    description: str | None = None
    data: dict[str, Any] | None = None


class AgentConfirmV5(BaseModel):
    id: str
    label: str
    description: str | None = None
    risk_level: str | None = None


class AgentResumeV5(BaseModel):
    type: Literal["choice", "confirm"]
    choice_id: str | None = None
    confirm_id: str | None = None
    confirmed: bool | None = None


class AgentRequestV5(BaseModel):
    question: str | None = Field(default=None, description="User question")
    session_id: str | None = Field(default=None, description="Session ID")
    resume: AgentResumeV5 | dict[str, Any] | None = Field(
        default=None,
        description="LangGraph interrupt resume payload",
    )


class AgentResponseV5(BaseModel):
    answer: str
    status: str
    session_id: str
    follow_up_question: str | None = None
    choices: list[AgentChoiceV5] = Field(default_factory=list)
    confirm: AgentConfirmV5 | None = None
    needs_input: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
