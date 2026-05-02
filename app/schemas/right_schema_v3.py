from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentChoiceV3(BaseModel):
    id: str
    label: str
    description: str | None = None
    data: dict[str, Any] | None = None


class AgentConfirmV3(BaseModel):
    id: str
    label: str
    description: str | None = None
    risk_level: str | None = None


class AgentResumeV3(BaseModel):
    type: Literal["choice", "confirm"]
    choice_id: str | None = None
    confirm_id: str | None = None
    confirmed: bool | None = None


class AgentRequestV3(BaseModel):
    question: str | None = Field(default=None, description="用户提问内容")
    session_id: str | None = Field(default=None, description="会话 ID")
    resume: AgentResumeV3 | dict[str, Any] | None = Field(
        default=None,
        description="LangGraph 中断续跑数据",
    )


class AgentResponseV3(BaseModel):
    answer: str
    status: str
    session_id: str
    follow_up_question: str | None = None
    choices: list[AgentChoiceV3] = Field(default_factory=list)
    confirm: AgentConfirmV3 | None = None
    needs_input: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
