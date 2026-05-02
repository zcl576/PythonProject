from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    question: str | None = Field(default=None, description="用户提问内容")
    session_id: str | None = Field(default=None, description="会话ID")


class AgentResponse(BaseModel):
    answer: str
    status: str
    follow_up_question: str