from pydantic import BaseModel, ConfigDict, Field


class AgentDiagnosisRequest(BaseModel):
    model_config = ConfigDict()
    question: str | None = Field(None,description="自然语言进行问题描述")
    session_id: str | None = Field(None,description="通话的上下文id")



class AgentDiagnosisResponse(BaseModel):
    answer: str
    status: str
    follow_up_question: str | None = None
