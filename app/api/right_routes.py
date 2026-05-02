from fastapi import APIRouter, Header

from app.schemas.right_schema import AgentRequest, AgentResponse
from app.services.right_agent_service import RightAgentService


router = APIRouter()
agent_service = RightAgentService()

@router.post(path="/api/web/agent",response_model=AgentResponse)
async def agent(
    request: AgentRequest,
    project_id: str | None = Header(None,alias="Project-Id",encoding="utf-8")
    ) -> AgentResponse:
    return await agent_service.do_execute(request,project_id)