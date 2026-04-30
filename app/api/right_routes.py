from fastapi import Header

from fastapi import APIRouter

from app.schemas.right import AgentDiagnosisResponse, AgentDiagnosisRequest
from app.services.right_agent import RightAgent

router = APIRouter()
right_agent = RightAgent()

@router.post(path = "/api/web/agent",responses=AgentDiagnosisResponse)
async def web_agent(
    request: AgentDiagnosisRequest,
    project_id: str | None = Header(None, alias="PROJECT-ID"),
) -> AgentDiagnosisResponse:
    return await right_agent.do_execute(request, project_id)