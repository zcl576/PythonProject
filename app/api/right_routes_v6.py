from fastapi import APIRouter, Header

from app.schemas.right_schema_v6 import AgentRequestV6, AgentResponseV6
from app.services.right_agent_graph_service_v6 import RightAgentGraphServiceV6

router = APIRouter()
agent_service_v6 = RightAgentGraphServiceV6()


@router.post(path="/api/web/agent/v6", response_model=AgentResponseV6)
async def agent_web_v6(
    request: AgentRequestV6,
    project_id: str | None = Header(None, alias="Project-Id", encoding="utf-8"),
) -> AgentResponseV6:
    return await agent_service_v6.do_execute(request, project_id)


@router.post(path="/api/agent/v6", response_model=AgentResponseV6)
async def agent_v6(
    request: AgentRequestV6,
    project_id: str | None = Header(None, alias="Project-Id", encoding="utf-8"),
) -> AgentResponseV6:
    return await agent_service_v6.do_execute(request, project_id)
