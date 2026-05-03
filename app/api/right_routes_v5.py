from fastapi import APIRouter, Header

from app.schemas.right_schema_v5 import AgentRequestV5, AgentResponseV5
from app.services.right_agent_graph_service_v5 import RightAgentGraphServiceV5

router = APIRouter()
agent_service_v5 = RightAgentGraphServiceV5()


@router.post(path="/api/web/agent/v5", response_model=AgentResponseV5)
async def agent_web_v5(
    request: AgentRequestV5,
    project_id: str | None = Header(None, alias="Project-Id", encoding="utf-8"),
) -> AgentResponseV5:
    return await agent_service_v5.do_execute(request, project_id)


@router.post(path="/api/agent/v5", response_model=AgentResponseV5)
async def agent_v5(
    request: AgentRequestV5,
    project_id: str | None = Header(None, alias="Project-Id", encoding="utf-8"),
) -> AgentResponseV5:
    return await agent_service_v5.do_execute(request, project_id)
