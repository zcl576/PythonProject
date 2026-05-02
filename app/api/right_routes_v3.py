from fastapi import APIRouter, Header

from app.schemas.right_schema_v3 import AgentRequestV3, AgentResponseV3
from app.services.right_agent_graph_service_v3 import RightAgentGraphServiceV3

router = APIRouter()
agent_service_v3 = RightAgentGraphServiceV3()


@router.post(path="/api/web/agent/v3", response_model=AgentResponseV3)
async def agent_v3(
    request: AgentRequestV3,
    project_id: str | None = Header(None, alias="Project-Id", encoding="utf-8"),
) -> AgentResponseV3:
    return await agent_service_v3.do_execute(request, project_id)
