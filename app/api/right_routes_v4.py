from fastapi import APIRouter, Header

from app.schemas.right_schema_v4 import AgentRequestV4, AgentResponseV4
from app.services.right_agent_graph_service_v4 import RightAgentGraphServiceV4

router = APIRouter()
agent_service_v4 = RightAgentGraphServiceV4()


@router.post(path="/api/web/agent/v4", response_model=AgentResponseV4)
async def agent_v4(
    request: AgentRequestV4,
    project_id: str | None = Header(None, alias="Project-Id", encoding="utf-8"),
) -> AgentResponseV4:
    return await agent_service_v4.do_execute(request, project_id)
