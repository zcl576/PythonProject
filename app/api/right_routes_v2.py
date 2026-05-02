from fastapi import APIRouter, Header

from app.schemas.right_schema_v2 import AgentRequestV2, AgentResponseV2
from app.services.right_agent_graph_service import RightAgentGraphService

router = APIRouter()
agent_service_v2 = RightAgentGraphService()


@router.post(path="/api/web/agent/v2", response_model=AgentResponseV2)
async def agent_v2(
    request: AgentRequestV2,
    project_id: str | None = Header(None, alias="Project-Id", encoding="utf-8"),
) -> AgentResponseV2:
    return await agent_service_v2.do_execute(request, project_id)
