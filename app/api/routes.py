from fastapi import APIRouter
from fastapi import Header

from app.schemas.diagnosis import DiagnosisAgentRequest, DiagnosisAgentResponse, FrontendDiagnosisResponse
from app.services.access_diagnosis_agent import AccessDiagnosisAgentService

router = APIRouter()
service = AccessDiagnosisAgentService()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/agent/access-diagnosis", response_model=DiagnosisAgentResponse)
async def diagnose(
    request: DiagnosisAgentRequest,
    project_id: str | None = Header(None, alias="PROJECT-ID"),
) -> DiagnosisAgentResponse:
    return await service.diagnose(request, project_id)


@router.post("/api/web/agent/access-diagnosis", response_model=FrontendDiagnosisResponse)
async def diagnose_for_frontend(
    request: DiagnosisAgentRequest,
    project_id: str | None = Header(None, alias="PROJECT-ID"),
) -> FrontendDiagnosisResponse:
    response = await service.diagnose(request, project_id)
    return FrontendDiagnosisResponse(
        answer=response.answer,
        status=response.status,
        follow_up_question=response.follow_up_question,
        needs_input=response.needs_input,
    )

