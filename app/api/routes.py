from fastapi import APIRouter
from fastapi.params import Header

from app.schemas.diagnosis import DiagnosisAgentRequest, DiagnosisAgentResponse
from app.services.access_diagnosis_agent import AccessDiagnosisAgentService

router = APIRouter()
service = AccessDiagnosisAgentService()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/agent/access-diagnosis", response_model=DiagnosisAgentResponse)
async def diagnose(request: DiagnosisAgentRequest,project_id: str = Header(None)) -> DiagnosisAgentResponse:
    return await service.diagnose(request,project_id)
