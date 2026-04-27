from typing import Any

from pydantic import BaseModel

from app.clients.estate_ai_client import EstateAiClient


class ToolCallResult(BaseModel):
    name: str
    input: dict[str, Any]
    output: dict[str, Any]


class AccessDiagnosisToolExecutor:
    def __init__(self, client: EstateAiClient) -> None:
        self._client = client

    async def run_diagnosis(self, project_id: int, normalized: dict[str, Any]) -> ToolCallResult:
        result = await self._client.get_result(project_id, normalized)
        return ToolCallResult(
            name="cloudx.access_diagnosis.result",
            input={"project_id": project_id, "payload": normalized},
            output=result,
        )
