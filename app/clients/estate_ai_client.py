from typing import Any

import httpx

from app.config import get_settings


class EstateAiClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.estate_ai_base_url.rstrip("/")
        self._timeout = settings.estate_ai_timeout_seconds

    async def get_context(self, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/ai/access-diagnosis/context", project_id, payload)

    async def get_result(self, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/ai/access-diagnosis/result", project_id, payload)

    async def _post(self, path: str, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            response = await client.post(path, headers={"PROJECT-ID": str(project_id)}, json=payload)
            response.raise_for_status()
            return response.json()
