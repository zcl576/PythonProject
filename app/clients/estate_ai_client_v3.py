from typing import Any

import httpx

from app.config import get_settings


class EstateAiClientV3:
    def __init__(self, renew_permission_path: str | None = None) -> None:
        settings = get_settings()
        self._base_url = settings.estate_ai_base_url.rstrip("/")
        self._timeout = settings.estate_ai_timeout_seconds
        self._renew_permission_path = (
            renew_permission_path
            or getattr(settings, "estate_ai_renew_permission_path", None)
            or "/ai/access-permission/renew"
        )

    async def search_person(self, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/ai/access-diagnosis/context", project_id, payload)

    async def diagnose_access_issue(
        self, project_id: int, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._post("/ai/access-diagnosis/result", project_id, payload)

    async def renew_permission(
        self, project_id: int, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._post(self._renew_permission_path, project_id, payload)

    async def _post(
        self, path: str, project_id: int, payload: dict[str, Any]
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            response = await client.post(
                path,
                headers={"PROJECT-ID": str(project_id)},
                json=payload,
            )
            response.raise_for_status()
            return response.json()
