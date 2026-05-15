from typing import Any

import httpx

from app.config import get_settings


class EstateAiClientV6:
    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.estate_ai_base_url.rstrip("/")
        self._timeout = settings.estate_ai_timeout_seconds

    async def search_person(self, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/ai/access-diagnosis/v6/person", project_id, payload)

    async def search_device(self, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/ai/access-diagnosis/v6/device", project_id, payload)

    async def query_permission(self, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/ai/access-diagnosis/v6/permission/query", project_id, payload)

    async def extend_permission(self, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/ai/access-diagnosis/v6/permission/extend", project_id, payload)

    async def enable_permission(self, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/ai/access-diagnosis/v6/permission/enable", project_id, payload)

    async def disable_permission(self, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/ai/access-diagnosis/v6/permission/disable", project_id, payload)

    async def grant_permission(self, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/ai/access-diagnosis/v6/permission/grant", project_id, payload)

    async def revoke_permission(self, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/ai/access-diagnosis/v6/permission/revoke", project_id, payload)

    async def _post(self, path: str, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            response = await client.post(path, headers={"PROJECT-ID": str(project_id)}, json=payload)
            response.raise_for_status()
            return response.json()
