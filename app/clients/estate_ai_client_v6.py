from typing import Any


class EstateAiClientV6:
    async def search_person(self, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self._not_configured("search_person")

    async def search_device(self, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self._not_configured("search_device")

    async def query_permission(self, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self._not_configured("query_permission")

    async def extend_permission(self, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self._not_configured("extend_permission")

    async def enable_permission(self, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self._not_configured("enable_permission")

    async def disable_permission(self, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self._not_configured("disable_permission")

    async def grant_permission(self, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self._not_configured("grant_permission")

    async def revoke_permission(self, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self._not_configured("revoke_permission")

    @staticmethod
    def _not_configured(tool_name: str) -> dict[str, Any]:
        return {
            "status": "not_configured",
            "tool": tool_name,
            "message": "Estate AI v6 API contract is not configured.",
        }
