from typing import Any


class EstateAiClientV5:
    async def search_person(
        self,
        project_id: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return self._not_configured("search_person")

    async def search_device(
        self,
        project_id: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return self._not_configured("search_device")

    async def query_permission(
        self,
        project_id: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return self._not_configured("query_permission")

    async def renew_permission(
        self,
        project_id: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return self._not_configured("renew_permission")

    @staticmethod
    def _not_configured(tool_name: str) -> dict[str, Any]:
        return {
            "status": "not_configured",
            "tool": tool_name,
            "message": "Estate AI v5 API contract is not configured.",
        }
