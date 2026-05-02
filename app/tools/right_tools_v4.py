from typing import Any

from app.clients.estate_ai_client_v4 import EstateAiClientV4
from app.services.right_agent_metadata_v4 import TOOL_METADATA


class RightToolsV4:
    def __init__(self, client: EstateAiClientV4 | None = None) -> None:
        self._client = client or EstateAiClientV4()

    async def execute(
        self,
        tool_name: str,
        project_id: int,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = TOOL_METADATA[tool_name]
        if not metadata.configured:
            return {
                "status": "not_configured",
                "tool": tool_name,
                "message": f"{metadata.description}能力尚未配置后端接口。",
            }
        if tool_name == "search_person":
            return await self.search_person(
                project_id,
                telephone=slots.get("telephone"),
                person_name=slots.get("personName"),
                person_id=slots.get("personId"),
            )
        if tool_name == "diagnose_access_issue":
            return await self.diagnose_access_issue(project_id, slots)
        if tool_name == "query_permission":
            return await self.query_permission(project_id, slots)
        if tool_name == "renew_permission":
            return await self.renew_permission(project_id, slots)
        return {
            "status": "not_configured",
            "tool": tool_name,
            "message": f"{metadata.description}能力尚未配置执行器。",
        }

    async def search_person(
        self,
        project_id: int,
        *,
        telephone: str | None = None,
        person_name: str | None = None,
        person_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._client.search_person(
            project_id,
            self._clean(
                {
                    "telephone": telephone,
                    "personName": person_name,
                    "personId": person_id,
                }
            ),
        )

    async def diagnose_access_issue(
        self,
        project_id: int,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._client.diagnose_access_issue(project_id, self._clean(slots))

    async def renew_permission(
        self,
        project_id: int,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._client.renew_permission(project_id, self._clean(slots))

    async def query_permission(
        self,
        project_id: int,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._client.query_permission(project_id, self._clean(slots))

    @staticmethod
    def _clean(payload: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in payload.items() if value not in (None, "", [], {})}
