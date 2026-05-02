from typing import Any

from app.clients.estate_ai_client_v3 import EstateAiClientV3


class RightToolsV3:
    def __init__(self, client: EstateAiClientV3 | None = None) -> None:
        self._client = client or EstateAiClientV3()

    async def search_person(
        self,
        project_id: int,
        *,
        telephone: str | None = None,
        person_name: str | None = None,
    ) -> dict[str, Any]:
        return await self._client.search_person(
            project_id,
            self._clean(
                {
                    "telephone": telephone,
                    "personName": person_name,
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

    @staticmethod
    def _clean(payload: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in payload.items() if value not in (None, "", [], {})}
