from typing import Any, Awaitable, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.clients.estate_ai_client_v5 import EstateAiClientV5
from app.services.right_agent_metadata_v5 import READ_TARGET_TOOLS_V5, TOOL_METADATA_V5

ToolHandlerV5 = Callable[[int, dict[str, Any]], Awaitable[dict[str, Any]]]


class SearchPersonInputV5(BaseModel):
    project_id: int = Field(description="项目 ID")
    personId: str | None = Field(default=None, description="人员 ID")
    personName: str | None = Field(default=None, description="人员姓名")
    telephone: str | None = Field(default=None, description="手机号")


class SearchDeviceInputV5(BaseModel):
    project_id: int = Field(description="项目 ID")
    deviceSn: str | None = Field(default=None, description="设备 SN")
    deviceId: str | None = Field(default=None, description="设备 ID")
    deviceName: str | None = Field(default=None, description="设备名称")


class QueryPermissionInputV5(BaseModel):
    project_id: int = Field(description="项目 ID")
    personId: str = Field(description="人员 ID")
    deviceSn: str = Field(description="设备 SN")


class RightToolsV5:
    def __init__(self, client: EstateAiClientV5 | None = None) -> None:
        self._client = client or EstateAiClientV5()
        self._registry: dict[str, ToolHandlerV5] = {
            "search_person": self.search_person,
            "search_device": self.search_device,
            "query_permission": self.query_permission,
            "renew_permission": self.renew_permission,
        }

    @property
    def registry(self) -> dict[str, ToolHandlerV5]:
        return dict(self._registry)

    def read_langchain_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                coroutine=self.search_person_tool,
                name="search_person",
                description=(
                    "查询人员信息。用于根据人员姓名、手机号或人员 ID 获取 personId。"
                    "这是只读工具。"
                ),
                args_schema=SearchPersonInputV5,
            ),
            StructuredTool.from_function(
                coroutine=self.search_device_tool,
                name="search_device",
                description=(
                    "查询门禁设备信息。用于根据设备名称、设备 ID 或设备 SN 获取 deviceSn。"
                    "这是只读工具。"
                ),
                args_schema=SearchDeviceInputV5,
            ),
            StructuredTool.from_function(
                coroutine=self.query_permission_tool,
                name="query_permission",
                description=(
                    "查询人员对门禁设备的权限状态。需要 personId 和 deviceSn。"
                    "这是只读工具；如果结果显示权限过期，续期仍需用户确认后由系统执行。"
                ),
                args_schema=QueryPermissionInputV5,
            ),
        ]

    async def execute(
        self,
        tool_name: str,
        project_id: int,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        if tool_name not in TOOL_METADATA_V5 or tool_name not in self._registry:
            return {
                "status": "not_configured",
                "tool": tool_name,
                "message": "Unknown v5 tool.",
            }
        return await self._registry[tool_name](project_id, slots)

    async def search_person_tool(
        self,
        project_id: int,
        personId: str | None = None,
        personName: str | None = None,
        telephone: str | None = None,
    ) -> dict[str, Any]:
        return await self.search_person(
            project_id,
            {
                "personId": personId,
                "personName": personName,
                "telephone": telephone,
            },
        )

    async def search_device_tool(
        self,
        project_id: int,
        deviceSn: str | None = None,
        deviceId: str | None = None,
        deviceName: str | None = None,
    ) -> dict[str, Any]:
        return await self.search_device(
            project_id,
            {
                "deviceSn": deviceSn,
                "deviceId": deviceId,
                "deviceName": deviceName,
            },
        )

    async def query_permission_tool(
        self,
        project_id: int,
        personId: str,
        deviceSn: str,
    ) -> dict[str, Any]:
        return await self.query_permission(
            project_id,
            {
                "personId": personId,
                "deviceSn": deviceSn,
            },
        )

    async def search_person(
        self,
        project_id: int,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._client.search_person(
            project_id,
            self._clean(
                {
                    "personId": slots.get("personId"),
                    "personName": slots.get("personName"),
                    "telephone": slots.get("telephone"),
                }
            ),
        )

    async def search_device(
        self,
        project_id: int,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._client.search_device(
            project_id,
            self._clean(
                {
                    "deviceSn": slots.get("deviceSn"),
                    "deviceId": slots.get("deviceId"),
                    "deviceName": slots.get("deviceName"),
                }
            ),
        )

    async def query_permission(
        self,
        project_id: int,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._client.query_permission(
            project_id,
            self._clean(
                {
                    "personId": slots.get("personId"),
                    "deviceSn": slots.get("deviceSn"),
                }
            ),
        )

    async def renew_permission(
        self,
        project_id: int,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._client.renew_permission(
            project_id,
            self._clean(
                {
                    "personId": slots.get("personId"),
                    "deviceSn": slots.get("deviceSn"),
                }
            ),
        )

    @staticmethod
    def _clean(payload: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


def read_langchain_tools_v5(client: EstateAiClientV5 | None = None) -> list[StructuredTool]:
    return RightToolsV5(client=client).read_langchain_tools()
