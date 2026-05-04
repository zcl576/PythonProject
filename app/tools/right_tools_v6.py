"""权限工具执行器 V6 版本

封装了与权限相关的工具调用，通过 EstateAiClientV6 客户端执行实际操作。
支持的工具包括：人员查询、设备查询、权限查询、权限续期、权限启用/禁用、权限授予/撤销。
"""
from typing import Any, Awaitable, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.clients.estate_ai_client_v6 import EstateAiClientV6
from app.services.right_agent_metadata_v6 import READ_TOOLS_V6, TOOL_METADATA_V6

# 工具处理器类型定义：接受项目ID和槽位数据，返回异步结果
ToolHandlerV6 = Callable[[int, dict[str, Any]], Awaitable[dict[str, Any]]]


class SearchPersonInputV6(BaseModel):
    """人员查询输入参数"""
    project_id: int = Field(description="项目 ID")
    personId: str | None = Field(default=None, description="人员 ID")
    personName: str | None = Field(default=None, description="人员姓名")
    telephone: str | None = Field(default=None, description="手机号")


class SearchDeviceInputV6(BaseModel):
    """设备查询输入参数"""
    project_id: int = Field(description="项目 ID")
    deviceSn: str | None = Field(default=None, description="设备 SN")
    deviceId: str | None = Field(default=None, description="设备 ID")
    deviceName: str | None = Field(default=None, description="设备名称")


class QueryPermissionInputV6(BaseModel):
    """权限查询输入参数"""
    project_id: int = Field(description="项目 ID")
    personId: str = Field(description="人员 ID")
    deviceSn: str = Field(description="设备 SN")


class RightToolsV6:
    """权限工具执行器 V6 版本

    封装了与权限相关的工具调用，通过 EstateAiClientV6 客户端执行实际操作。

    支持的工具：
    - search_person: 人员查询（只读）
    - search_device: 设备查询（只读）
    - query_permission: 权限查询（只读）
    - extend_permission: 权限续期（写操作）
    - enable_permission: 权限启用（写操作）
    - disable_permission: 权限禁用（写操作）
    - grant_permission: 权限授予（写操作）
    - revoke_permission: 权限撤销（写操作）
    """

    def __init__(self, client: EstateAiClientV6 | None = None) -> None:
        """初始化工具执行器

        Args:
            client: EstateAiClientV6 客户端实例，如果为 None 则自动创建
        """
        self._client = client or EstateAiClientV6()
        self._registry: dict[str, ToolHandlerV6] = {
            "search_person": self.search_person,
            "search_device": self.search_device,
            "query_permission": self.query_permission,
            "extend_permission": self.extend_permission,
            "enable_permission": self.enable_permission,
            "disable_permission": self.disable_permission,
            "grant_permission": self.grant_permission,
            "revoke_permission": self.revoke_permission,
        }

    @property
    def registry(self) -> dict[str, ToolHandlerV6]:
        """获取工具注册表的副本"""
        return dict(self._registry)

    def read_langchain_tools(self) -> list[StructuredTool]:
        """获取只读工具的 LangChain 格式列表

        返回可用于 LangChain 的结构化工具列表，仅包含只读工具。

        Returns:
            结构化工具列表
        """
        return [
            StructuredTool.from_function(
                coroutine=self.search_person_tool,
                name="search_person",
                description="查询人员信息，只读工具。根据姓名、手机号或人员 ID 获取 personId。",
                args_schema=SearchPersonInputV6,
            ),
            StructuredTool.from_function(
                coroutine=self.search_device_tool,
                name="search_device",
                description="查询门禁设备信息，只读工具。根据设备名称、设备 ID 或设备 SN 获取 deviceSn。",
                args_schema=SearchDeviceInputV6,
            ),
            StructuredTool.from_function(
                coroutine=self.query_permission_tool,
                name="query_permission",
                description="查询门禁权限状态，只读工具。需要 personId 和 deviceSn。",
                args_schema=QueryPermissionInputV6,
            ),
        ]

    async def execute(
        self,
        tool_name: str,
        project_id: int,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        """执行指定的工具

        根据工具名称路由到对应的处理方法。

        Args:
            tool_name: 工具名称
            project_id: 项目ID
            slots: 槽位数据（包含工具所需的参数）

        Returns:
            工具执行结果
        """
        if tool_name not in TOOL_METADATA_V6 or tool_name not in self._registry:
            return {"status": "not_configured", "tool": tool_name, "message": "Unknown v6 tool."}
        return await self._registry[tool_name](project_id, slots)

    async def search_person_tool(
        self,
        project_id: int,
        personId: str | None = None,
        personName: str | None = None,
        telephone: str | None = None,
    ) -> dict[str, Any]:
        """人员查询工具（LangChain 格式）

        Args:
            project_id: 项目ID
            personId: 人员ID（可选）
            personName: 人员姓名（可选）
            telephone: 手机号（可选）

        Returns:
            人员查询结果
        """
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
        """设备查询工具（LangChain 格式）

        Args:
            project_id: 项目ID
            deviceSn: 设备SN（可选）
            deviceId: 设备ID（可选）
            deviceName: 设备名称（可选）

        Returns:
            设备查询结果
        """
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
        """权限查询工具（LangChain 格式）

        Args:
            project_id: 项目ID
            personId: 人员ID
            deviceSn: 设备SN

        Returns:
            权限查询结果
        """
        return await self.query_permission(project_id, {"personId": personId, "deviceSn": deviceSn})

    async def search_person(self, project_id: int, slots: dict[str, Any]) -> dict[str, Any]:
        """查询人员信息

        Args:
            project_id: 项目ID
            slots: 槽位数据

        Returns:
            人员查询结果
        """
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

    async def search_device(self, project_id: int, slots: dict[str, Any]) -> dict[str, Any]:
        """查询设备信息

        Args:
            project_id: 项目ID
            slots: 槽位数据

        Returns:
            设备查询结果
        """
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

    async def query_permission(self, project_id: int, slots: dict[str, Any]) -> dict[str, Any]:
        """查询权限状态

        Args:
            project_id: 项目ID
            slots: 槽位数据

        Returns:
            权限查询结果
        """
        return await self._client.query_permission(project_id, self._permission_payload(slots))

    async def extend_permission(self, project_id: int, slots: dict[str, Any]) -> dict[str, Any]:
        """续期权限

        Args:
            project_id: 项目ID
            slots: 槽位数据

        Returns:
            续期结果
        """
        return await self._client.extend_permission(project_id, self._permission_payload(slots, include_duration=True))

    async def enable_permission(self, project_id: int, slots: dict[str, Any]) -> dict[str, Any]:
        """启用权限

        Args:
            project_id: 项目ID
            slots: 槽位数据

        Returns:
            启用结果
        """
        return await self._client.enable_permission(project_id, self._permission_payload(slots))

    async def disable_permission(self, project_id: int, slots: dict[str, Any]) -> dict[str, Any]:
        """禁用权限

        Args:
            project_id: 项目ID
            slots: 槽位数据

        Returns:
            禁用结果
        """
        return await self._client.disable_permission(project_id, self._permission_payload(slots))

    async def grant_permission(self, project_id: int, slots: dict[str, Any]) -> dict[str, Any]:
        """授予权限

        Args:
            project_id: 项目ID
            slots: 槽位数据

        Returns:
            授予结果
        """
        return await self._client.grant_permission(project_id, self._permission_payload(slots, include_duration=True))

    async def revoke_permission(self, project_id: int, slots: dict[str, Any]) -> dict[str, Any]:
        """撤销权限

        Args:
            project_id: 项目ID
            slots: 槽位数据

        Returns:
            撤销结果
        """
        return await self._client.revoke_permission(project_id, self._permission_payload(slots))

    def _permission_payload(self, slots: dict[str, Any], *, include_duration: bool = False) -> dict[str, Any]:
        """构建权限操作请求载荷

        Args:
            slots: 槽位数据
            include_duration: 是否包含续期天数

        Returns:
            请求载荷
        """
        payload = {
            "personId": slots.get("personId"),
            "deviceSn": slots.get("deviceSn"),
        }
        if include_duration:
            payload["durationDays"] = slots.get("durationDays")
        return self._clean(payload)

    @staticmethod
    def _clean(payload: dict[str, Any]) -> dict[str, Any]:
        """清理载荷数据

        移除空值、空字符串、空列表和空字典。

        Args:
            payload: 原始载荷

        Returns:
            清理后的载荷
        """
        return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


def read_langchain_tools_v6(client: EstateAiClientV6 | None = None) -> list[StructuredTool]:
    """获取 V6 版本的只读工具列表

    Args:
        client: EstateAiClientV6 客户端实例

    Returns:
        LangChain 结构化工具列表
    """
    return RightToolsV6(client=client).read_langchain_tools()