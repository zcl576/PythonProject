"""权限工具执行器 V5 版本

本模块提供权限相关工具的执行能力，包括人员查询、设备查询、权限查询和权限续期。
"""
from typing import Any

from app.clients.estate_ai_client_v5 import EstateAiClientV5
from app.services.right_agent_metadata_v5 import TOOL_METADATA_V5


class RightToolsV5:
    """权限工具执行器 V5 版本

    封装了与权限相关的工具调用，通过 EstateAiClientV5 客户端执行实际操作。
    支持的工具包括：人员查询、设备查询、权限查询、权限续期。
    """

    def __init__(self, client: EstateAiClientV5 | None = None) -> None:
        """初始化工具执行器

        Args:
            client: EstateAiClientV5 客户端实例，用于调用底层 API
        """
        self._client = client or EstateAiClientV5()

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
        if tool_name not in TOOL_METADATA_V5:
            return {
                "status": "not_configured",
                "tool": tool_name,
                "message": "Unknown v5 tool.",
            }
        if tool_name == "search_person":
            return await self.search_person(project_id, slots)
        if tool_name == "search_device":
            return await self.search_device(project_id, slots)
        if tool_name == "query_permission":
            return await self.query_permission(project_id, slots)
        if tool_name == "renew_permission":
            return await self.renew_permission(project_id, slots)
        return {
            "status": "not_configured",
            "tool": tool_name,
            "message": "Tool executor is not configured.",
        }

    async def search_person(
        self,
        project_id: int,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        """查询人员信息

        Args:
            project_id: 项目ID
            slots: 槽位数据，包含 personId、personName、telephone

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

    async def search_device(
        self,
        project_id: int,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        """查询设备信息

        Args:
            project_id: 项目ID
            slots: 槽位数据，包含 deviceSn、deviceId、deviceName

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

    async def query_permission(
        self,
        project_id: int,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        """查询权限信息

        Args:
            project_id: 项目ID
            slots: 槽位数据，包含人员和设备信息

        Returns:
            权限查询结果
        """
        return await self._client.query_permission(project_id, self._clean(slots))

    async def renew_permission(
        self,
        project_id: int,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        """续期权限

        Args:
            project_id: 项目ID
            slots: 槽位数据，包含人员和设备信息

        Returns:
            权限续期结果
        """
        return await self._client.renew_permission(project_id, self._clean(slots))

    @staticmethod
    def _clean(payload: dict[str, Any]) -> dict[str, Any]:
        """清理请求载荷

        移除空值、空字符串、空列表和空字典。

        Args:
            payload: 原始载荷

        Returns:
            清理后的载荷
        """
        return {key: value for key, value in payload.items() if value not in (None, "", [], {})}