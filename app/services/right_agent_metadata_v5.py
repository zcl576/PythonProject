"""权限代理元数据 V5 版本

本模块定义了权限代理 V5 版本中使用的工具元数据和相关常量。

主要内容：
- ToolMetadataV5: 工具元数据数据类，定义了工具的属性和配置
- TOOL_METADATA_V5: 工具元数据字典，包含所有支持的工具定义
- PLANNER_TARGET_TO_TOOL_V5: 意图到工具的映射表
- READ_TARGET_TOOLS_V5: 只读工具集合
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# 风险级别类型：read（只读）、write（写操作）
RiskLevel = Literal["read", "write"]

# 权限过期状态标识
PERMISSION_EXPIRED = "PERMISSION_EXPIRED"
# 续期权限工具名称
RENEW_PERMISSION = "renew_permission"


@dataclass(frozen=True)
class ToolMetadataV5:
    """工具元数据定义

    定义了权限代理中每个工具的属性和配置。

    Attributes:
        name: 工具名称
        description: 工具描述
        risk: 风险级别（read/write）
        required_slots: 必填槽位（所有槽位都必须填充）
        any_required_slots: 可选必填槽位（填充任意一个即可）
        slot_resolvers: 槽位解析器映射，指定如何获取缺失的槽位
        requires_confirm: 是否需要用户确认
        confirm_id: 确认操作的ID
    """
    name: str
    description: str
    risk: RiskLevel
    required_slots: tuple[str, ...] = ()
    any_required_slots: tuple[str, ...] = ()
    slot_resolvers: dict[str, str] = field(default_factory=dict)
    requires_confirm: bool = False
    confirm_id: str | None = None


# 工具元数据字典，包含所有支持的工具定义
TOOL_METADATA_V5: dict[str, ToolMetadataV5] = {
    "search_person": ToolMetadataV5(
        name="search_person",
        description="Search person information",
        risk="read",
        any_required_slots=("personId", "personName", "telephone"),
    ),
    "search_device": ToolMetadataV5(
        name="search_device",
        description="Search access device information",
        risk="read",
        any_required_slots=("deviceSn", "deviceId", "deviceName"),
    ),
    "query_permission": ToolMetadataV5(
        name="query_permission",
        description="Query access permission",
        risk="read",
        required_slots=("personId", "deviceSn"),
        slot_resolvers={
            "personId": "search_person",
            "deviceSn": "search_device",
        },
    ),
    "renew_permission": ToolMetadataV5(
        name="renew_permission",
        description="Renew access permission",
        risk="write",
        required_slots=("personId", "deviceSn"),
        requires_confirm=True,
        confirm_id=RENEW_PERMISSION,
    ),
}


# 意图到工具的映射表
# 将 LLM 识别的意图映射到具体的工具名称
PLANNER_TARGET_TO_TOOL_V5 = {
    "access_issue": "query_permission",      # 权限问题 -> 查询权限
    "permission_lookup": "query_permission", # 权限查询 -> 查询权限
    "query_permission": "query_permission",  # 查询权限 -> 查询权限
    "person_lookup": "search_person",        # 人员查询 -> 查询人员
    "search_person": "search_person",        # 查询人员 -> 查询人员
    "device_lookup": "search_device",        # 设备查询 -> 查询设备
    "search_device": "search_device",        # 查询设备 -> 查询设备
    # 续期权限会在状态机中被规范化为先查询权限
    "renew_permission": "renew_permission",  # 续期权限 -> 续期权限
}


# 只读工具集合
# 这些工具不会修改数据，可直接执行
READ_TARGET_TOOLS_V5 = {
    "search_person",
    "search_device",
    "query_permission",
}