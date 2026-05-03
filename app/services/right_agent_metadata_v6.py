from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

RiskLevel = Literal["read", "write"]


class PermissionStatus(StrEnum):
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    DISABLED = "DISABLED"
    REVOKED = "REVOKED"
    NOT_FOUND = "NOT_FOUND"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ToolMetadataV6:
    name: str
    description: str
    risk: RiskLevel
    required_slots: tuple[str, ...] = ()
    any_required_slots: tuple[str, ...] = ()
    slot_resolvers: dict[str, str] = field(default_factory=dict)
    requires_confirm: bool = False
    allowed_statuses: tuple[PermissionStatus, ...] = ()
    confirm_id: str | None = None


READ_TOOLS_V6 = {"search_person", "search_device", "query_permission"}
WRITE_TOOLS_V6 = {
    "extend_permission",
    "enable_permission",
    "disable_permission",
    "grant_permission",
    "revoke_permission",
}


TOOL_METADATA_V6: dict[str, ToolMetadataV6] = {
    "search_person": ToolMetadataV6(
        name="search_person",
        description="Search person information",
        risk="read",
        any_required_slots=("personId", "personName", "telephone"),
    ),
    "search_device": ToolMetadataV6(
        name="search_device",
        description="Search access device information",
        risk="read",
        any_required_slots=("deviceSn", "deviceId", "deviceName"),
    ),
    "query_permission": ToolMetadataV6(
        name="query_permission",
        description="Query access permission status",
        risk="read",
        required_slots=("personId", "deviceSn"),
        slot_resolvers={
            "personId": "search_person",
            "deviceSn": "search_device",
        },
    ),
    "extend_permission": ToolMetadataV6(
        name="extend_permission",
        description="Extend access permission",
        risk="write",
        required_slots=("personId", "deviceSn"),
        requires_confirm=True,
        allowed_statuses=(PermissionStatus.EXPIRED, PermissionStatus.ACTIVE),
        confirm_id="extend_permission",
    ),
    "enable_permission": ToolMetadataV6(
        name="enable_permission",
        description="Enable disabled access permission",
        risk="write",
        required_slots=("personId", "deviceSn"),
        requires_confirm=True,
        allowed_statuses=(PermissionStatus.DISABLED,),
        confirm_id="enable_permission",
    ),
    "disable_permission": ToolMetadataV6(
        name="disable_permission",
        description="Disable active access permission",
        risk="write",
        required_slots=("personId", "deviceSn"),
        requires_confirm=True,
        allowed_statuses=(PermissionStatus.ACTIVE,),
        confirm_id="disable_permission",
    ),
    "grant_permission": ToolMetadataV6(
        name="grant_permission",
        description="Grant new access permission",
        risk="write",
        required_slots=("personId", "deviceSn"),
        requires_confirm=True,
        allowed_statuses=(PermissionStatus.NOT_FOUND, PermissionStatus.REVOKED),
        confirm_id="grant_permission",
    ),
    "revoke_permission": ToolMetadataV6(
        name="revoke_permission",
        description="Revoke access permission",
        risk="write",
        required_slots=("personId", "deviceSn"),
        requires_confirm=True,
        allowed_statuses=(
            PermissionStatus.ACTIVE,
            PermissionStatus.EXPIRED,
            PermissionStatus.DISABLED,
        ),
        confirm_id="revoke_permission",
    ),
}


PLANNER_TARGET_TO_TOOL_V6 = {
    "access_issue": "query_permission",
    "permission_lookup": "query_permission",
    "query_permission": "query_permission",
    "person_lookup": "search_person",
    "search_person": "search_person",
    "device_lookup": "search_device",
    "search_device": "search_device",
    "extend_permission": "extend_permission",
    "renew_permission": "extend_permission",
    "enable_permission": "enable_permission",
    "disable_permission": "disable_permission",
    "grant_permission": "grant_permission",
    "revoke_permission": "revoke_permission",
}
