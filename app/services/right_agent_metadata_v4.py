from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RiskLevel = Literal["read", "write"]

PERMISSION_EXPIRED = "PERMISSION_EXPIRED"
RENEW_PERMISSION = "renew_permission"


@dataclass(frozen=True)
class ToolMetadataV4:
    name: str
    description: str
    risk: RiskLevel
    configured: bool
    required_slots: tuple[str, ...] = ()
    any_required_slots: tuple[str, ...] = ()
    slot_resolvers: dict[str, str] = field(default_factory=dict)
    requires_confirm: bool = False
    allowed_when: dict[str, Any] = field(default_factory=dict)
    confirm_id: str | None = None


SUPPORTED_CAPABILITIES = [
    "查询人员信息",
    "查询设备信息",
    "查询卡号/凭证",
    "查询门禁权限",
    "查询刷卡记录",
    "诊断开不了门原因",
    "权限开通",
    "权限续期",
]


TOOL_METADATA: dict[str, ToolMetadataV4] = {
    "search_person": ToolMetadataV4(
        name="search_person",
        description="通过姓名或手机号查询人员信息",
        risk="read",
        configured=True,
        any_required_slots=("personName", "telephone", "personId"),
    ),
    "search_device": ToolMetadataV4(
        name="search_device",
        description="通过设备名称、设备 ID 或设备 SN 查询设备信息",
        risk="read",
        configured=False,
        any_required_slots=("deviceName", "deviceId", "deviceSn"),
    ),
    "query_permission": ToolMetadataV4(
        name="query_permission",
        description="查询人员对设备的门禁权限",
        risk="read",
        configured=True,
        required_slots=("personId",),
        slot_resolvers={"personId": "search_person"},
    ),
    "query_card": ToolMetadataV4(
        name="query_card",
        description="查询人员卡号",
        risk="read",
        configured=False,
        required_slots=("personId",),
        slot_resolvers={"personId": "search_person"},
    ),
    "query_credential": ToolMetadataV4(
        name="query_credential",
        description="查询人员凭证",
        risk="read",
        configured=False,
        required_slots=("personId",),
        slot_resolvers={"personId": "search_person"},
    ),
    "query_access_record": ToolMetadataV4(
        name="query_access_record",
        description="查询刷卡或通行记录",
        risk="read",
        configured=False,
        required_slots=("personId",),
        slot_resolvers={"personId": "search_person"},
    ),
    "diagnose_access_issue": ToolMetadataV4(
        name="diagnose_access_issue",
        description="诊断门禁打不开、刷不开、进不去等异常原因",
        risk="read",
        configured=True,
        any_required_slots=(
            "personId",
            "telephone",
            "cardNo",
            "personName",
            "deviceId",
            "deviceName",
            "deviceSn",
        ),
    ),
    "open_permission": ToolMetadataV4(
        name="open_permission",
        description="开通门禁权限",
        risk="write",
        configured=False,
        required_slots=("personId", "deviceId"),
        slot_resolvers={"personId": "search_person", "deviceId": "search_device"},
        requires_confirm=True,
        confirm_id="open_permission",
    ),
    "renew_permission": ToolMetadataV4(
        name="renew_permission",
        description="续期门禁权限",
        risk="write",
        configured=True,
        slot_resolvers={"personId": "search_person"},
        requires_confirm=True,
        allowed_when={"data.diagnosis.mainCause": PERMISSION_EXPIRED},
        confirm_id=RENEW_PERMISSION,
    ),
    "disable_card": ToolMetadataV4(
        name="disable_card",
        description="禁用或挂失门禁卡",
        risk="write",
        configured=False,
        required_slots=("cardNo",),
        requires_confirm=True,
        confirm_id="disable_card",
    ),
}


INTENT_TO_TOOL = {
    "person_lookup": "search_person",
    "device_lookup": "search_device",
    "card_lookup": "query_card",
    "credential_lookup": "query_credential",
    "permission_lookup": "query_permission",
    "access_record_lookup": "query_access_record",
    "access_issue": "query_permission",
    "open_permission": "open_permission",
    "renew_permission": "renew_permission",
    "disable_card": "disable_card",
}
