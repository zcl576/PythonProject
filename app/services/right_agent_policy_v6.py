from __future__ import annotations

from typing import Any

from app.services.right_agent_metadata_v6 import PermissionStatus, ToolMetadataV6


class PolicyDecisionV6(dict):
    @property
    def allowed(self) -> bool:
        return bool(self.get("allowed"))


class RightAgentPolicyCheckerV6:
    def check_write(
        self,
        *,
        tool: ToolMetadataV6,
        permission_result: dict[str, Any],
        confirmed: bool = False,
    ) -> PolicyDecisionV6:
        status = self.permission_status(permission_result)
        if tool.risk != "write":
            return PolicyDecisionV6(allowed=True, permission_status=status.value)
        if tool.allowed_statuses and status not in tool.allowed_statuses:
            return PolicyDecisionV6(
                allowed=False,
                reason="status_not_allowed",
                status="policy_denied",
                permission_status=status.value,
                message=f"{tool.name} is not allowed when permission status is {status.value}.",
            )
        if tool.requires_confirm and not confirmed:
            return PolicyDecisionV6(
                allowed=False,
                reason="requires_confirm",
                status="need_confirm",
                permission_status=status.value,
                message=f"{tool.name} requires user confirmation.",
            )
        return PolicyDecisionV6(allowed=True, permission_status=status.value)

    @classmethod
    def permission_status(cls, payload: dict[str, Any]) -> PermissionStatus:
        values = cls._status_values(payload)
        for value in values:
            normalized = str(value).upper()
            aliases = {
                "PERMISSION_EXPIRED": PermissionStatus.EXPIRED,
                "EXPIRED": PermissionStatus.EXPIRED,
                "PERMISSION_DISABLED": PermissionStatus.DISABLED,
                "DISABLED": PermissionStatus.DISABLED,
                "FORBIDDEN": PermissionStatus.DISABLED,
                "ACTIVE": PermissionStatus.ACTIVE,
                "ENABLED": PermissionStatus.ACTIVE,
                "NORMAL": PermissionStatus.ACTIVE,
                "REVOKED": PermissionStatus.REVOKED,
                "DELETED": PermissionStatus.REVOKED,
                "NOT_FOUND": PermissionStatus.NOT_FOUND,
                "NO_PERMISSION": PermissionStatus.NOT_FOUND,
                "MISSING": PermissionStatus.NOT_FOUND,
            }
            if normalized in aliases:
                return aliases[normalized]
        return PermissionStatus.UNKNOWN

    @classmethod
    def _status_values(cls, payload: Any) -> list[Any]:
        if not isinstance(payload, dict):
            return []
        values: list[Any] = []
        for key in (
            "status",
            "permissionStatus",
            "permission_status",
            "mainCause",
            "main_cause",
            "code",
        ):
            if payload.get(key):
                values.append(payload[key])
        for child_key in ("permission", "diagnosis", "data", "result"):
            child = payload.get(child_key)
            if isinstance(child, dict):
                values.extend(cls._status_values(child))
        return values
