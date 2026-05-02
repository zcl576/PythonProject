from __future__ import annotations

from typing import Any

from app.services.right_agent_metadata_v4 import ToolMetadataV4


class PolicyDecisionV4(dict):
    @property
    def allowed(self) -> bool:
        return bool(self.get("allowed"))


class RightAgentPolicyCheckerV4:
    def check(
        self,
        *,
        tool: ToolMetadataV4,
        state: dict[str, Any],
        confirmed: bool = False,
    ) -> PolicyDecisionV4:
        if not tool.configured:
            return PolicyDecisionV4(
                allowed=False,
                reason="not_configured",
                status="not_configured",
                message=f"{tool.description}能力尚未配置后端接口，当前不能执行。",
            )

        for path, expected in tool.allowed_when.items():
            actual = self._get_path(state, path)
            if actual != expected:
                return PolicyDecisionV4(
                    allowed=False,
                    reason="condition_not_met",
                    status="policy_denied",
                    message="当前业务条件不满足，不能执行该操作。",
                    expected=expected,
                    actual=actual,
                )

        if tool.risk == "write" and tool.requires_confirm and not confirmed:
            return PolicyDecisionV4(
                allowed=False,
                reason="requires_confirm",
                status="need_confirm",
                message="该操作会修改业务数据，需要用户确认。",
            )

        return PolicyDecisionV4(allowed=True)

    @staticmethod
    def _get_path(state: dict[str, Any], path: str) -> Any:
        current: Any = state
        for part in path.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current
