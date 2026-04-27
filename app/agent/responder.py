from typing import Any

import httpx

from app.clients.llm_client import LlmClient


class DiagnosisResponder:
    def __init__(self, llm: LlmClient) -> None:
        self._llm = llm

    async def build_answer(
        self,
        question: str | None,
        normalized: dict[str, Any],
        result: dict[str, Any],
        warnings: list[str],
    ) -> str:
        if self._llm.enabled:
            try:
                answer = await self._llm.explain_diagnosis(question, normalized, result)
                if answer:
                    return answer.strip()
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                warnings.append(f"LLM诊断解释失败，已切换规则解释: {exc}")
        return self._rule_answer(normalized, result.get("diagnosis") or {})

    def _rule_answer(self, normalized: dict[str, Any], diagnosis: dict[str, Any]) -> str:
        main_cause_name = diagnosis.get("mainCauseName", "未知原因")
        summary = diagnosis.get("summary", "")
        evidences = diagnosis.get("evidences") or []
        actions = diagnosis.get("suggestedActions") or []

        segments = [f"诊断结论：{main_cause_name}。"]
        if summary:
            segments.append(summary)
        if any(normalized.values()):
            formatted = ", ".join(f"{key}={value}" for key, value in normalized.items() if value)
            segments.append(f"本次诊断输入：{formatted}。")
        if evidences:
            segments.append("关键证据：" + "；".join(evidences[:3]) + "。")
        if actions:
            action_names = "、".join(item.get("actionName", item.get("action", "")) for item in actions[:3])
            segments.append(f"建议优先处理：{action_names}。")
        return "".join(segments)
