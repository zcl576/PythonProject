import re
from typing import Any

import httpx

from app.clients.estate_ai_client import EstateAiClient
from app.clients.llm_client import LlmClient
from app.schemas.diagnosis import DiagnosisAgentRequest, DiagnosisAgentResponse, SuggestedAction


class AccessDiagnosisAgentService:
    def __init__(self) -> None:
        self._client = EstateAiClient()
        self._llm = LlmClient()

    async def diagnose(self, request: DiagnosisAgentRequest) -> DiagnosisAgentResponse:
        warnings: list[str] = []
        normalized = await self._normalize_request(request, warnings)
        result = await self._client.get_result(request.project_id, normalized)
        context = result.get("context") or {}
        diagnosis = result.get("diagnosis") or {}
        answer = await self._build_answer(request.question, normalized, result, warnings)
        llm_used = self._llm.enabled and not any(item.startswith("LLM") for item in warnings)

        return DiagnosisAgentResponse(
            normalized_request=normalized,
            answer=answer,
            summary=diagnosis.get("summary", ""),
            main_cause=diagnosis.get("mainCause", "UNKNOWN"),
            main_cause_name=diagnosis.get("mainCauseName", "未知原因"),
            confidence=diagnosis.get("confidence"),
            agent_mode="llm" if llm_used else "rule",
            llm_used=llm_used,
            warnings=warnings,
            evidences=diagnosis.get("evidences") or [],
            suggested_actions=[SuggestedAction.model_validate(item) for item in diagnosis.get("suggestedActions") or []],
            context=context,
            diagnosis=diagnosis,
        )

    async def _normalize_request(self, request: DiagnosisAgentRequest, warnings: list[str]) -> dict[str, Any]:
        payload = {
            "personId": request.person_id,
            "telephone": request.telephone,
            "cardNo": request.card_no,
            "deviceId": request.device_id,
        }
        if any(payload.values()):
            return payload

        question = (request.question or "").strip()
        if not question:
            return payload

        if self._llm.enabled:
            try:
                extracted = await self._llm.extract_diagnosis_fields(question)
                if extracted and any(extracted.values()):
                    return extracted
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                warnings.append(f"LLM参数抽取失败，已切换规则抽取: {exc}")

        return self._rule_extract(question)

    def _rule_extract(self, question: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "personId": None,
            "telephone": None,
            "cardNo": None,
            "deviceId": None,
        }

        phone_match = re.search(r"1\d{10}", question)
        if phone_match:
            payload["telephone"] = phone_match.group(0)

        device_match = re.search(r"deviceId[:：\s]*([A-Za-z0-9_-]+)", question, re.IGNORECASE)
        if device_match:
            payload["deviceId"] = device_match.group(1)

        card_match = re.search(r"card(?:No)?[:：\s]*([A-Za-z0-9_-]+)", question, re.IGNORECASE)
        if card_match:
            payload["cardNo"] = card_match.group(1)

        person_match = re.search(r"personId[:：\s]*([A-Za-z0-9_-]+)", question, re.IGNORECASE)
        if person_match:
            payload["personId"] = person_match.group(1)

        return payload

    async def _build_answer(
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
