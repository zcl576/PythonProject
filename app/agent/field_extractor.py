import re
from typing import Any

import httpx

from app.clients.llm_client import LlmClient
from app.schemas.diagnosis import DiagnosisAgentRequest


class DiagnosisFieldExtractor:
    def __init__(self, llm: LlmClient) -> None:
        self._llm = llm

    async def extract(self, request: DiagnosisAgentRequest, warnings: list[str]) -> dict[str, Any]:
        payload = {
            "personId": request.person_id,
            "telephone": request.telephone,
            "cardNo": request.card_no,
            "deviceId": request.device_id,
            "deviceName": request.device_name,
            "deviceSn": request.device_sn,
            "personName": request.person_name,
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
            "deviceName": None,
            "deviceSn": None,
            "personName": None,
        }

        phone_match = re.search(r"1\d{10}", question)
        if phone_match:
            payload["telephone"] = phone_match.group(0)

        device_match = re.search(r"deviceId[:：\s]*([A-Za-z0-9_-]+)", question, re.IGNORECASE)
        if device_match:
            payload["deviceId"] = device_match.group(1)

        device_sn_match = re.search(r"(?:deviceSn|设备SN|设备序列号|sn)[:：\s]*([A-Za-z0-9_-]+)", question, re.IGNORECASE)
        if device_sn_match:
            payload["deviceSn"] = device_sn_match.group(1)

        device_name_match = re.search(r"(?:deviceName|设备名称)[:：\s]*([\u4e00-\u9fa5A-Za-z0-9_-]+)", question, re.IGNORECASE)
        if device_name_match:
            payload["deviceName"] = device_name_match.group(1)

        card_match = re.search(r"card(?:No)?[:：\s]*([A-Za-z0-9_-]+)", question, re.IGNORECASE)
        if card_match:
            payload["cardNo"] = card_match.group(1)

        person_match = re.search(r"personId[:：\s]*([A-Za-z0-9_-]+)", question, re.IGNORECASE)
        if person_match:
            payload["personId"] = person_match.group(1)

        person_name_match = re.search(r"(?:personName|人员姓名|姓名)[:：\s]*([\u4e00-\u9fa5A-Za-z0-9_-]+)", question, re.IGNORECASE)
        if person_name_match:
            payload["personName"] = person_name_match.group(1)

        return payload
