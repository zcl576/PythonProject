from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import get_settings


class LlmClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._enabled = settings.llm_enabled and bool(settings.llm_api_key)
        self._base_url = settings.llm_base_url.rstrip("/")
        self._api_key = settings.llm_api_key
        self._model = settings.llm_model
        self._timeout = settings.llm_timeout_seconds

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def extract_diagnosis_fields(self, question: str) -> dict[str, Any] | None:
        if not self._enabled or not question.strip():
            return None
        messages = [
            {
                "role": "system",
                "content": (
                    "你是门禁异常诊断参数抽取器。只输出 JSON，不要输出解释。"
                    "JSON 字段只能包含 personId、telephone、cardNo、deviceId。"
                    "无法确定的字段输出 null。"
                ),
            },
            {"role": "user", "content": question},
        ]
        content = await self._chat(messages, temperature=0)
        return self._parse_json(content)

    async def explain_diagnosis(self, question: str | None, normalized: dict[str, Any], result: dict[str, Any]) -> str | None:
        if not self._enabled:
            return None
        diagnosis = result.get("diagnosis") or {}
        compact_payload = {
            "question": question,
            "normalizedRequest": normalized,
            "diagnosis": diagnosis,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "你是物业门禁异常诊断助手。根据诊断 JSON 给物业人员一句清晰结论，"
                    "包含最可能原因、关键证据和建议动作。不要编造 JSON 中没有的信息。"
                    "不要承诺已经执行开门、补发或工单动作。"
                ),
            },
            {"role": "user", "content": json.dumps(compact_payload, ensure_ascii=False)},
        ]
        return await self._chat(messages, temperature=0.2)

    async def _chat(self, messages: list[dict[str, str]], temperature: float) -> str:
        payload = {"model": self._model, "messages": messages, "temperature": temperature}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            response = await client.post("/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"]

    def _parse_json(self, content: str) -> dict[str, Any] | None:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return {
            "personId": parsed.get("personId"),
            "telephone": parsed.get("telephone"),
            "cardNo": parsed.get("cardNo"),
            "deviceId": parsed.get("deviceId"),
        }

