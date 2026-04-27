from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import get_settings


class LlmClient:
    """LLM 客户端类
    
    负责与 LLM 服务交互，执行参数抽取和诊断解释等任务。
    """
    
    def __init__(self) -> None:
        """初始化 LLM 客户端
        
        从配置中读取 LLM 相关设置，包括启用状态、API 密钥、基础 URL 等。
        """
        settings = get_settings()
        self._enabled = settings.llm_enabled and bool(settings.llm_api_key)
        self._base_url = settings.llm_base_url.rstrip("/")
        self._api_key = settings.llm_api_key
        self._model = settings.llm_model
        self._timeout = settings.llm_timeout_seconds

    @property
    def enabled(self) -> bool:
        """获取 LLM 客户端是否启用
        
        Returns:
            bool: LLM 客户端是否启用
        """
        return self._enabled

    async def extract_diagnosis_fields(self, question: str) -> dict[str, Any] | None:
        """从问题描述中提取诊断参数
        
        使用 LLM 从问题描述中提取 personId、telephone、cardNo、deviceId 等诊断参数。
        
        Args:
            question: 问题描述文本
            
        Returns:
            dict[str, Any] | None: 提取的参数字典，或 None（如果提取失败）
        """
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
        """生成诊断解释
        
        使用 LLM 根据诊断结果生成清晰的诊断结论，包含最可能原因、关键证据和建议动作。
        
        Args:
            question: 原始问题描述
            normalized: 标准化后的参数
            result: 诊断结果
            
        Returns:
            str | None: 诊断解释文本，或 None（如果生成失败）
        """
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
        """与 LLM 服务进行对话
        
        向 LLM 服务发送请求并获取响应。
        
        Args:
            messages: 消息列表，包含系统消息和用户消息
            temperature: 生成文本的随机性，0 表示确定性输出
            
        Returns:
            str: LLM 生成的响应内容
        """
        payload = {"model": self._model, "messages": messages, "temperature": temperature}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            response = await client.post("/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"]

    def _parse_json(self, content: str) -> dict[str, Any] | None:
        """解析 JSON 内容
        
        解析 LLM 返回的 JSON 内容，处理代码块标记，并提取所需字段。
        
        Args:
            content: LLM 返回的内容
            
        Returns:
            dict[str, Any] | None: 解析后的参数字典，或 None（如果解析失败）
        """
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