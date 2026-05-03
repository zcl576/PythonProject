from __future__ import annotations

import json
import os
from typing import Any

from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr

from app.config import get_settings
from loguru import logger as log


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
                "content": self._get_prompt(),
            },
            {"role": "user", "content": question},
        ]
        content = await self._json_chat(messages, temperature=0)
        return self._parse_json(content)

    async def understand_right_agent_v4(
        self,
        *,
        question: str,
        intents: tuple[str, ...],
        slot_names: tuple[str, ...],
    ) -> dict[str, Any] | None:
        """理解权限代理 V4 版本的用户意图

        使用 LLM 从用户问题中提取意图和槽位信息。

        Args:
            question: 用户问题文本
            intents: 允许的意图列表
            slot_names: 允许的槽位名称列表

        Returns:
            dict[str, Any] | None: 包含意图和槽位的字典，或 None（如果解析失败）
        """
        if not self._enabled or not question.strip():
            return None
        messages = [
            {
                "role": "system",
                "content": self._right_agent_v4_prompt(intents, slot_names),
            },
            {"role": "user", "content": question},
        ]
        content = await self._json_chat(messages, temperature=0)
        return self._parse_right_agent_v4_json(content, intents, slot_names)

    async def understand_right_agent_v5(
        self,
        *,
        question: str,
        target_tools: tuple[str, ...],
        slot_names: tuple[str, ...],
    ) -> dict[str, Any] | None:
        """理解权限代理 V5 版本的用户意图

        使用 LLM 从用户问题中提取目标工具和槽位信息，生成执行计划。

        Args:
            question: 用户问题文本
            target_tools: 允许的目标工具列表
            slot_names: 允许的槽位名称列表

        Returns:
            dict[str, Any] | None: 包含意图、目标工具和槽位的字典，或 None（如果解析失败）
        """
        if not self._enabled or not question.strip():
            return None
        messages = [
            {
                "role": "system",
                "content": self._right_agent_v5_planner_prompt(target_tools, slot_names),
            },
            {"role": "user", "content": question},
        ]
        content = await self._json_chat(messages, temperature=0)
        return self._parse_right_agent_v5_plan(content, target_tools, slot_names)

    async def answer_right_agent_v5(
        self,
        *,
        question: str | None,
        slots: dict[str, Any],
        tool_history: list[dict[str, Any]],
        permission_result: dict[str, Any] | None = None,
        renew_result: dict[str, Any] | None = None,
    ) -> str | None:
        """生成权限代理 V5 版本的最终回答

        使用 LLM 根据工具执行结果生成面向用户的中文回答。

        Args:
            question: 用户原始问题
            slots: 槽位数据
            tool_history: 工具调用历史
            permission_result: 权限查询结果
            renew_result: 续期操作结果

        Returns:
            str | None: 生成的回答文本，或 None（如果生成失败）
        """
        if not self._enabled:
            return None
        payload = {
            "question": question,
            "slots": self._without_internal_fields(slots),
            "tool_history": self._without_internal_fields(tool_history),
            "permission_result": self._without_internal_fields(permission_result or {}),
            "renew_result": self._without_internal_fields(renew_result or {}),
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an access-control operations assistant. "
                    "Write one concise user-facing answer in Chinese. "
                    "Use only the provided tool results. Do not expose project IDs, "
                    "session IDs, trace IDs, raw JSON, or internal orchestration fields. "
                    "Do not claim a write action succeeded unless renew_result says it did."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        return await self._text_chat(messages, temperature=0.2)

    async def understand_right_agent_v6(
        self,
        *,
        question: str,
        target_tools: tuple[str, ...],
        slot_names: tuple[str, ...],
    ) -> dict[str, Any] | None:
        if not self._enabled or not question.strip():
            return None
        messages = [
            {
                "role": "system",
                "content": self._right_agent_v6_planner_prompt(target_tools, slot_names),
            },
            {"role": "user", "content": question},
        ]
        content = await self._json_chat(messages, temperature=0)
        return self._parse_right_agent_v5_plan(content, target_tools, slot_names)

    async def answer_right_agent_v6(
        self,
        *,
        question: str | None,
        slots: dict[str, Any],
        tool_history: list[dict[str, Any]],
        permission_status: str | None = None,
        permission_result: dict[str, Any] | None = None,
        write_result: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
    ) -> str | None:
        if not self._enabled:
            return None
        payload = {
            "question": question,
            "slots": self._without_internal_fields(slots),
            "permission_status": permission_status,
            "permission_result": self._without_internal_fields(permission_result or {}),
            "write_result": self._without_internal_fields(write_result or {}),
            "policy": self._without_internal_fields(policy or {}),
            "tool_history": self._without_internal_fields(tool_history),
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "你是门禁权限运营助手。请根据给定工具结果生成一句面向用户的中文回答。"
                    "只能使用输入中提供的信息，不要编造权限状态或操作结果。"
                    "不要暴露 project_id、session_id、trace_id、原始 JSON 或内部编排字段。"
                    "只有 write_result 明确表示成功时，才能说已经完成写操作。"
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        return await self._text_chat(messages, temperature=0.2)

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
        diagnosis = self._without_internal_fields(result.get("diagnosis") or {})
        compact_payload = {
            "问题": question,
            "标准化请求": normalized,
            "诊断结果": diagnosis,
        }
        messages = [

            {
                "role": "system",
                "content": (
                    "你是物业门禁异常诊断助手。根据诊断 JSON 给物业人员一句清晰结论，"
                    "包含最可能原因、关键证据和建议动作。不要编造 JSON 中没有的信息。"
                    "不要承诺已经执行开门、补发或工单动作。"
                    "不要展示项目ID、projectId、project_id、traceId、sessionId 等内部技术字段。"
                ),
            },
            {"role": "user", "content": json.dumps(compact_payload, ensure_ascii=False)},
        ]
        return await self._text_chat(messages, temperature=0.2)

    async def _json_chat(self, messages: list[dict[str, str]], temperature: float) -> str:
        """与 LLM 服务进行对话
        
        向 LLM 服务发送请求并获取响应。
        
        Args:
            messages: 消息列表，包含系统消息和用户消息
            temperature: 生成文本的随机性，0 表示确定性输出
            
        Returns:
            str: LLM 生成的响应内容
        """
        model = ChatDeepSeek(
            model=self._model,
            temperature=temperature,
            api_key=SecretStr(self._api_key or ""),
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=2,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        response = await model.ainvoke(messages)
        return str(response.content)

    async def _text_chat(self, messages: list[dict[str, str]], temperature: float) -> str:
        """与 LLM 服务进行文本对话

        向 LLM 服务发送请求并获取文本响应，不要求 JSON 格式。

        Args:
            messages: 消息列表，包含系统消息和用户消息
            temperature: 生成文本的随机性，0 表示确定性输出

        Returns:
            str: LLM 生成的响应内容
        """
        model = ChatDeepSeek(
            model=self._model,
            temperature=temperature,
            api_key=SecretStr(self._api_key or ""),
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=2,
        )
        response = await model.ainvoke(messages)
        return str(response.content)

    def _without_internal_fields(self, payload: Any) -> Any:
        """移除内部字段

        递归地从字典或列表中移除项目ID、会话ID、追踪ID等内部技术字段。

        Args:
            payload: 要处理的数据（字典、列表或其他类型）

        Returns:
            Any: 移除内部字段后的数据
        """
        if isinstance(payload, dict):
            return {
                key: self._without_internal_fields(value)
                for key, value in payload.items()
                if key not in {"projectId", "project_id", "traceId", "trace_id", "sessionId", "session_id"}
            }
        if isinstance(payload, list):
            return [self._without_internal_fields(item) for item in payload]
        return payload

    def _parse_json(self, content: str) -> dict[str, Any] | None:
        """解析 JSON 内容
        
        解析 LLM 返回的 JSON 内容，处理代码块标记，并提取所需字段。
        
        Args:
            content: LLM 返回的内容
            
        Returns:
            dict[str, Any] | None: 解析后的参数字典，或 None（如果解析失败）
        """
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            log.error("json转换异常")
            return None
        if not isinstance(parsed, dict):
            return None
        return {
            "personId": parsed.get("personId") or parsed.get("person_id"),
            "telephone": parsed.get("telephone"),
            "cardNo": parsed.get("cardNo") or parsed.get("card_no"),
            "deviceId": parsed.get("deviceId") or parsed.get("device_id"),
            "deviceName": parsed.get("deviceName") or parsed.get("device_name"),
            "deviceSn": parsed.get("deviceSn") or parsed.get("device_sn"),
            "personName": parsed.get("personName") or parsed.get("person_name"),
        }

    def _parse_right_agent_v4_json(
        self,
        content: str,
        intents: tuple[str, ...],
        slot_names: tuple[str, ...],
    ) -> dict[str, Any] | None:
        """解析权限代理 V4 版本的 LLM 响应

        解析 LLM 返回的 JSON 内容，验证意图和槽位，进行字段别名转换。

        Args:
            content: LLM 返回的内容
            intents: 允许的意图列表
            slot_names: 允许的槽位名称列表

        Returns:
            dict[str, Any] | None: 包含意图和槽位的字典，或 None（如果解析失败）
        """
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            log.error("json转换异常")
            return None
        if not isinstance(parsed, dict):
            return None

        intent = parsed.get("intent")
        if intent not in intents:
            intent = "unsupported"

        raw_slots = parsed.get("slots") or {}
        if not isinstance(raw_slots, dict):
            raw_slots = {}
        aliases = {
            "person_id": "personId",
            "card_no": "cardNo",
            "device_id": "deviceId",
            "device_name": "deviceName",
            "device_sn": "deviceSn",
            "person_name": "personName",
        }
        allowed = set(slot_names)
        slots: dict[str, Any] = {}
        for key, value in raw_slots.items():
            normalized_key = aliases.get(key, key)
            if normalized_key not in allowed or value in (None, ""):
                continue
            slots[normalized_key] = value
        return {"intent": intent, "slots": slots}

    @staticmethod
    def _right_agent_v4_prompt(
        intents: tuple[str, ...],
        slot_names: tuple[str, ...],
    ) -> str:
        """生成权限代理 V4 版本的系统提示词

        构建用于意图和槽位提取的系统提示词。

        Args:
            intents: 允许的意图列表
            slot_names: 允许的槽位名称列表

        Returns:
            str: 系统提示词内容
        """
        return (
            "You are an access-control assistant intent and slot extractor. "
            "Return only valid JSON, with no markdown and no explanation. "
            f"Allowed intent values: {', '.join(intents)}. "
            "Use unsupported when the user request does not match an allowed intent. "
            f"Allowed slot keys: {', '.join(slot_names)}. "
            "Extract only values explicitly present or clearly implied by the user. "
            "Do not invent IDs. Keep original casing for IDs, card numbers, and SNs. "
            'Output shape: {"intent":"access_issue","slots":{"personName":"张三","deviceName":"三号门"}}.'
        )

    def _parse_right_agent_v5_plan(
        self,
        content: str,
        target_tools: tuple[str, ...],
        slot_names: tuple[str, ...],
    ) -> dict[str, Any] | None:
        """解析权限代理 V5 版本的 LLM 响应

        解析 LLM 返回的 JSON 内容，验证目标工具和槽位，进行字段别名转换。

        Args:
            content: LLM 返回的内容
            target_tools: 允许的目标工具列表
            slot_names: 允许的槽位名称列表

        Returns:
            dict[str, Any] | None: 包含意图、目标工具和槽位的字典，或 None（如果解析失败）
        """
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            log.error("json转换异常")
            return None
        if not isinstance(parsed, dict):
            return None

        target_tool = parsed.get("target_tool") or parsed.get("targetTool")
        if target_tool not in target_tools:
            target_tool = None

        raw_slots = parsed.get("slots") or {}
        if not isinstance(raw_slots, dict):
            raw_slots = {}
        aliases = {
            "person_id": "personId",
            "person_name": "personName",
            "card_no": "cardNo",
            "device_id": "deviceId",
            "device_name": "deviceName",
            "device_sn": "deviceSn",
        }
        allowed = set(slot_names)
        slots: dict[str, Any] = {}
        for key, value in raw_slots.items():
            normalized_key = aliases.get(key, key)
            if normalized_key not in allowed or value in (None, ""):
                continue
            slots[normalized_key] = value
        return {
            "intent": parsed.get("intent"),
            "target_tool": target_tool,
            "slots": slots,
        }

    @staticmethod
    def _right_agent_v5_planner_prompt(
        target_tools: tuple[str, ...],
        slot_names: tuple[str, ...],
    ) -> str:
        """生成权限代理 V5 版本的规划器提示词

        构建用于生成执行计划的系统提示词，指导 LLM 选择目标工具和提取槽位。

        Args:
            target_tools: 允许的目标工具列表
            slot_names: 允许的槽位名称列表

        Returns:
            str: 系统提示词内容
        """
        return (
            "你是一个门禁业务 Agent 的规划器。"
            "你的任务是根据用户问题识别业务意图、选择最终业务目标工具，并抽取可确认的槽位。"
            "只能返回合法 JSON，不要输出 Markdown、解释、注释或代码块。"
            f"target_tool 只能从以下值中选择：{', '.join(target_tools)}。"
            f"slots 只能包含以下字段：{', '.join(slot_names)}。"
            "请选择最终业务目标工具，不要把补槽用的中间查询步骤写成 target_tool。"
            "如果用户在问门打不开、刷不开、进不去、有没有权限、为什么不能开门等问题，"
            "target_tool 应选择 query_permission。"
            "即使用户直接要求续期，也不要直接选择 renew_permission；"
            "续期必须先查询权限，并且只有权限确认为过期后，经过用户确认才能执行。"
            "只抽取用户明确提供或上下文中可以确定的信息，不要猜测 personId、deviceSn、deviceId、cardNo 等编号。"
            "编号、SN、卡号、ID 必须保留原始大小写。"
            '输出格式示例：{"intent":"access_issue","target_tool":"query_permission",'
            '"slots":{"personName":"张三","deviceSn":"D001"}}。'
        )

    @staticmethod
    def _right_agent_v6_planner_prompt(
        target_tools: tuple[str, ...],
        slot_names: tuple[str, ...],
    ) -> str:
        return (
            "你是门禁权限 Agent 的规划器。"
            "只能返回合法 JSON，不要输出 Markdown、解释、注释或代码块。"
            f"target_tool 只能从以下值中选择：{', '.join(target_tools)}。"
            f"slots 只能包含以下字段：{', '.join(slot_names)}。"
            "请选择用户最终想完成的业务目标，不要输出补槽工具链。"
            "查询用户信息选择 search_person；查询设备信息选择 search_device；"
            "查询权限、门打不开、刷不开、为什么不能进门选择 query_permission；"
            "权限过期需要延期选择 extend_permission；权限被禁用需要开启选择 enable_permission；"
            "禁用权限选择 disable_permission；开通新权限选择 grant_permission；回收权限选择 revoke_permission。"
            "写操作只是目标意图，系统会先查询权限、校验策略并要求用户确认后才执行。"
            "只抽取用户明确提供或上下文可以确定的信息，不要猜测 personId、deviceSn、deviceId、cardNo。"
            "编号、SN、卡号、ID 必须保留原始大小写。"
            '输出格式示例：{"intent":"extend_permission","target_tool":"extend_permission",'
            '"slots":{"personName":"张三","deviceName":"三号门","durationDays":30}}。'
        )

    def _get_prompt(self) -> str:
        """获取 LLM 提示

        从文件中读取 LLM 提示模板。

        Returns:
            str: LLM 提示模板内容
        """
        with open(os.path.join(get_settings().base_dir, "prompts", "field_extraction_system_v1.md"), "r", encoding="utf-8") as f:
            return f.read()
