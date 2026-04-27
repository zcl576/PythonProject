import re
import uuid
from typing import Any

import httpx

from app.clients.estate_ai_client import EstateAiClient
from app.clients.llm_client import LlmClient
from app.schemas.diagnosis import DiagnosisAgentRequest, DiagnosisAgentResponse, SuggestedAction


class AccessDiagnosisAgentService:
    """门禁诊断代理服务类
    
    负责处理门禁异常的诊断请求，包括请求标准化、会话管理、调用诊断工具和构建响应。
    """
    
    def __init__(self) -> None:
        """初始化门禁诊断代理服务
        
        初始化 EstateAiClient 和 LlmClient 客户端，以及会话存储。
        """
        self._client = EstateAiClient()
        self._llm = LlmClient()
        self._sessions: dict[str, dict[str, Any]] = {}

    async def diagnose(self, request: DiagnosisAgentRequest) -> DiagnosisAgentResponse:
        """执行门禁异常诊断
        
        处理诊断请求，包括标准化输入、合并会话上下文、调用诊断工具并构建响应。
        
        Args:
            request: 诊断请求对象，包含会话ID、问题描述和各种标识信息
            
        Returns:
            DiagnosisAgentResponse: 诊断响应对象，包含诊断结果、状态、建议操作等信息
        """
        trace_id = uuid.uuid4().hex
        session_id = request.session_id or trace_id
        warnings: list[str] = []
        steps: list[str] = ["识别问题类型：门禁异常诊断"]
        normalized = await self._normalize_request(request, warnings)
        normalized = self._merge_session_context(session_id, normalized)
        self._sessions[session_id] = normalized
        steps.append("抽取并合并诊断参数")

        if not any(normalized.values()):
            follow_up_question = "请提供手机号、卡号、人员ID或设备ID中的任意一个，我再继续诊断。"
            steps.append("诊断信息不足，等待补充关键标识")
            return DiagnosisAgentResponse(
                trace_id=trace_id,
                session_id=session_id,
                status="need_more_info",
                normalized_request=normalized,
                answer=follow_up_question,
                follow_up_question=follow_up_question,
                needs_input=["person_id", "telephone", "card_no", "device_id"],
                steps=steps,
                available_actions=[],
                summary="缺少可定位人员或设备的诊断信息",
                main_cause="CONTEXT_INSUFFICIENT",
                main_cause_name="诊断信息不足",
                confidence=0.99,
                agent_mode="llm" if self._llm.enabled else "rule",
                llm_used=False,
                warnings=warnings,
                evidences=["未提供 personId、telephone、cardNo 或 deviceId"],
                suggested_actions=[],
                context=None,
                diagnosis=None,
            )

        steps.append("调用 cloudx-estate-ai 诊断工具")
        result = await self._client.get_result(request.project_id, normalized)
        context = result.get("context") or {}
        diagnosis = result.get("diagnosis") or {}
        answer = await self._build_answer(request.question, normalized, result, warnings)
        llm_used = self._llm.enabled and not any(item.startswith("LLM") for item in warnings)
        steps.append(f"分析诊断结果：{diagnosis.get('mainCauseName', '未知原因')}")
        actions = [SuggestedAction.model_validate(item) for item in diagnosis.get("suggestedActions") or []]
        status = "done"
        follow_up_question = None
        needs_input: list[str] = []
        if diagnosis.get("mainCause") == "CONTEXT_INSUFFICIENT":
            status = "need_more_info"
            follow_up_question = "请继续补充手机号、卡号、人员ID或设备ID中的有效信息，我再接着诊断。"
            needs_input = ["person_id", "telephone", "card_no", "device_id"]
            steps.append("诊断结果仍需要补充信息")

        return DiagnosisAgentResponse(
            trace_id=trace_id,
            session_id=session_id,
            status=status,
            normalized_request=normalized,
            answer=answer,
            follow_up_question=follow_up_question,
            needs_input=needs_input,
            steps=steps,
            available_actions=self._build_available_actions(actions),
            summary=diagnosis.get("summary", ""),
            main_cause=diagnosis.get("mainCause", "UNKNOWN"),
            main_cause_name=diagnosis.get("mainCauseName", "未知原因"),
            confidence=diagnosis.get("confidence"),
            agent_mode="llm" if llm_used else "rule",
            llm_used=llm_used,
            warnings=warnings,
            evidences=diagnosis.get("evidences") or [],
            suggested_actions=actions,
            context=context,
            diagnosis=diagnosis,
        )

    def _merge_session_context(self, session_id: str, current: dict[str, Any]) -> dict[str, Any]:
        """合并会话上下文
        
        将当前请求的参数与之前会话的参数合并，保留非空值。
        
        Args:
            session_id: 会话ID
            current: 当前请求的标准化参数
            
        Returns:
            dict[str, Any]: 合并后的参数字典
        """
        previous = self._sessions.get(session_id) or {}
        merged = previous.copy()
        for key, value in current.items():
            if value:
                merged[key] = value
            else:
                merged.setdefault(key, value)
        return merged

    def _build_available_actions(self, actions: list[SuggestedAction]) -> list[dict[str, Any]]:
        """构建可用操作列表
        
        将 SuggestedAction 对象列表转换为前端可用的操作字典列表。
        
        Args:
            actions: 建议操作对象列表
            
        Returns:
            list[dict[str, Any]]: 可用操作字典列表
        """
        return [
            {
                "action": item.action,
                "label": item.action_name,
                "description": item.description,
                "risk_level": item.risk_level,
                "need_confirm": True,
            }
            for item in actions
        ]

    async def _normalize_request(self, request: DiagnosisAgentRequest, warnings: list[str]) -> dict[str, Any]:
        """标准化请求参数
        
        从请求对象或问题描述中提取标准化的诊断参数。
        
        Args:
            request: 诊断请求对象
            warnings: 警告信息列表，用于记录处理过程中的异常
            
        Returns:
            dict[str, Any]: 标准化后的参数字典，包含 personId、telephone、cardNo、deviceId
        """
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
        """使用规则从问题描述中提取参数
        
        通过正则表达式从问题描述中提取手机号、设备ID、卡号和人员ID。
        
        Args:
            question: 问题描述文本
            
        Returns:
            dict[str, Any]: 提取的参数字典
        """
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
        """构建诊断回答
        
        优先使用LLM生成诊断回答，失败时使用规则生成。
        
        Args:
            question: 原始问题描述
            normalized: 标准化后的参数
            result: 诊断结果
            warnings: 警告信息列表
            
        Returns:
            str: 诊断回答文本
        """
        if self._llm.enabled:
            try:
                answer = await self._llm.explain_diagnosis(question, normalized, result)
                if answer:
                    return answer.strip()
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                warnings.append(f"LLM诊断解释失败，已切换规则解释: {exc}")
        return self._rule_answer(normalized, result.get("diagnosis") or {})

    def _rule_answer(self, normalized: dict[str, Any], diagnosis: dict[str, Any]) -> str:
        """使用规则生成诊断回答
        
        基于诊断结果和标准化参数，使用规则模板生成诊断回答。
        
        Args:
            normalized: 标准化后的参数
            diagnosis: 诊断结果
            
        Returns:
            str: 诊断回答文本
        """
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