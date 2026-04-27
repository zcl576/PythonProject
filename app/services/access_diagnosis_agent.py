import uuid
from typing import Any

from app.agent.field_extractor import DiagnosisFieldExtractor
from app.agent.memory import InMemorySessionStore
from app.agent.responder import DiagnosisResponder
from app.agent.tools import AccessDiagnosisToolExecutor
from app.agent.trace import TraceRecorder
from app.clients.estate_ai_client import EstateAiClient
from app.clients.llm_client import LlmClient
from app.schemas.diagnosis import DiagnosisAgentRequest, DiagnosisAgentResponse, SuggestedAction


class AccessDiagnosisAgentService:
    """门禁诊断代理编排服务
    
    负责编排门禁诊断的完整流程，包括参数提取、会话管理、工具调用和响应生成。
    """

    def __init__(
        self,
        client: EstateAiClient | None = None,
        llm: LlmClient | None = None,
        session_store: InMemorySessionStore | None = None,
    ) -> None:
        """初始化门禁诊断代理服务
        
        Args:
            client: Estate AI 客户端实例，默认为 None，会自动创建
            llm: LLM 客户端实例，默认为 None，会自动创建
            session_store: 会话存储实例，默认为 None，会自动创建
        """
        self._client = client or EstateAiClient()  # Estate AI 客户端
        self._llm = llm or LlmClient()  # LLM 客户端
        self._sessions = session_store or InMemorySessionStore()  # 会话存储
        self._field_extractor = DiagnosisFieldExtractor(self._llm)  # 字段提取器
        self._tools = AccessDiagnosisToolExecutor(self._client)  # 工具执行器
        self._responder = DiagnosisResponder(self._llm)  # 响应生成器

    async def diagnose(self, request: DiagnosisAgentRequest,project_id: int | None = None) -> DiagnosisAgentResponse:
        """执行门禁诊断
        
        完整的诊断流程：
        1. 生成追踪 ID 和会话 ID
        2. 提取诊断参数
        3. 合并会话上下文
        4. 检查诊断信息是否足够
        5. 调用诊断工具
        6. 分析诊断结果
        7. 生成诊断回复
        8. 构建诊断响应
        
        Args:
            request: 诊断请求对象
            
        Returns:
            DiagnosisAgentResponse: 诊断响应对象
        """
        trace_id = uuid.uuid4().hex  # 生成唯一追踪 ID
        session_id = request.session_id or trace_id  # 使用请求中的会话 ID 或生成新的
        warnings: list[str] = []  # 警告信息列表
        trace = TraceRecorder(trace_id)  # 创建追踪记录器
        trace.add("intent", "识别问题类型：门禁异常诊断")

        # 提取诊断参数
        extracted = await self._field_extractor.extract(request, warnings)
        trace.add("extract", "抽取诊断参数", {"extracted": self._compact(extracted)})
        
        # 合并会话上下文
        normalized = self._sessions.merge(session_id, extracted)
        trace.add("memory", "合并会话上下文", {"session_id": session_id, "normalized": self._compact(normalized)})

        # 检查诊断信息是否足够
        if not any(normalized.values()):
            follow_up_question = "请提供手机号、卡号、人员ID或设备ID中的任意一个，我再继续诊断。"
            trace.add("clarify", "诊断信息不足，等待补充关键标识")
            return DiagnosisAgentResponse(
                trace_id=trace_id,
                session_id=session_id,
                status="need_more_info",
                normalized_request=normalized,
                answer=follow_up_question,
                follow_up_question=follow_up_question,
                needs_input=["person_id", "telephone", "card_no", "device_id"],
                steps=trace.steps(),
                trace=trace.trace,
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

        # 调用诊断工具
        trace.add("tool", "调用 cloudx-estate-ai 诊断工具", {"tool": "cloudx.access_diagnosis.result"})
        tool_result = await self._tools.run_diagnosis(project_id, normalized)
        result = tool_result.output
        context = result.get("context") or {}
        diagnosis = result.get("diagnosis") or {}
        
        # 分析诊断结果
        trace.add(
            "reason",
            f"分析诊断结果：{diagnosis.get('mainCauseName', '未知原因')}",
            {"mainCause": diagnosis.get("mainCause"), "confidence": diagnosis.get("confidence")},
        )

        # 生成诊断回复
        answer = await self._responder.build_answer(request.question, normalized, result, warnings)
        # LLM 启用 且 没有 LLM 相关警告
        llm_used = self._llm.enabled and not any(item.startswith("LLM") for item in warnings)
        actions = [SuggestedAction.model_validate(item) for item in diagnosis.get("suggestedActions") or []]
        status = "done"
        follow_up_question = None
        needs_input: list[str] = []
        
        # 检查是否需要更多信息
        if diagnosis.get("mainCause") == "CONTEXT_INSUFFICIENT":
            status = "need_more_info"
            follow_up_question = "请继续补充手机号、卡号、人员ID或设备ID中的有效信息，我再接着诊断。"
            needs_input = ["person_id", "telephone", "card_no", "device_id"]
            trace.add("clarify", "诊断结果仍需要补充信息")

        # 构建诊断响应
        trace.add("respond", "生成诊断回复", {"status": status, "llm_used": llm_used})
        return DiagnosisAgentResponse(
            trace_id=trace_id,
            session_id=session_id,
            status=status,
            normalized_request=normalized,
            answer=answer,
            follow_up_question=follow_up_question,
            needs_input=needs_input,
            steps=trace.steps(),
            trace=trace.trace,
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

    def _build_available_actions(self, actions: list[SuggestedAction]) -> list[dict[str, Any]]:
        """构建可用操作列表
        
        将 SuggestedAction 对象转换为前端可用的操作字典列表。
        
        Args:
            actions: 建议操作列表
            
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

    def _compact(self, payload: dict[str, Any]) -> dict[str, Any]:
        """压缩字典，移除空值
        
        移除字典中的空值，返回只包含非空值的新字典。
        
        Args:
            payload: 原始字典
            
        Returns:
            dict[str, Any]: 压缩后的字典
        """
        return {key: value for key, value in payload.items() if value}