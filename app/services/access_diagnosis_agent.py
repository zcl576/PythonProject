import uuid
from typing import Any

from app.agent.field_extractor import DiagnosisFieldExtractor
from app.agent.langchain_agent import LangChainAccessAgent
from app.memory.memory import InMemorySessionStore
from app.agent.responder import DiagnosisResponder
from app.agent.tools import AccessDiagnosisToolExecutor
from app.agent.trace import TraceRecorder
from app.clients.estate_ai_client import EstateAiClient
from app.clients.llm_client import LlmClient
from app.schemas.diagnosis import DiagnosisAgentRequest, DiagnosisAgentResponse, SuggestedAction
from loguru import logger as log

class AccessDiagnosisAgentService:
    """门禁诊断代理编排服务
    
    负责编排门禁诊断的完整流程，包括参数提取、会话管理、工具调用和响应生成。
    
    该服务支持两种诊断模式：
    1. 确定性流程：基于预定义规则和工具调用进行诊断
    2. LangChain Agent 模式：使用 AI 自主规划工具调用来进行诊断
    
    主要职责：
    - 解析和提取诊断请求中的关键信息
    - 管理会话状态和上下文
    - 调用底层诊断工具获取结果
    - 生成结构化的诊断响应
    """

    def __init__(
        self,
        client: EstateAiClient | None = None,
        llm: LlmClient | None = None,
        session_store: InMemorySessionStore | None = None,
        langchain_agent: LangChainAccessAgent | None = None,
    ) -> None:
        """初始化门禁诊断代理服务
        
        初始化服务所需的各个组件，包括AI客户端、LLM客户端、会话存储等。
        如果某些组件未提供，将创建默认实例。
        
        Args:
            client: Estate AI 客户端实例，用于调用底层诊断服务
            llm: LLM 客户端实例，用于自然语言处理和响应生成
            session_store: 会话存储实例，用于维护会话上下文
            langchain_agent: LangChain代理实例，用于AI自主诊断
        """
        self._client = client or EstateAiClient()  # Estate AI 客户端，用于调用底层诊断服务
        self._llm = llm or LlmClient()  # LLM 客户端，用于自然语言理解和生成
        self._uses_default_llm = llm is None  # 标记是否使用默认LLM客户端
        self._sessions = session_store or InMemorySessionStore()  # 会话存储，用于维护对话上下文
        self._field_extractor = DiagnosisFieldExtractor(self._llm)  # 字段提取器，用于从请求中提取关键信息
        self._tools = AccessDiagnosisToolExecutor(self._client)  # 工具执行器，用于执行诊断工具
        self._responder = DiagnosisResponder(self._llm)  # 响应生成器，用于生成自然语言回复
        self._langchain_agent = langchain_agent or LangChainAccessAgent()  # LangChain代理，用于AI自主诊断

    async def diagnose(self, request: DiagnosisAgentRequest, project_id: int | str | None = None) -> DiagnosisAgentResponse:
        """执行门禁诊断
        
        完整的诊断流程：
        1. 生成追踪 ID 和会话 ID
        2. 验证项目 ID 是否存在
        3. 尝试使用 LangChain Agent 进行诊断（如果启用且有问句）
        4. 提取诊断参数
        5. 合并会话上下文
        6. 检查诊断信息是否足够
        7. 调用诊断工具
        8. 分析诊断结果
        9. 生成诊断回复
        10. 构建诊断响应
        
        该方法支持两种诊断模式：
        - LangChain Agent 模式：当使用默认LLM且Agent启用时，使用AI自主规划工具调用
        - 确定性流程：基于预定义规则和工具调用进行诊断
        
        Args:
            request: 诊断请求对象，包含用户问题和相关信息
            project_id: 项目ID，优先级高于request中的project_id
            
        Returns:
            DiagnosisAgentResponse: 包含诊断结果、建议操作等信息的响应对象
        """
        trace_id = uuid.uuid4().hex  # 生成唯一追踪 ID
        session_id = request.session_id or trace_id  # 使用请求中的会话 ID 或生成新的
        warnings: list[str] = []  # 警告信息列表
        trace = TraceRecorder(trace_id)  # 创建追踪记录器
        trace.add("intent", "识别问题类型：门禁异常诊断")
        effective_project_id = self._resolve_project_id(project_id, request.project_id)

        if effective_project_id is None:
            follow_up_question = "请在 PROJECT-ID 请求头中提供项目ID，我再继续诊断。"
            trace.add("clarify", "缺少项目ID，等待补充")
            return DiagnosisAgentResponse(
                trace_id=trace_id,
                session_id=session_id,
                status="need_more_info",
                normalized_request={},
                answer=follow_up_question,
                follow_up_question=follow_up_question,
                needs_input=["project_id"],
                steps=trace.steps(),
                trace=trace.trace,
                available_actions=[],
                summary="缺少项目ID",
                main_cause="PROJECT_ID_REQUIRED",
                main_cause_name="缺少项目ID",
                confidence=0.99,
                agent_mode="rule",
                llm_used=False,
                warnings=warnings,
                evidences=["未提供 PROJECT-ID 请求头或 request.project_id"],
                suggested_actions=[],
                context=None,
                diagnosis=None,
            )

        if self._uses_default_llm and self._langchain_agent.enabled and request.question:
            try:
                return await self._diagnose_with_langchain(request, effective_project_id, session_id, trace, warnings)
            except Exception as exc:
                log.warning(f"LangChain Agent 调用失败，已切换确定性流程: {exc}")
                warnings.append(f"LangChain Agent 调用失败，已切换确定性流程: {exc}")

        # 提取诊断参数
        extracted = await self._field_extractor.extract(request, warnings)
        trace.add("extract", "抽取诊断参数", {"extracted": self._compact(extracted)})
        
        # 合并会话上下文
        normalized = self._sessions.merge(session_id, extracted)
        trace.add("memory", "合并会话上下文", {"session_id": session_id, "normalized": self._compact(normalized)})

        # 检查诊断信息是否足够
        if not any(normalized.values()):
            follow_up_question = "请提供手机号、卡号、人员ID、人员姓名、设备ID、设备名称或设备SN中的任意一个，我再继续诊断。"
            trace.add("clarify", "诊断信息不足，等待补充关键标识")
            return DiagnosisAgentResponse(
                trace_id=trace_id,
                session_id=session_id,
                status="need_more_info",
                normalized_request=normalized,
                answer=follow_up_question,
                follow_up_question=follow_up_question,
                needs_input=["person_id", "telephone", "card_no", "person_name", "device_id", "device_name", "device_sn"],
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
                evidences=["未提供 personId、telephone、cardNo、personName、deviceId、deviceName 或 deviceSn"],
                suggested_actions=[],
                context=None,
                diagnosis=None,
            )

        # 调用诊断工具
        trace.add("tool", "调用 cloudx-estate-ai 诊断工具", {"tool": "cloudx.access_diagnosis.result"})
        tool_result = await self._tools.run_diagnosis(effective_project_id, normalized)
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
            follow_up_question = "请继续补充手机号、卡号、人员ID、人员姓名、设备ID、设备名称或设备SN中的有效信息，我再接着诊断。"
            needs_input = ["person_id", "telephone", "card_no", "person_name", "device_id", "device_name", "device_sn"]
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

    async def _diagnose_with_langchain(
        self,
        request: DiagnosisAgentRequest,
        project_id: int,
        session_id: str,
        trace: TraceRecorder,
        warnings: list[str],
    ) -> DiagnosisAgentResponse:
        """使用 LangChain Agent 执行门禁诊断
        
        该方法利用 LangChain Agent 的自主规划能力，让 AI 自主决定如何调用诊断工具。
        Agent 会根据用户问题和已知信息，自主规划工具调用序列以获得最佳诊断结果。
        
        Args:
            request: 诊断请求对象
            project_id: 项目ID
            session_id: 会话ID
            trace: 追踪记录器，用于记录诊断过程
            warnings: 警告信息列表
            
        Returns:
            DiagnosisAgentResponse: 包含诊断结果的响应对象
        """
        known_fields = {
            "personId": request.person_id,
            "telephone": request.telephone,
            "cardNo": request.card_no,
            "deviceId": request.device_id,
            "deviceName": request.device_name,
            "deviceSn": request.device_sn,
            "personName": request.person_name,
        }
        trace.add("planner", "使用 LangChain Agent 自主规划工具调用", {"known_fields": self._compact(known_fields)})
        agent_result = await self._langchain_agent.run(project_id, request.question or "", self._compact(known_fields), session_id)
        trace.add("tool", "LangChain Agent 已完成工具调用", {"tool_outputs": len(agent_result.tool_outputs)})

        result = agent_result.tool_outputs[-1] if agent_result.tool_outputs else {}
        context = result.get("context") or {}
        diagnosis = result.get("diagnosis") or {}
        actions = [SuggestedAction.model_validate(item) for item in diagnosis.get("suggestedActions") or []]
        status = "done" if result else "need_more_info"
        follow_up_question = None if result else "我还没有拿到可用的工具结果，请补充手机号、卡号、人员姓名、设备名称或设备SN。"
        needs_input = [] if result else ["person_id", "telephone", "card_no", "person_name", "device_id", "device_name", "device_sn"]

        trace.add(
            "respond",
            "LangChain Agent 生成最终回复",
            {"status": status, "mainCause": diagnosis.get("mainCause")},
        )
        return DiagnosisAgentResponse(
            trace_id=trace.trace.trace_id,
            session_id=session_id,
            status=status,
            normalized_request=self._compact(known_fields),
            answer=agent_result.answer or follow_up_question or "",
            follow_up_question=follow_up_question,
            needs_input=needs_input,
            steps=trace.steps(),
            trace=trace.trace,
            available_actions=self._build_available_actions(actions),
            summary=diagnosis.get("summary", ""),
            main_cause=diagnosis.get("mainCause", "UNKNOWN" if result else "CONTEXT_INSUFFICIENT"),
            main_cause_name=diagnosis.get("mainCauseName", "未知原因" if result else "诊断信息不足"),
            confidence=diagnosis.get("confidence"),
            agent_mode="langchain",
            llm_used=True,
            warnings=warnings,
            evidences=diagnosis.get("evidences") or [],
            suggested_actions=actions,
            context=context,
            diagnosis=diagnosis or None,
        )

    def _build_available_actions(self, actions: list[SuggestedAction]) -> list[dict[str, Any]]:
        """构建可用操作列表
        
        将 SuggestedAction 对象转换为前端友好的操作字典列表，便于前端展示和处理。
        每个操作字典包含操作类型、标签、描述、风险等级等信息。
        
        Args:
            actions: 建议操作对象列表
            
        Returns:
            list[dict[str, Any]]: 包含操作信息的字典列表，每个字典包含：
                - action: 操作类型
                - label: 操作标签
                - description: 操作描述
                - risk_level: 风险等级
                - need_confirm: 是否需要确认
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
        
        移除字典中的空值（如 None, "", [], {} 等），返回只包含非空值的新字典。
        该方法用于清理数据，减少传输的数据量并提高处理效率。
        
        Args:
            payload: 原始字典，可能包含空值
            
        Returns:
            dict[str, Any]: 压缩后的字典，只包含非空值的键值对
        """
        return {key: value for key, value in payload.items() if value}

    def _resolve_project_id(self, header_project_id: int | str | None, body_project_id: int | None) -> int | None:
        """解析项目ID
        
        从请求头或请求体中解析项目ID，优先使用请求头中的项目ID。
        如果两个来源都没有有效的项目ID，则返回 None。
        同时会尝试将字符串类型的ID转换为整数。
        
        Args:
            header_project_id: 从请求头获取的项目ID
            body_project_id: 从请求体获取的项目ID
            
        Returns:
            int | None: 解析后的项目ID，如果无法解析则返回 None
        """
        # 优先使用请求头中的项目ID，如果为空则使用请求体中的
        raw = header_project_id if header_project_id not in (None, "") else body_project_id
        if raw in (None, ""):
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            log.error(f"项目ID提取失败")
            return None
