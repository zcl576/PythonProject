"""权限代理图服务 V6 版本

基于 LangGraph 构建的状态机服务，实现权限查询相关的对话流程。
支持的工具包括：人员查询、设备查询、权限查询、权限续期等。

状态机节点说明：
- understand_with_llm:    LLM 意图理解和槽位提取
- normalize_plan:         计划规范化（意图转工具）
- resolve_slots:          槽位解析（补全缺失信息）
- execute_tool:           工具执行
- handle_tool_result:     工具结果处理
- confirm_write:          写操作确认（权限续期等需要确认的操作）
- final_llm_answer:       最终回答生成
"""
from __future__ import annotations

import uuid
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.clients.llm_client import LlmClient
from app.config import get_settings
from app.memory.langgraph_mysql_checkpoint_v6 import get_v6_mysql_checkpointer
from app.schemas.right_schema_v6 import (
    AgentChoiceV6,
    AgentConfirmV6,
    AgentRequestV6,
    AgentResponseV6,
)
from app.services.right_agent_metadata_v6 import (
    PLANNER_TARGET_TO_TOOL_V6,
    READ_TOOLS_V6,
    TOOL_METADATA_V6,
    WRITE_TOOLS_V6,
)
from app.services.right_agent_policy_v6 import RightAgentPolicyCheckerV6
from app.tools.right_tools_v6 import RightToolsV6

# 中断键名，用于存储 LangGraph 中断信息
INTERRUPT_KEY = "__interrupt__"

# V6 版本支持的槽位名称
SLOT_NAMES_V6 = (
    "personId",      # 人员ID
    "personName",    # 人员姓名
    "telephone",     # 手机号
    "deviceSn",      # 设备序列号
    "deviceId",      # 设备ID
    "deviceName",    # 设备名称
    "cardNo",        # 卡号
    "durationDays",  # 续期天数
)


class RightAgentStateV6(TypedDict, total=False):
    """权限代理状态 V6 版本

    定义了状态机执行过程中的所有状态字段。

    Attributes:
        session_id: 会话ID
        project_id: 项目ID
        question: 用户问题
        intent: 用户意图
        requested_tool: 用户请求的工具
        target_tool: 目标工具
        current_tool: 当前执行的工具
        pending_write_tool: 待执行的写工具（如续期）
        slots: 槽位数据
        tool_result: 当前工具执行结果
        tool_history: 工具执行历史
        permission_result: 权限查询结果
        permission_status: 权限状态
        write_result: 写操作结果
        policy: 策略决策结果
        status: 当前状态（ok/error/need_more_info/need_confirm/cancelled等）
        answer: 最终回答
        follow_up_question: 追问问题
        needs_input: 需要用户输入的字段列表
        choices: 选择选项列表
        confirm: 确认信息
        data: 附加数据
    """
    session_id: str
    project_id: int
    question: str | None
    intent: str | None
    requested_tool: str
    target_tool: str
    current_tool: str
    pending_write_tool: str | None
    slots: dict[str, Any]
    tool_result: dict[str, Any]
    tool_history: list[dict[str, Any]]
    permission_result: dict[str, Any]
    permission_status: str
    write_result: dict[str, Any]
    policy: dict[str, Any]
    status: str
    answer: str
    follow_up_question: str | None
    needs_input: list[str]
    choices: list[dict[str, Any]]
    confirm: dict[str, Any] | None
    data: dict[str, Any]


class RightAgentGraphServiceV6:
    """权限代理图服务 V6 版本

    基于 LangGraph 构建的状态机服务，实现权限查询相关的对话流程。

    核心特性：
    - 使用 LLM 进行意图理解和槽位提取
    - 支持自动解析缺失的槽位（人员/设备查询）
    - 策略检查机制控制写操作权限
    - 支持中断机制处理用户选择和确认
    - 持久化会话状态支持断点续传

    状态机节点说明：
    - understand_with_llm:    LLM 意图理解和槽位提取
    - normalize_plan:         计划规范化（意图转工具）
    - resolve_slots:          槽位解析（补全缺失信息）
    - execute_tool:           工具执行
    - handle_tool_result:     工具结果处理
    - confirm_write:          写操作确认
    - final_llm_answer:       最终回答生成
    """

    def __init__(
        self,
        *,
        tools: RightToolsV6 | None = None,
        llm: LlmClient | None = None,
        policy: RightAgentPolicyCheckerV6 | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        """初始化权限代理图服务

        Args:
            tools: 工具执行器，用于执行人员查询、设备查询、权限查询等操作
            llm: LLM 客户端，用于意图理解和回答生成
            policy: 策略检查器，用于控制写操作权限
            checkpointer: 状态检查点，用于持久化会话状态
        """
        self._settings = get_settings()
        self._tools = tools or RightToolsV6()
        self._llm = llm or LlmClient()
        self._policy = policy or RightAgentPolicyCheckerV6()
        self._checkpointer = checkpointer
        self._graph = None

    async def do_execute(
        self,
        request: AgentRequestV6,
        project_id: str | None,
    ) -> AgentResponseV6:
        """执行权限代理服务

        处理用户请求，执行状态机流程，返回响应。

        Args:
            request: 代理请求对象
            project_id: 项目ID字符串

        Returns:
            AgentResponseV6: 代理响应对象
        """
        effective_project_id = self._parse_project_id(project_id)
        session_id = request.session_id or uuid.uuid4().hex

        # 验证项目ID
        if effective_project_id is None:
            return AgentResponseV6(
                answer="",
                status="error",
                session_id=session_id,
                follow_up_question="Project-Id is required.",
                needs_input=["Project-Id"],
            )

        # 获取状态机并执行
        graph = await self._get_graph()
        config = {"configurable": {"thread_id": self._thread_id(effective_project_id, session_id)}}

        try:
            # 如果是恢复请求，使用 resume 命令
            if request.resume is not None:
                state = await graph.ainvoke(
                    Command(resume=self._resume_payload(request.resume)),
                    config=config,
                )
            else:
                # 新请求，初始化状态
                state = await graph.ainvoke(
                    {
                        "session_id": session_id,
                        "project_id": effective_project_id,
                        "question": request.question,
                        "slots": {},
                        "choices": [],
                        "data": {},
                        "needs_input": [],
                        "tool_history": [],
                    },
                    config=config,
                )
        except ValueError as exc:
            return AgentResponseV6(answer="", status="error", session_id=session_id, data={"error": str(exc)})

        return self._to_response(state, session_id)

    async def _get_graph(self):
        """获取状态机实例

        懒加载状态机，确保只创建一次。

        Returns:
            编译后的状态机实例
        """
        if self._graph is not None:
            return self._graph
        checkpointer = self._checkpointer or await get_v6_mysql_checkpointer()
        self._graph = self._build_graph().compile(checkpointer=checkpointer)
        return self._graph

    def _build_graph(self):
        """构建状态机图

        定义状态机的节点和边，构建完整的对话流程。

        Returns:
            StateGraph: 状态机构建器
        """
        builder = StateGraph(RightAgentStateV6)

        # 添加节点
        builder.add_node("understand_with_llm", self._understand_with_llm)    # LLM 意图理解
        builder.add_node("normalize_plan", self._normalize_plan)                # 计划规范化
        builder.add_node("resolve_slots", self._resolve_slots)                  # 槽位解析
        builder.add_node("execute_tool", self._execute_tool)                    # 工具执行
        builder.add_node("handle_tool_result", self._handle_tool_result)        # 结果处理
        builder.add_node("confirm_write", self._confirm_write)                  # 写操作确认
        builder.add_node("final_llm_answer", self._final_llm_answer)            # 最终回答

        # 添加边（定义流程）
        builder.add_edge(START, "understand_with_llm")
        builder.add_edge("understand_with_llm", "normalize_plan")

        # 规范化后的路由
        builder.add_conditional_edges(
            "normalize_plan",
            self._route_after_normalize,
            {"resolve_slots": "resolve_slots", "final_llm_answer": "final_llm_answer"},
        )

        # 槽位解析后的路由
        builder.add_conditional_edges(
            "resolve_slots",
            self._route_after_resolve_slots,
            {"execute_tool": "execute_tool", "final_llm_answer": "final_llm_answer"},
        )

        builder.add_edge("execute_tool", "handle_tool_result")

        # 工具结果处理后的路由
        builder.add_conditional_edges(
            "handle_tool_result",
            self._route_after_tool_result,
            {
                "resolve_slots": "resolve_slots",
                "confirm_write": "confirm_write",
                "final_llm_answer": "final_llm_answer",
            },
        )

        # 确认后的路由
        builder.add_conditional_edges(
            "confirm_write",
            self._route_after_confirm_write,
            {"execute_tool": "execute_tool", "final_llm_answer": "final_llm_answer"},
        )

        builder.add_edge("final_llm_answer", END)
        return builder

    async def _understand_with_llm(self, state: RightAgentStateV6) -> RightAgentStateV6:
        """LLM 意图理解节点

        使用 LLM 从用户问题中提取意图、目标工具和槽位信息。

        Args:
            state: 当前状态

        Returns:
            更新后的状态，包含意图、目标工具和槽位
        """
        question = (state.get("question") or "").strip()

        # 检查问题是否为空
        if not question:
            return {
                "status": "need_more_info",
                "answer": "Please describe the permission request.",
                "follow_up_question": "Please provide a person, device, or permission question.",
                "needs_input": ["question"],
            }

        # 调用 LLM 进行意图理解
        try:
            plan = await self._llm.understand_right_agent_v6(
                question=question,
                target_tools=tuple(TOOL_METADATA_V6.keys()),
                slot_names=SLOT_NAMES_V6,
            )
        except Exception as exc:
            return {"status": "error", "answer": "", "data": {"llm_error": str(exc), "stage": "planner"}}

        # 检查 LLM 是否返回有效结果
        if not plan:
            return {"status": "error", "answer": "", "data": {"llm_error": "planner returned no plan"}}

        return {
            "intent": plan.get("intent"),
            "requested_tool": plan.get("target_tool") or "",
            "slots": self._clean_slots(plan.get("slots") or {}),
            "data": {"planner": plan},
        }

    async def _normalize_plan(self, state: RightAgentStateV6) -> RightAgentStateV6:
        """计划规范化节点

        将用户意图转换为具体的工具，并处理写操作的前置检查。

        Args:
            state: 当前状态

        Returns:
            更新后的状态，包含目标工具和当前工具
        """
        # 如果已有错误或需要更多信息，直接返回
        if state.get("status") in ("need_more_info", "error"):
            return {}

        requested = state.get("requested_tool") or ""
        intent = state.get("intent")

        # 将意图/请求工具映射到实际工具
        target = PLANNER_TARGET_TO_TOOL_V6.get(requested) or PLANNER_TARGET_TO_TOOL_V6.get(str(intent))
        data = state.get("data") or {}

        # 处理写工具（需要先查询权限）
        if target in WRITE_TOOLS_V6:
            return {
                "target_tool": "query_permission",
                "current_tool": "query_permission",
                "pending_write_tool": target,
                "status": "",
                "data": {**data, "requested_write_tool": target},
            }

        # 处理读工具
        if target in READ_TOOLS_V6:
            return {
                "target_tool": target,
                "current_tool": target,
                "pending_write_tool": None,
                "status": "",
                "data": data,
            }

        # 未知工具
        return {
            "status": "error",
            "answer": "",
            "data": {**data, "error": "unsupported target tool", "target_tool": target},
        }

    def _route_after_normalize(self, state: RightAgentStateV6) -> str:
        """计划规范化后的路由函数

        Args:
            state: 当前状态

        Returns:
            下一个节点名称
        """
        if state.get("status") in ("need_more_info", "error"):
            return "final_llm_answer"
        return "resolve_slots"

    async def _resolve_slots(self, state: RightAgentStateV6) -> RightAgentStateV6:
        """槽位解析节点

        检查工具所需的槽位是否完整，缺失时触发解析器或请求用户输入。

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        current_tool = state.get("current_tool") or state.get("target_tool")

        # 验证工具是否有效
        if not current_tool or current_tool not in TOOL_METADATA_V6:
            return {"status": "error", "answer": "", "data": {"error": "unknown current tool"}}

        metadata = TOOL_METADATA_V6[current_tool]
        slots = state.get("slots") or {}

        # 检查可选必填槽位
        missing_any = self._missing_any_required(metadata.any_required_slots, slots)
        if missing_any:
            return {
                "status": "need_more_info",
                "answer": "Missing information for the next tool.",
                "follow_up_question": f"Please provide one of: {', '.join(missing_any)}.",
                "needs_input": missing_any,
            }

        # 检查必填槽位
        for slot in metadata.required_slots:
            if slots.get(slot):
                continue

            # 尝试使用解析器获取缺失槽位
            resolver = metadata.slot_resolvers.get(slot)
            if resolver:
                return {
                    "current_tool": resolver,
                    "status": "",
                    "tool_result": {},
                    "data": {**(state.get("data") or {}), "after_resolver_tool": state.get("target_tool")},
                }

            # 请求用户输入
            return {
                "status": "need_more_info",
                "answer": "Missing required information.",
                "follow_up_question": f"Please provide {slot}.",
                "needs_input": [slot],
            }

        return {"current_tool": current_tool, "status": ""}

    def _route_after_resolve_slots(self, state: RightAgentStateV6) -> str:
        """槽位解析后的路由函数

        Args:
            state: 当前状态

        Returns:
            下一个节点名称
        """
        if state.get("status") in ("need_more_info", "error"):
            return "final_llm_answer"
        return "execute_tool"

    async def _execute_tool(self, state: RightAgentStateV6) -> RightAgentStateV6:
        """工具执行节点

        调用工具执行器执行当前工具。

        Args:
            state: 当前状态

        Returns:
            更新后的状态，包含工具执行结果
        """
        result = await self._tools.execute(
            state["current_tool"],
            state["project_id"],
            state.get("slots") or {},
        )
        return {"tool_result": result}

    async def _handle_tool_result(self, state: RightAgentStateV6) -> RightAgentStateV6:
        """工具结果处理节点

        根据工具类型处理执行结果，包括人员查询、设备查询、权限查询和写操作。

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        result = state.get("tool_result") or {}
        current_tool = state["current_tool"]
        history = self._append_tool_history(state, current_tool, result)
        data = {**(state.get("data") or {}), "tool_history": history}

        # 处理未配置的工具
        if result.get("status") == "not_configured":
            return {"status": "error", "answer": "", "tool_history": history, "data": {**data, "result": result}}

        target = state.get("target_tool") or current_tool

        # 处理解析器结果（当前工具是解析器，不是目标工具）
        if current_tool != target:
            return self._handle_resolver_result(state, current_tool, result, history, data)

        # 处理人员查询
        if current_tool == "search_person":
            return self._handle_person_lookup(state, result, history, data)

        # 处理设备查询
        if current_tool == "search_device":
            return self._handle_device_lookup(state, result, history, data)

        # 处理权限查询
        if current_tool == "query_permission":
            permission_status = self._policy.permission_status(result).value
            data = {**data, "permission_result": result, "permission_status": permission_status}

            # 如果有待执行的写操作，进行策略检查
            pending_write = state.get("pending_write_tool")
            if pending_write:
                decision = self._policy.check_write(
                    tool=TOOL_METADATA_V6[pending_write],
                    permission_result=result,
                    confirmed=False,
                )
                data = {**data, "policy": dict(decision)}
                return {
                    "status": decision.get("status") or ("need_confirm" if not decision.allowed else "ok"),
                    "permission_result": result,
                    "permission_status": permission_status,
                    "policy": dict(decision),
                    "tool_history": history,
                    "data": data,
                }

            return {
                "status": "ok",
                "permission_result": result,
                "permission_status": permission_status,
                "tool_history": history,
                "data": data,
            }

        # 处理写工具
        if current_tool in WRITE_TOOLS_V6:
            return {
                "status": "ok",
                "write_result": result,
                "tool_history": history,
                "data": {**data, "write_result": result},
            }

        return {"status": "error", "answer": "", "tool_history": history, "data": {**data, "error": "unhandled tool"}}

    def _handle_resolver_result(
        self,
        state: RightAgentStateV6,
        current_tool: str,
        result: dict[str, Any],
        history: list[dict[str, Any]],
        data: dict[str, Any],
    ) -> RightAgentStateV6:
        """处理解析器结果

        当使用解析器工具（search_person/search_device）获取缺失槽位时，处理其结果。

        Args:
            state: 当前状态
            current_tool: 当前工具名称
            result: 工具执行结果
            history: 工具历史
            data: 附加数据

        Returns:
            更新后的状态
        """
        if current_tool == "search_person":
            people = self._extract_people(result)
            selected = self._select_candidate(
                kind="person",
                candidates=people,
                choices=self._person_choices(people),
                empty_answer="No matching person was found.",
            )

            # 如果有状态（错误或需要选择），直接返回
            if selected.get("status"):
                return {**selected, "tool_history": history, "data": {**data, **(selected.get("data") or {})}}

            # 更新槽位并继续解析
            return {
                "slots": {**(state.get("slots") or {}), **self._person_slots(selected["data"])},
                "current_tool": state.get("target_tool") or "query_permission",
                "status": "continue_resolve",
                "tool_history": history,
                "data": data,
            }

        if current_tool == "search_device":
            devices = self._extract_devices(result)
            selected = self._select_candidate(
                kind="device",
                candidates=devices,
                choices=self._device_choices(devices),
                empty_answer="No matching device was found.",
            )

            # 如果有状态（错误或需要选择），直接返回
            if selected.get("status"):
                return {**selected, "tool_history": history, "data": {**data, **(selected.get("data") or {})}}

            # 更新槽位并继续解析
            return {
                "slots": {**(state.get("slots") or {}), **self._device_slots(selected["data"])},
                "current_tool": state.get("target_tool") or "query_permission",
                "status": "continue_resolve",
                "tool_history": history,
                "data": data,
            }

        return {"status": "error", "answer": "", "tool_history": history, "data": {**data, "error": "bad resolver"}}

    def _handle_person_lookup(
        self,
        state: RightAgentStateV6,
        result: dict[str, Any],
        history: list[dict[str, Any]],
        data: dict[str, Any],
    ) -> RightAgentStateV6:
        """处理人员查询结果

        提取人员列表并处理选择逻辑。

        Args:
            state: 当前状态
            result: 工具执行结果
            history: 工具历史
            data: 附加数据

        Returns:
            更新后的状态
        """
        people = self._extract_people(result)
        selected = self._select_candidate(
            kind="person",
            candidates=people,
            choices=self._person_choices(people),
            empty_answer="No matching person was found.",
        )

        # 如果需要用户选择或有错误，返回选择状态
        if selected.get("status"):
            return {**selected, "tool_history": history, "data": {**data, **(selected.get("data") or {})}}

        return {
            "status": "ok",
            "slots": {**(state.get("slots") or {}), **self._person_slots(selected["data"])},
            "tool_history": history,
            "data": {**data, "person": selected["data"]},
        }

    def _handle_device_lookup(
        self,
        state: RightAgentStateV6,
        result: dict[str, Any],
        history: list[dict[str, Any]],
        data: dict[str, Any],
    ) -> RightAgentStateV6:
        """处理设备查询结果

        提取设备列表并处理选择逻辑。

        Args:
            state: 当前状态
            result: 工具执行结果
            history: 工具历史
            data: 附加数据

        Returns:
            更新后的状态
        """
        devices = self._extract_devices(result)
        selected = self._select_candidate(
            kind="device",
            candidates=devices,
            choices=self._device_choices(devices),
            empty_answer="No matching device was found.",
        )

        # 如果需要用户选择或有错误，返回选择状态
        if selected.get("status"):
            return {**selected, "tool_history": history, "data": {**data, **(selected.get("data") or {})}}

        return {
            "status": "ok",
            "slots": {**(state.get("slots") or {}), **self._device_slots(selected["data"])},
            "tool_history": history,
            "data": {**data, "device": selected["data"]},
        }

    def _route_after_tool_result(self, state: RightAgentStateV6) -> str:
        """工具结果处理后的路由函数

        Args:
            state: 当前状态

        Returns:
            下一个节点名称
        """
        if state.get("status") == "continue_resolve":
            return "resolve_slots"
        if state.get("status") == "need_confirm":
            return "confirm_write"
        return "final_llm_answer"

    async def _confirm_write(self, state: RightAgentStateV6) -> RightAgentStateV6:
        """写操作确认节点

        处理写操作的用户确认逻辑，使用中断机制暂停执行并等待用户确认。

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        write_tool = state.get("pending_write_tool")
        permission_result = state.get("permission_result") or (state.get("data") or {}).get("permission_result") or {}

        # 验证写工具是否有效
        if not write_tool or write_tool not in WRITE_TOOLS_V6:
            return {"status": "error", "answer": "", "data": {**(state.get("data") or {}), "error": "missing write tool"}}

        metadata = TOOL_METADATA_V6[write_tool]
        decision = self._policy.check_write(tool=metadata, permission_result=permission_result, confirmed=False)

        # 如果不需要确认，直接返回策略决策
        if decision.get("reason") != "requires_confirm":
            return {
                "status": decision.get("status") or "policy_denied",
                "answer": "",
                "policy": dict(decision),
                "data": {**(state.get("data") or {}), "policy": dict(decision)},
            }

        # 生成确认中断
        confirm_id = metadata.confirm_id or write_tool
        resume = interrupt(
            {
                "type": "confirm",
                "answer": f"{metadata.description} requires confirmation.",
                "follow_up_question": "Please confirm whether to continue.",
                "confirm": {
                    "id": confirm_id,
                    "label": metadata.description,
                    "description": f"Confirm to execute {metadata.name}.",
                    "risk_level": "medium",
                },
                "data": {
                    "permission_status": decision.get("permission_status"),
                    "write_tool": write_tool,
                },
            }
        )

        # 检查用户是否确认
        if not self._is_confirmed(resume, confirm_id):
            return {
                "status": "cancelled",
                "data": {**(state.get("data") or {}), "permission_result": permission_result, "policy": dict(decision)},
            }

        # 用户确认后再次检查策略
        confirmed_decision = self._policy.check_write(tool=metadata, permission_result=permission_result, confirmed=True)
        if not confirmed_decision.allowed:
            return {
                "status": confirmed_decision.get("status") or "policy_denied",
                "answer": "",
                "policy": dict(confirmed_decision),
                "data": {**(state.get("data") or {}), "policy": dict(confirmed_decision)},
            }

        # 准备执行写工具
        return {
            "current_tool": write_tool,
            "target_tool": write_tool,
            "status": "",
            "policy": dict(confirmed_decision),
            "data": {**(state.get("data") or {}), "policy": dict(confirmed_decision)},
        }

    def _route_after_confirm_write(self, state: RightAgentStateV6) -> str:
        """确认后的路由函数

        Args:
            state: 当前状态

        Returns:
            下一个节点名称
        """
        if state.get("status") in ("cancelled", "error", "policy_denied"):
            return "final_llm_answer"
        return "execute_tool"

    async def _final_llm_answer(self, state: RightAgentStateV6) -> RightAgentStateV6:
        """最终回答节点

        使用 LLM 根据工具执行结果生成面向用户的最终回答。

        Args:
            state: 当前状态

        Returns:
            更新后的状态，包含最终回答
        """
        # 如果已有回答，直接返回
        if state.get("answer"):
            return {}

        # 如果需要更多信息，直接返回（使用已有的追问问题）
        if state.get("status") == "need_more_info" and state.get("follow_up_question"):
            return {}

        # 调用 LLM 生成最终回答
        try:
            answer = await self._llm.answer_right_agent_v6(
                question=state.get("question"),
                slots=state.get("slots") or {},
                tool_history=state.get("tool_history") or [],
                permission_status=state.get("permission_status") or (state.get("data") or {}).get("permission_status"),
                permission_result=state.get("permission_result") or (state.get("data") or {}).get("permission_result"),
                write_result=state.get("write_result") or (state.get("data") or {}).get("write_result"),
                policy=state.get("policy") or (state.get("data") or {}).get("policy"),
            )
        except Exception as exc:
            return {"status": "error", "answer": "", "data": {**(state.get("data") or {}), "llm_error": str(exc), "stage": "final_answer"}}

        # 检查回答是否为空
        if not answer:
            return {"status": "error", "answer": "", "data": {**(state.get("data") or {}), "llm_error": "empty final answer"}}

        return {"status": state.get("status") or "ok", "answer": answer}

    @staticmethod
    def _append_tool_history(state: RightAgentStateV6, tool_name: str, result: dict[str, Any]) -> list[dict[str, Any]]:
        """追加工具执行历史

        Args:
            state: 当前状态
            tool_name: 工具名称
            result: 工具执行结果

        Returns:
            更新后的工具历史列表
        """
        return [*(state.get("tool_history") or []), {"tool": tool_name, "result": result}]

    @staticmethod
    def _missing_any_required(required: tuple[str, ...], slots: dict[str, Any]) -> list[str]:
        """检查可选必填槽位是否缺失

        如果任意一个槽位有值，则认为满足；否则返回所有缺失的槽位。

        Args:
            required: 可选必填槽位列表（填充任意一个即可）
            slots: 当前槽位数据

        Returns:
            缺失的槽位列表（如果已有任意槽位有值则返回空列表）
        """
        if not required:
            return []
        if any(slots.get(slot) for slot in required):
            return []
        return list(required)

    @staticmethod
    def _clean_slots(slots: dict[str, Any]) -> dict[str, Any]:
        """清理槽位数据

        过滤掉不在允许列表中的槽位和空值槽位。

        Args:
            slots: 原始槽位数据

        Returns:
            清理后的槽位数据
        """
        return {key: value for key, value in slots.items() if key in SLOT_NAMES_V6 and value not in (None, "")}

    @staticmethod
    def _extract_people(result: dict[str, Any]) -> list[dict[str, Any]]:
        """从工具结果中提取人员列表

        支持多种数据格式，自动识别并提取人员信息。

        Args:
            result: 工具执行结果

        Returns:
            人员列表
        """
        candidates = [
            result.get("people"),
            result.get("persons"),
            result.get("records"),
            result.get("list"),
            (result.get("data") or {}).get("people") if isinstance(result.get("data"), dict) else None,
            (result.get("data") or {}).get("records") if isinstance(result.get("data"), dict) else None,
        ]
        for value in candidates:
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        context = result.get("context")
        if isinstance(context, dict) and (context.get("personId") or context.get("personName")):
            return [context]
        return []

    @staticmethod
    def _extract_devices(result: dict[str, Any]) -> list[dict[str, Any]]:
        """从工具结果中提取设备列表

        支持多种数据格式，自动识别并提取设备信息。

        Args:
            result: 工具执行结果

        Returns:
            设备列表
        """
        candidates = [
            result.get("devices"),
            result.get("deviceList"),
            result.get("records"),
            result.get("list"),
            (result.get("data") or {}).get("devices") if isinstance(result.get("data"), dict) else None,
            (result.get("data") or {}).get("records") if isinstance(result.get("data"), dict) else None,
        ]
        for value in candidates:
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        context = result.get("context")
        if isinstance(context, dict) and (context.get("deviceSn") or context.get("deviceId") or context.get("deviceName")):
            return [context]
        return []

    @classmethod
    def _person_choices(cls, people: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将人员列表转换为选择选项格式

        Args:
            people: 人员列表

        Returns:
            选择选项列表
        """
        return [
            {
                "id": str(person.get("personId") or person.get("id") or index),
                "label": cls._person_label(person),
                "description": person.get("description"),
                "data": person,
            }
            for index, person in enumerate(people, start=1)
        ]

    @classmethod
    def _device_choices(cls, devices: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将设备列表转换为选择选项格式

        Args:
            devices: 设备列表

        Returns:
            选择选项列表
        """
        return [
            {
                "id": str(device.get("deviceSn") or device.get("deviceId") or device.get("id") or index),
                "label": cls._device_label(device),
                "description": device.get("description"),
                "data": device,
            }
            for index, device in enumerate(devices, start=1)
        ]

    @staticmethod
    def _person_label(person: dict[str, Any]) -> str:
        """生成人员显示标签

        Args:
            person: 人员信息

        Returns:
            人员显示标签（姓名 + 手机号）
        """
        return f"{person.get('personName') or person.get('name') or 'Unknown person'} {person.get('telephone') or person.get('phone') or ''}".strip()

    @staticmethod
    def _device_label(device: dict[str, Any]) -> str:
        """生成设备显示标签

        Args:
            device: 设备信息

        Returns:
            设备显示标签（名称 + 序列号）
        """
        return f"{device.get('deviceName') or device.get('name') or 'Unknown device'} {device.get('deviceSn') or device.get('sn') or device.get('deviceId') or ''}".strip()

    @staticmethod
    def _person_slots(person: dict[str, Any]) -> dict[str, Any]:
        """从人员信息中提取槽位数据

        Args:
            person: 人员信息

        Returns:
            槽位数据
        """
        return {
            "personId": person.get("personId") or person.get("id"),
            "personName": person.get("personName") or person.get("name"),
            "telephone": person.get("telephone") or person.get("phone"),
        }

    @staticmethod
    def _device_slots(device: dict[str, Any]) -> dict[str, Any]:
        """从设备信息中提取槽位数据

        Args:
            device: 设备信息

        Returns:
            槽位数据
        """
        return {
            "deviceSn": device.get("deviceSn") or device.get("sn"),
            "deviceId": device.get("deviceId") or device.get("id"),
            "deviceName": device.get("deviceName") or device.get("name"),
        }

    def _select_candidate(self, *, kind: str, candidates: list[dict[str, Any]], choices: list[dict[str, Any]], empty_answer: str) -> dict[str, Any]:
        """选择候选对象

        根据候选数量处理选择逻辑：
        - 无候选：返回需要更多信息状态
        - 单候选：直接返回该候选
        - 多候选：触发中断等待用户选择

        Args:
            kind: 对象类型（person/device）
            candidates: 候选列表
            choices: 选择选项列表
            empty_answer: 无候选时的提示信息

        Returns:
            包含选择结果的字典
        """
        if not candidates:
            return {"status": "need_more_info", "answer": empty_answer, "data": {"result": []}}
        if len(candidates) == 1:
            return {"data": candidates[0]}

        # 多候选时触发中断
        resume = interrupt(
            {
                "type": "choice",
                "kind": kind,
                "answer": "Multiple matches found. Please choose one.",
                "follow_up_question": "Please choose the object to continue.",
                "choices": choices,
            }
        )

        choice = self._choice_from_resume(resume, choices)
        if choice is None:
            return {"status": "error", "answer": "", "data": {"error": "invalid choice"}}
        return {"data": choice.get("data") or {}}

    @staticmethod
    def _choice_from_resume(resume: Any, choices: list[dict[str, Any]]) -> dict[str, Any] | None:
        """从恢复数据中提取用户选择

        Args:
            resume: 恢复数据
            choices: 可选选项列表

        Returns:
            用户选择的选项，如果无效则返回 None
        """
        if not isinstance(resume, dict) or resume.get("type") != "choice":
            return None
        choice_id = resume.get("choice_id")
        for choice in choices:
            if str(choice.get("id")) == str(choice_id):
                return choice
        return None

    @staticmethod
    def _is_confirmed(resume: Any, confirm_id: str) -> bool:
        """检查用户是否确认

        Args:
            resume: 恢复数据
            confirm_id: 确认操作ID

        Returns:
            用户是否确认
        """
        return (
            isinstance(resume, dict)
            and resume.get("type") == "confirm"
            and resume.get("confirm_id") == confirm_id
            and resume.get("confirmed") is True
        )

    def _to_response(self, state: dict[str, Any], session_id: str) -> AgentResponseV6:
        """将状态转换为响应对象

        Args:
            state: 当前状态
            session_id: 会话ID

        Returns:
            AgentResponseV6: 代理响应对象
        """
        interrupt_payload = self._interrupt_payload(state)
        if interrupt_payload:
            return self._interrupt_to_response(interrupt_payload, session_id)
        return AgentResponseV6(
            answer=state.get("answer") or "",
            status=state.get("status") or "ok",
            session_id=state.get("session_id") or session_id,
            follow_up_question=state.get("follow_up_question"),
            choices=[AgentChoiceV6.model_validate(item) for item in state.get("choices") or []],
            confirm=AgentConfirmV6.model_validate(state["confirm"]) if state.get("confirm") else None,
            needs_input=state.get("needs_input") or [],
            data=state.get("data") or {},
        )

    @staticmethod
    def _interrupt_payload(state: dict[str, Any]) -> dict[str, Any] | None:
        """从状态中提取中断载荷

        Args:
            state: 当前状态

        Returns:
            中断载荷，如果没有中断则返回 None
        """
        interrupts = state.get(INTERRUPT_KEY)
        if not interrupts:
            return None
        first = interrupts[0] if isinstance(interrupts, (list, tuple)) else interrupts
        value = getattr(first, "value", first)
        return value if isinstance(value, dict) else {"value": value}

    def _interrupt_to_response(self, payload: dict[str, Any], session_id: str) -> AgentResponseV6:
        """将中断载荷转换为响应对象

        Args:
            payload: 中断载荷
            session_id: 会话ID

        Returns:
            AgentResponseV6: 代理响应对象
        """
        if payload.get("type") == "choice":
            return AgentResponseV6(
                answer=payload.get("answer") or "",
                status="need_choice",
                session_id=session_id,
                follow_up_question=payload.get("follow_up_question"),
                choices=[AgentChoiceV6.model_validate(item) for item in payload.get("choices") or []],
                data=payload.get("data") or {},
            )
        if payload.get("type") == "confirm":
            confirm = payload.get("confirm")
            return AgentResponseV6(
                answer=payload.get("answer") or "",
                status="need_confirm",
                session_id=session_id,
                follow_up_question=payload.get("follow_up_question"),
                confirm=AgentConfirmV6.model_validate(confirm) if confirm else None,
                data=payload.get("data") or {},
            )
        return AgentResponseV6(answer="", status="interrupted", session_id=session_id, data=payload)

    @staticmethod
    def _resume_payload(resume: Any) -> dict[str, Any]:
        """处理恢复数据

        将恢复数据转换为统一的字典格式。

        Args:
            resume: 恢复数据

        Returns:
            处理后的恢复载荷
        """
        if hasattr(resume, "model_dump"):
            return resume.model_dump(exclude_none=True)
        if isinstance(resume, dict):
            return {key: value for key, value in resume.items() if value is not None}
        return {"value": resume}

    @staticmethod
    def _parse_project_id(raw: str | None) -> int | None:
        """解析项目ID

        将字符串形式的项目ID转换为整数。

        Args:
            raw: 原始项目ID字符串

        Returns:
            整数形式的项目ID，如果解析失败则返回None
        """
        if raw in (None, ""):
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _thread_id(project_id: int, session_id: str) -> str:
        """生成线程ID

        根据项目ID和会话ID生成唯一的线程ID，用于状态持久化。

        Args:
            project_id: 项目ID
            session_id: 会话ID

        Returns:
            线程ID字符串
        """
        return f"right-agent-v6:{project_id}:{session_id}"