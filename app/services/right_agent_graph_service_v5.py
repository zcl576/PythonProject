"""权限代理图服务 V5 版本

本模块实现了基于 LangGraph 的权限查询代理服务，支持人员查询、设备查询、
权限查询和权限续期等功能。通过状态机驱动的方式，实现了对话流程的管理和
工具调用的编排。

主要特性：
- 基于 LangGraph 状态机实现对话流程控制
- 支持会话状态持久化（MySQL 检查点）
- 支持工具调用、槽位填充、用户确认等交互模式
- 集成 LLM 进行意图理解和最终回答生成
"""
from __future__ import annotations

import uuid
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.clients.llm_client import LlmClient
from app.config import get_settings
from app.memory.langgraph_mysql_checkpoint_v5 import get_v5_mysql_checkpointer
from app.schemas.right_schema_v5 import (
    AgentChoiceV5,
    AgentConfirmV5,
    AgentRequestV5,
    AgentResponseV5,
)
from app.services.right_agent_metadata_v5 import (
    PERMISSION_EXPIRED,
    PLANNER_TARGET_TO_TOOL_V5,
    READ_TARGET_TOOLS_V5,
    RENEW_PERMISSION,
    TOOL_METADATA_V5,
)
from app.tools.right_tools_v5 import RightToolsV5

# 中断键名，用于存储 LangGraph 中断信息
INTERRUPT_KEY = "__interrupt__"

# V5 版本支持的槽位名称（人员、设备相关字段）
SLOT_NAMES_V5 = (
    "personId",     # 人员ID
    "personName",   # 人员姓名
    "telephone",    # 电话号码
    "deviceSn",     # 设备序列号
    "deviceId",     # 设备ID
    "deviceName",   # 设备名称
    "cardNo",       # 卡号
)


class RightAgentStateV5(TypedDict, total=False):
    """权限代理状态类型定义

    用于存储 LangGraph 状态机执行过程中的所有状态信息。
    """
    session_id: str                    # 会话ID
    project_id: int                    # 项目ID
    question: str | None               # 用户问题
    intent: str | None                 # 意图识别结果
    target_tool: str                   # 目标工具（最终要执行的工具）
    current_tool: str                  # 当前工具（正在执行的工具）
    resolving_slot: str                # 当前正在解析的槽位
    slots: dict[str, Any]              # 已填充的槽位数据
    tool_result: dict[str, Any]        # 最近一次工具执行结果
    tool_history: list[dict[str, Any]] # 工具执行历史记录
    permission_result: dict[str, Any]  # 权限查询结果
    renew_result: dict[str, Any]       # 权限续期结果
    status: str                        # 当前状态（ok/error/need_more_info/need_confirm等）
    answer: str                        # 最终回答
    follow_up_question: str | None     # 追问问题
    needs_input: list[str]             # 需要用户输入的字段列表
    choices: list[dict[str, Any]]      # 供用户选择的选项列表
    confirm: dict[str, Any] | None     # 需要确认的操作
    data: dict[str, Any]               # 额外数据存储


class RightAgentGraphServiceV5:
    """权限代理图服务 V5 版本

    基于 LangGraph 构建的状态机服务，实现权限查询相关的对话流程。
    支持的工具包括：人员查询、设备查询、权限查询、权限续期。

    状态机节点说明：
    - understand_with_llm:    LLM 意图理解和槽位提取
    - normalize_plan:         计划规范化（意图转工具）
    - resolve_slots:          槽位解析（补全缺失信息）
    - execute_tool:           工具执行
    - handle_tool_result:     工具结果处理
    - confirm_renew:          续期确认
    - final_llm_answer:       最终回答生成
    """

    def __init__(
        self,
        *,
        tools: RightToolsV5 | None = None,
        llm: LlmClient | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        """初始化权限代理图服务

        Args:
            tools: 工具实例，用于执行具体的工具调用
            llm: LLM客户端，用于意图理解和回答生成
            checkpointer: 检查点存储，用于会话状态持久化
        """
        self._settings = get_settings()
        self._tools = tools or RightToolsV5()
        self._llm = llm or LlmClient()
        self._checkpointer = checkpointer
        self._graph = None  # 缓存的编译后图实例

    async def do_execute(
        self,
        request: AgentRequestV5,
        project_id: str | None,
    ) -> AgentResponseV5:
        """执行权限代理服务的主入口

        Args:
            request: 代理请求对象，包含用户问题、会话ID和恢复数据
            project_id: 项目ID（从请求头获取）

        Returns:
            AgentResponseV5: 代理响应对象
        """
        effective_project_id = self._parse_project_id(project_id)
        session_id = request.session_id or uuid.uuid4().hex

        # 验证项目ID
        if effective_project_id is None:
            return AgentResponseV5(
                answer="",
                status="error",
                session_id=session_id,
                follow_up_question="Project-Id is required.",
                needs_input=["Project-Id"],
            )

        # 获取编译后的状态机图
        graph = await self._get_graph()
        config = {
            "configurable": {
                "thread_id": self._thread_id(effective_project_id, session_id)
            }
        }

        try:
            # 根据是否有恢复数据选择不同的调用方式
            if request.resume is not None:
                # 恢复中断的会话
                state = await graph.ainvoke(
                    Command(resume=self._resume_payload(request.resume)),
                    config=config,
                )
            else:
                # 新会话，初始化状态
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
            return AgentResponseV5(
                answer="",
                status="error",
                session_id=session_id,
                data={"error": str(exc)},
            )

        return self._to_response(state, session_id)

    async def _get_graph(self):
        """获取编译后的状态机图（懒加载）

        Returns:
            编译后的 LangGraph 状态机实例
        """
        if self._graph is not None:
            return self._graph
        checkpointer = self._checkpointer or await get_v5_mysql_checkpointer()
        self._graph = self._build_graph().compile(checkpointer=checkpointer)
        return self._graph

    def _build_graph(self):
        """构建状态机图结构

        定义状态机的节点和边，构建完整的对话流程。

        Returns:
            StateGraph: 未编译的状态机构建器
        """
        builder = StateGraph(RightAgentStateV5)

        # 添加状态节点
        builder.add_node("understand_with_llm", self._understand_with_llm)  # LLM理解
        builder.add_node("normalize_plan", self._normalize_plan)            # 计划规范化
        builder.add_node("resolve_slots", self._resolve_slots)              # 槽位解析
        builder.add_node("execute_tool", self._execute_tool)                # 工具执行
        builder.add_node("handle_tool_result", self._handle_tool_result)    # 结果处理
        builder.add_node("confirm_renew", self._confirm_renew)              # 续期确认
        builder.add_node("final_llm_answer", self._final_llm_answer)        # 最终回答

        # 添加边（定义流程走向）
        builder.add_edge(START, "understand_with_llm")
        builder.add_edge("understand_with_llm", "normalize_plan")

        # 规范化后的路由：槽位解析 或 直接回答
        builder.add_conditional_edges(
            "normalize_plan",
            self._route_after_normalize,
            {"resolve_slots": "resolve_slots", "final_llm_answer": "final_llm_answer"},
        )

        # 槽位解析后的路由：执行工具 或 直接回答
        builder.add_conditional_edges(
            "resolve_slots",
            self._route_after_resolve_slots,
            {"execute_tool": "execute_tool", "final_llm_answer": "final_llm_answer"},
        )

        builder.add_edge("execute_tool", "handle_tool_result")

        # 工具结果处理后的路由：继续解析 / 续期确认 / 直接回答
        builder.add_conditional_edges(
            "handle_tool_result",
            self._route_after_tool_result,
            {
                "resolve_slots": "resolve_slots",
                "confirm_renew": "confirm_renew",
                "final_llm_answer": "final_llm_answer",
            },
        )

        # 续期确认后的路由：执行续期工具 或 直接回答
        builder.add_conditional_edges(
            "confirm_renew",
            self._route_after_confirm_renew,
            {"execute_tool": "execute_tool", "final_llm_answer": "final_llm_answer"},
        )

        builder.add_edge("final_llm_answer", END)
        return builder

    async def _understand_with_llm(self, state: RightAgentStateV5) -> RightAgentStateV5:
        """LLM意图理解节点

        使用LLM解析用户问题，提取意图、目标工具和槽位信息。

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
                "answer": "Please describe the access-control request.",
                "follow_up_question": "Please provide a person, device, or permission question.",
                "needs_input": ["question"],
            }

        try:
            # 调用LLM进行意图理解和槽位提取
            plan = await self._llm.understand_right_agent_v5(
                question=question,
                target_tools=tuple(TOOL_METADATA_V5.keys()),
                slot_names=SLOT_NAMES_V5,
            )
        except Exception as exc:
            return {
                "status": "error",
                "answer": "",
                "data": {"llm_error": str(exc), "stage": "planner"},
            }

        if not plan:
            return {
                "status": "error",
                "answer": "",
                "data": {"llm_error": "planner returned no structured plan"},
            }

        return {
            "intent": plan.get("intent"),
            "target_tool": plan.get("target_tool") or "",
            "slots": self._clean_slots(plan.get("slots") or {}),
            "data": {"planner": plan},
        }

    async def _normalize_plan(self, state: RightAgentStateV5) -> RightAgentStateV5:
        """计划规范化节点

        将意图转换为具体的工具名称，并处理特殊情况（如续期需要先查询）。

        Args:
            state: 当前状态

        Returns:
            更新后的状态，包含规范化后的目标工具
        """
        # 如果已有错误或需要更多信息，直接返回
        if state.get("status") in ("need_more_info", "error"):
            return {}

        target_tool = state.get("target_tool")

        # 如果没有目标工具，尝试从意图映射
        if not target_tool:
            intent = state.get("intent")
            target_tool = PLANNER_TARGET_TO_TOOL_V5.get(str(intent)) if intent else None

        data = state.get("data") or {}

        # 续期操作需要先查询权限状态，不能直接执行
        if target_tool == RENEW_PERMISSION:
            target_tool = "query_permission"
            data = {
                **data,
                "planner_rejected_tool": RENEW_PERMISSION,
                "planner_rejection_reason": "renew_permission requires permission query and human confirmation",
            }

        # 验证工具是否支持
        if target_tool not in READ_TARGET_TOOLS_V5:
            return {
                "status": "error",
                "answer": "",
                "data": {**data, "error": "unsupported target tool", "target_tool": target_tool},
            }

        return {
            "target_tool": target_tool,
            "current_tool": target_tool,
            "status": "",
            "data": data,
        }

    def _route_after_normalize(self, state: RightAgentStateV5) -> str:
        """规范化后的路由函数

        判断是继续槽位解析还是直接生成回答。

        Args:
            state: 当前状态

        Returns:
            下一个节点名称
        """
        if state.get("status") in ("need_more_info", "error"):
            return "final_llm_answer"
        return "resolve_slots"

    async def _resolve_slots(self, state: RightAgentStateV5) -> RightAgentStateV5:
        """槽位解析节点

        检查并补全工具执行所需的槽位信息。如果槽位缺失，尝试使用解析器工具获取，
        或向用户询问缺失的信息。

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        current_tool = state.get("current_tool") or state.get("target_tool")

        # 验证当前工具是否有效
        if not current_tool or current_tool not in TOOL_METADATA_V5:
            return {"status": "error", "answer": "", "data": {"error": "unknown current tool"}}

        metadata = TOOL_METADATA_V5[current_tool]
        slots = state.get("slots") or {}

        # 检查任意必填槽位（满足其一即可）
        missing_any = self._missing_any_required(metadata.any_required_slots, slots)
        if missing_any:
            return {
                "status": "need_more_info",
                "answer": "Missing information for the next tool.",
                "follow_up_question": f"Please provide one of: {', '.join(missing_any)}.",
                "needs_input": missing_any,
            }

        # 检查所有必填槽位
        for slot in metadata.required_slots:
            if slots.get(slot):
                continue

            # 尝试使用解析器工具获取槽位
            resolver = metadata.slot_resolvers.get(slot)
            if resolver:
                return {
                    "current_tool": resolver,
                    "resolving_slot": slot,
                    "status": "",
                    "tool_result": {},
                    "data": {
                        **(state.get("data") or {}),
                        "after_resolver_tool": state.get("target_tool"),
                    },
                }

            # 无法自动获取，向用户询问
            return {
                "status": "need_more_info",
                "answer": "Missing required information.",
                "follow_up_question": f"Please provide {slot}.",
                "needs_input": [slot],
            }

        # 所有槽位已填充
        return {"current_tool": current_tool, "status": ""}

    def _route_after_resolve_slots(self, state: RightAgentStateV5) -> str:
        """槽位解析后的路由函数

        Args:
            state: 当前状态

        Returns:
            下一个节点名称
        """
        if state.get("status") in ("need_more_info", "error"):
            return "final_llm_answer"
        return "execute_tool"

    async def _execute_tool(self, state: RightAgentStateV5) -> RightAgentStateV5:
        """工具执行节点

        调用实际的工具执行器执行当前工具。

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

    async def _handle_tool_result(self, state: RightAgentStateV5) -> RightAgentStateV5:
        """工具结果处理节点

        根据不同的工具类型处理执行结果，包括解析器结果、人员/设备查询结果、
        权限查询结果和续期结果。

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        result = state.get("tool_result") or {}
        current_tool = state["current_tool"]
        history = self._append_tool_history(state, current_tool, result)
        base_data = {**(state.get("data") or {}), "tool_history": history}

        # 检查工具是否配置
        if result.get("status") == "not_configured":
            return {
                "status": "error",
                "answer": "",
                "tool_history": history,
                "data": {**base_data, "result": result},
            }

        target_tool = state.get("target_tool") or current_tool

        # 如果当前工具是解析器（非目标工具），处理解析器结果
        if current_tool != target_tool:
            return self._handle_resolver_result(state, current_tool, result, history, base_data)

        # 根据工具类型处理结果
        if current_tool == "search_person":
            return self._handle_person_lookup_result(state, result, history, base_data)
        if current_tool == "search_device":
            return self._handle_device_lookup_result(state, result, history, base_data)
        if current_tool == "query_permission":
            return self._handle_permission_result(state, result, history, base_data)
        if current_tool == "renew_permission":
            return {
                "status": "ok",
                "renew_result": result,
                "tool_history": history,
                "data": {**base_data, "renew_result": result},
            }

        return {
            "status": "error",
            "answer": "",
            "tool_history": history,
            "data": {**base_data, "error": "unhandled tool result"},
        }

    def _handle_resolver_result(
        self,
        state: RightAgentStateV5,
        current_tool: str,
        result: dict[str, Any],
        history: list[dict[str, Any]],
        base_data: dict[str, Any],
    ) -> RightAgentStateV5:
        """处理解析器工具的结果

        解析器工具用于获取槽位信息（如通过姓名查询人员ID）。

        Args:
            state: 当前状态
            current_tool: 当前工具名称
            result: 工具执行结果
            history: 工具历史
            base_data: 基础数据

        Returns:
            更新后的状态
        """
        if current_tool == "search_person":
            # 处理人员搜索结果
            selected = self._select_candidate(
                kind="person",
                candidates=self._extract_people(result),
                choices=self._person_choices(self._extract_people(result)),
                empty_answer="No matching person was found.",
            )
            if selected.get("status"):
                return {**selected, "tool_history": history, "data": {**base_data, **(selected.get("data") or {})}}

            # 提取人员槽位并继续解析
            slots = {**(state.get("slots") or {}), **self._person_slots(selected["data"])}
            return {
                "slots": slots,
                "current_tool": state.get("target_tool") or "query_permission",
                "status": "continue_resolve",
                "tool_history": history,
                "data": base_data,
            }

        if current_tool == "search_device":
            # 处理设备搜索结果
            selected = self._select_candidate(
                kind="device",
                candidates=self._extract_devices(result),
                choices=self._device_choices(self._extract_devices(result)),
                empty_answer="No matching device was found.",
            )
            if selected.get("status"):
                return {**selected, "tool_history": history, "data": {**base_data, **(selected.get("data") or {})}}

            # 提取设备槽位并继续解析
            slots = {**(state.get("slots") or {}), **self._device_slots(selected["data"])}
            return {
                "slots": slots,
                "current_tool": state.get("target_tool") or "query_permission",
                "status": "continue_resolve",
                "tool_history": history,
                "data": base_data,
            }

        return {
            "status": "error",
            "answer": "",
            "tool_history": history,
            "data": {**base_data, "error": "resolver result is not supported"},
        }

    def _handle_person_lookup_result(
        self,
        state: RightAgentStateV5,
        result: dict[str, Any],
        history: list[dict[str, Any]],
        base_data: dict[str, Any],
    ) -> RightAgentStateV5:
        """处理人员查询工具的结果

        Args:
            state: 当前状态
            result: 工具执行结果
            history: 工具历史
            base_data: 基础数据

        Returns:
            更新后的状态
        """
        selected = self._select_candidate(
            kind="person",
            candidates=self._extract_people(result),
            choices=self._person_choices(self._extract_people(result)),
            empty_answer="No matching person was found.",
        )
        if selected.get("status"):
            return {**selected, "tool_history": history, "data": {**base_data, **(selected.get("data") or {})}}
        return {
            "status": "ok",
            "slots": {**(state.get("slots") or {}), **self._person_slots(selected["data"])},
            "tool_history": history,
            "data": {**base_data, "person": selected["data"]},
        }

    def _handle_device_lookup_result(
        self,
        state: RightAgentStateV5,
        result: dict[str, Any],
        history: list[dict[str, Any]],
        base_data: dict[str, Any],
    ) -> RightAgentStateV5:
        """处理设备查询工具的结果

        Args:
            state: 当前状态
            result: 工具执行结果
            history: 工具历史
            base_data: 基础数据

        Returns:
            更新后的状态
        """
        selected = self._select_candidate(
            kind="device",
            candidates=self._extract_devices(result),
            choices=self._device_choices(self._extract_devices(result)),
            empty_answer="No matching device was found.",
        )
        if selected.get("status"):
            return {**selected, "tool_history": history, "data": {**base_data, **(selected.get("data") or {})}}
        return {
            "status": "ok",
            "slots": {**(state.get("slots") or {}), **self._device_slots(selected["data"])},
            "tool_history": history,
            "data": {**base_data, "device": selected["data"]},
        }

    def _handle_permission_result(
        self,
        state: RightAgentStateV5,
        result: dict[str, Any],
        history: list[dict[str, Any]],
        base_data: dict[str, Any],
    ) -> RightAgentStateV5:
        """处理权限查询工具的结果

        检查权限是否过期，如果过期则需要用户确认续期。

        Args:
            state: 当前状态
            result: 工具执行结果
            history: 工具历史
            base_data: 基础数据

        Returns:
            更新后的状态
        """
        data = {**base_data, "permission_result": result}

        # 检查权限是否过期
        if self._is_permission_expired(result):
            return {
                "status": "need_confirm",
                "permission_result": result,
                "tool_history": history,
                "data": data,
            }

        return {
            "status": "ok",
            "permission_result": result,
            "tool_history": history,
            "data": data,
        }

    def _route_after_tool_result(self, state: RightAgentStateV5) -> str:
        """工具结果处理后的路由函数

        Args:
            state: 当前状态

        Returns:
            下一个节点名称
        """
        if state.get("status") == "continue_resolve":
            return "resolve_slots"
        if state.get("status") == "need_confirm":
            return "confirm_renew"
        return "final_llm_answer"

    async def _confirm_renew(self, state: RightAgentStateV5) -> RightAgentStateV5:
        permission_result = state.get("permission_result") or (state.get("data") or {}).get("permission_result") or {}
        if not self._is_permission_expired(permission_result):
            return {
                "status": "error",
                "answer": "",
                "data": {**(state.get("data") or {}), "error": "permission is not expired"},
            }
        confirm = {
            "id": RENEW_PERMISSION,
            "label": "Confirm permission renewal",
            "description": "Permission is expired. Confirm before renewal is executed.",
            "risk_level": "medium",
        }
        resume = interrupt(
            {
                "type": "confirm",
                "answer": "Permission is expired. Confirm renewal?",
                "follow_up_question": "Please confirm whether to renew the permission.",
                "confirm": confirm,
                "data": {"permission_result": permission_result},
            }
        )
        if not self._is_renew_confirmed(resume):
            return {
                "status": "cancelled",
                "data": {**(state.get("data") or {}), "permission_result": permission_result},
            }
        return {
            "current_tool": RENEW_PERMISSION,
            "target_tool": RENEW_PERMISSION,
            "status": "",
            "tool_result": {},
            "data": {**(state.get("data") or {}), "permission_result": permission_result},
        }

    def _route_after_confirm_renew(self, state: RightAgentStateV5) -> str:
        if state.get("status") in ("cancelled", "error"):
            return "final_llm_answer"
        return "execute_tool"

    async def _final_llm_answer(self, state: RightAgentStateV5) -> RightAgentStateV5:
        if state.get("answer"):
            return {}
        if state.get("status") in ("need_more_info", "error") and state.get("follow_up_question"):
            return {}
        try:
            answer = await self._llm.answer_right_agent_v5(
                question=state.get("question"),
                slots=state.get("slots") or {},
                tool_history=state.get("tool_history") or [],
                permission_result=state.get("permission_result") or (state.get("data") or {}).get("permission_result"),
                renew_result=state.get("renew_result") or (state.get("data") or {}).get("renew_result"),
            )
        except Exception as exc:
            return {
                "status": "error",
                "answer": "",
                "data": {**(state.get("data") or {}), "llm_error": str(exc), "stage": "final_answer"},
            }
        if not answer:
            return {
                "status": "error",
                "answer": "",
                "data": {
                    **(state.get("data") or {}),
                    "llm_error": "final answer generation returned empty content",
                    "stage": "final_answer",
                },
            }
        return {"status": state.get("status") or "ok", "answer": answer}

    @staticmethod
    def _append_tool_history(
        state: RightAgentStateV5,
        tool_name: str,
        result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """追加工具执行历史

        Args:
            state: 当前状态
            tool_name: 工具名称
            result: 工具执行结果

        Returns:
            更新后的工具历史列表
        """
        return [
            *(state.get("tool_history") or []),
            {"tool": tool_name, "result": result},
        ]

    @staticmethod
    def _missing_any_required(required: tuple[str, ...], slots: dict[str, Any]) -> list[str]:
        """检查是否缺少任意必填槽位

        如果必填槽位中有任意一个已填充，则返回空列表。

        Args:
            required: 必填槽位列表（满足其一即可）
            slots: 当前槽位数据

        Returns:
            缺少的槽位列表
        """
        if not required:
            return []
        if any(slots.get(slot) for slot in required):
            return []
        return list(required)

    @staticmethod
    def _clean_slots(slots: dict[str, Any]) -> dict[str, Any]:
        """清理槽位数据

        过滤掉不在允许列表中的槽位以及空值。

        Args:
            slots: 原始槽位数据

        Returns:
            清理后的槽位数据
        """
        return {
            key: value
            for key, value in slots.items()
            if key in SLOT_NAMES_V5 and value not in (None, "")
        }

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
        if isinstance(context, dict) and (
            context.get("deviceSn") or context.get("deviceId") or context.get("deviceName")
        ):
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
        choices = []
        for index, person in enumerate(people, start=1):
            person_id = str(person.get("personId") or person.get("id") or index)
            choices.append(
                {
                    "id": person_id,
                    "label": cls._person_label(person),
                    "description": person.get("description"),
                    "data": person,
                }
            )
        return choices

    @classmethod
    def _device_choices(cls, devices: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将设备列表转换为选择选项格式

        Args:
            devices: 设备列表

        Returns:
            选择选项列表
        """
        choices = []
        for index, device in enumerate(devices, start=1):
            device_id = str(device.get("deviceSn") or device.get("deviceId") or device.get("id") or index)
            choices.append(
                {
                    "id": device_id,
                    "label": cls._device_label(device),
                    "description": device.get("description"),
                    "data": device,
                }
            )
        return choices

    @staticmethod
    def _person_label(person: dict[str, Any]) -> str:
        """生成人员显示标签

        Args:
            person: 人员信息

        Returns:
            显示标签
        """
        name = person.get("personName") or person.get("name") or "Unknown person"
        phone = person.get("telephone") or person.get("phone") or ""
        return f"{name} {phone}".strip()

    @staticmethod
    def _device_label(device: dict[str, Any]) -> str:
        """生成设备显示标签

        Args:
            device: 设备信息

        Returns:
            显示标签
        """
        name = device.get("deviceName") or device.get("name") or "Unknown device"
        sn = device.get("deviceSn") or device.get("sn") or device.get("deviceId") or ""
        return f"{name} {sn}".strip()

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

    def _select_candidate(
        self,
        *,
        kind: str,
        candidates: list[dict[str, Any]],
        choices: list[dict[str, Any]],
        empty_answer: str,
    ) -> dict[str, Any]:
        """选择候选对象

        如果没有候选结果，返回错误；如果只有一个候选，直接使用；
        如果有多个候选，中断等待用户选择。

        Args:
            kind: 类型（person/device）
            candidates: 候选列表
            choices: 选择选项列表
            empty_answer: 空结果时的提示信息

        Returns:
            选择结果
        """
        if not candidates:
            return {"status": "need_more_info", "answer": empty_answer, "data": {"result": []}}
        if len(candidates) == 1:
            return {"data": candidates[0]}
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
    def _choice_from_resume(
        resume: Any,
        choices: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """从恢复数据中获取用户选择

        Args:
            resume: 恢复数据
            choices: 可用选项列表

        Returns:
            用户选择的选项，如果无效则返回None
        """
        if not isinstance(resume, dict) or resume.get("type") != "choice":
            return None
        choice_id = resume.get("choice_id")
        for choice in choices:
            if str(choice.get("id")) == str(choice_id):
                return choice
        return None

    @staticmethod
    def _is_permission_expired(result: dict[str, Any]) -> bool:
        """检查权限是否过期

        从多个可能的字段中检测权限过期状态。

        Args:
            result: 权限查询结果

        Returns:
            True表示过期，False表示未过期
        """
        diagnosis = result.get("diagnosis") if isinstance(result.get("diagnosis"), dict) else {}
        permission = result.get("permission") if isinstance(result.get("permission"), dict) else {}
        values = {
            result.get("mainCause"),
            result.get("status"),
            result.get("permissionStatus"),
            diagnosis.get("mainCause"),
            diagnosis.get("status"),
            permission.get("mainCause"),
            permission.get("status"),
            permission.get("permissionStatus"),
        }
        return any(str(value).upper() in {PERMISSION_EXPIRED, "EXPIRED"} for value in values if value)

    @staticmethod
    def _is_renew_confirmed(resume: Any) -> bool:
        """检查用户是否确认续期

        Args:
            resume: 恢复数据

        Returns:
            True表示已确认，False表示未确认
        """
        return (
            isinstance(resume, dict)
            and resume.get("type") == "confirm"
            and resume.get("confirm_id") == RENEW_PERMISSION
            and resume.get("confirmed") is True
        )

    def _to_response(self, state: dict[str, Any], session_id: str) -> AgentResponseV5:
        """将状态转换为响应对象

        Args:
            state: 当前状态
            session_id: 会话ID

        Returns:
            代理响应对象
        """
        interrupt_payload = self._interrupt_payload(state)
        if interrupt_payload:
            return self._interrupt_to_response(interrupt_payload, session_id)
        return AgentResponseV5(
            answer=state.get("answer") or "",
            status=state.get("status") or "ok",
            session_id=state.get("session_id") or session_id,
            follow_up_question=state.get("follow_up_question"),
            choices=[AgentChoiceV5.model_validate(item) for item in state.get("choices") or []],
            confirm=(
                AgentConfirmV5.model_validate(state["confirm"])
                if state.get("confirm")
                else None
            ),
            needs_input=state.get("needs_input") or [],
            data=state.get("data") or {},
        )

    @staticmethod
    def _interrupt_payload(state: dict[str, Any]) -> dict[str, Any] | None:
        """从状态中提取中断载荷

        Args:
            state: 当前状态

        Returns:
            中断载荷，如果没有中断则返回None
        """
        interrupts = state.get(INTERRUPT_KEY)
        if not interrupts:
            return None
        first = interrupts[0] if isinstance(interrupts, (list, tuple)) else interrupts
        value = getattr(first, "value", first)
        return value if isinstance(value, dict) else {"value": value}

    def _interrupt_to_response(
        self,
        payload: dict[str, Any],
        session_id: str,
    ) -> AgentResponseV5:
        """将中断载荷转换为响应对象

        Args:
            payload: 中断载荷
            session_id: 会话ID

        Returns:
            代理响应对象
        """
        if payload.get("type") == "choice":
            return AgentResponseV5(
                answer=payload.get("answer") or "",
                status="need_choice",
                session_id=session_id,
                follow_up_question=payload.get("follow_up_question"),
                choices=[
                    AgentChoiceV5.model_validate(item)
                    for item in payload.get("choices") or []
                ],
                data=payload.get("data") or {},
            )
        if payload.get("type") == "confirm":
            confirm = payload.get("confirm")
            return AgentResponseV5(
                answer=payload.get("answer") or "",
                status="need_confirm",
                session_id=session_id,
                follow_up_question=payload.get("follow_up_question"),
                confirm=AgentConfirmV5.model_validate(confirm) if confirm else None,
                data=payload.get("data") or {},
            )
        return AgentResponseV5(
            answer="",
            status="interrupted",
            session_id=session_id,
            data=payload,
        )

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

        用于LangGraph状态持久化的唯一标识。

        Args:
            project_id: 项目ID
            session_id: 会话ID

        Returns:
            线程ID字符串
        """
        return f"right-agent-v5:{project_id}:{session_id}"