"""权限代理图服务 V2 版本

本模块实现了基于 LangGraph 的权限查询代理服务，支持人员查询和权限诊断功能。
通过状态机驱动的方式，实现了对话流程的管理和工具调用的编排。

主要特性：
- 基于 LangGraph 状态机实现对话流程控制
- 支持会话状态持久化（通过 AgentStateRepository）
- 支持人员查询、权限诊断、权限续期等功能
- 支持多人员选择和续期确认等交互模式
"""
from __future__ import annotations

import os.path
import re
import uuid
from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.repositories.agent_state_repository import AgentStateRepository
from app.schemas.right_schema_v2 import (
    AgentChoice,
    AgentConfirm,
    AgentRequestV2,
    AgentResponseV2,
)
from app.tools.right_tools_v2 import RightToolsV2

# 权限过期状态标识
PERMISSION_EXPIRED = "PERMISSION_EXPIRED"
# 续期权限工具名称
RENEW_PERMISSION = "renew_permission"


class RightAgentState(TypedDict, total=False):
    """权限代理状态类型定义

    用于存储 LangGraph 状态机执行过程中的所有状态信息。
    """
    session_id: str                    # 会话ID
    project_id: int                    # 项目ID
    question: str | None               # 用户问题
    choice_id: str | None              # 用户选择的ID（用于恢复选择）
    confirm_id: str | None             # 用户确认的ID（用于恢复确认）
    intent: str                        # 意图识别结果
    slots: dict[str, Any]              # 已填充的槽位数据
    active_tool: str                   # 当前激活的工具
    tool_result: dict[str, Any]        # 工具执行结果
    status: str                        # 当前状态
    answer: str                        # 最终回答
    follow_up_question: str | None     # 追问问题
    needs_input: list[str]             # 需要用户输入的字段列表
    choices: list[dict[str, Any]]      # 供用户选择的选项列表
    confirm: dict[str, Any] | None     # 需要确认的操作
    data: dict[str, Any]               # 额外数据存储


class RightAgentGraphService:
    """权限代理图服务 V2 版本

    基于 LangGraph 构建的状态机服务，实现权限查询相关的对话流程。
    支持的工具包括：人员查询、权限诊断、权限续期。

    状态机节点说明：
    - entry:              入口节点
    - understand:         意图理解和槽位提取
    - call_tool:          工具执行
    - handle_tool_result: 工具结果处理
    - need_choice:        需要用户选择（多人员匹配时）
    - continue_choice:    继续选择（用户已选择后）
    - need_confirm:       需要用户确认（权限过期续期）
    - execute_action:     执行操作（续期确认后）
    - answer:             最终回答生成
    """

    def __init__(
        self,
        *,
        repository: AgentStateRepository | None = None,
        tools: RightToolsV2 | None = None,
    ) -> None:
        """初始化权限代理图服务

        Args:
            repository: 状态仓库，用于存储会话快照
            tools: 工具实例，用于执行具体的工具调用
        """
        self._settings = get_settings()
        self._repository = repository or AgentStateRepository()
        self._tools = tools or RightToolsV2()
        self._graph = self._build_graph()

    async def do_execute(
        self, request: AgentRequestV2, project_id: str | None
    ) -> AgentResponseV2:
        """执行权限代理服务的主入口

        Args:
            request: 代理请求对象，包含用户问题、会话ID、选择ID和确认ID
            project_id: 项目ID（从请求头获取）

        Returns:
            AgentResponseV2: 代理响应对象
        """
        effective_project_id = self._parse_project_id(project_id)
        session_id = request.session_id or uuid.uuid4().hex

        # 验证项目ID
        if effective_project_id is None:
            return AgentResponseV2(
                answer="缺少项目上下文，无法继续处理。",
                status="error",
                session_id=session_id,
                follow_up_question="请刷新页面或重新进入项目后再试。",
                needs_input=["Project-Id"],
            )

        # 确保会话线程存在
        await self._repository.ensure_thread(session_id, effective_project_id)

        # 调用状态机执行
        state = await self._graph.ainvoke(
            {
                "session_id": session_id,
                "project_id": effective_project_id,
                "question": request.question,
                "choice_id": request.choice_id,
                "confirm_id": request.confirm_id,
                "slots": {},
                "choices": [],
                "data": {},
                "needs_input": [],
            },
            config={"configurable": {"thread_id": session_id}},
        )

        return self._to_response(state)

    def _build_graph(self):
        """构建状态机图结构

        定义状态机的节点和边，构建完整的对话流程。

        Returns:
            编译后的 LangGraph 状态机实例
        """
        builder = StateGraph(RightAgentState)

        # 添加状态节点
        builder.add_node("entry", self._entry)                    # 入口节点
        builder.add_node("understand", self._understand)          # 意图理解
        builder.add_node("call_tool", self._call_tool)            # 工具执行
        builder.add_node("handle_tool_result", self._handle_tool_result)  # 结果处理
        builder.add_node("need_choice", self._need_choice)        # 需要选择
        builder.add_node("continue_choice", self._continue_choice)# 继续选择
        builder.add_node("need_confirm", self._need_confirm)      # 需要确认
        builder.add_node("execute_action", self._execute_action)  # 执行操作
        builder.add_node("answer", self._answer)                  # 最终回答

        # 添加边（定义流程走向）
        builder.add_edge(START, "entry")

        # 入口路由：继续选择 / 执行操作 / 理解
        builder.add_conditional_edges(
            "entry",
            self._route_entry,
            {
                "continue_choice": "continue_choice",
                "execute_action": "execute_action",
                "understand": "understand",
            },
        )

        # 理解后的路由：回答 / 调用工具
        builder.add_conditional_edges(
            "understand",
            self._route_after_understand,
            {
                "answer": "answer",
                "call_tool": "call_tool",
            },
        )

        builder.add_edge("call_tool", "handle_tool_result")

        # 工具结果后的路由：需要选择 / 需要确认 / 调用工具 / 回答
        builder.add_conditional_edges(
            "handle_tool_result",
            self._route_after_tool,
            {
                "need_choice": "need_choice",
                "need_confirm": "need_confirm",
                "call_tool": "call_tool",
                "answer": "answer",
            },
        )

        # 继续选择后的路由：调用工具 / 回答
        builder.add_conditional_edges(
            "continue_choice",
            self._route_after_continue_choice,
            {
                "call_tool": "call_tool",
                "answer": "answer",
            },
        )

        # 终端节点
        builder.add_edge("need_choice", END)
        builder.add_edge("need_confirm", END)
        builder.add_edge("execute_action", END)
        builder.add_edge("answer", END)

        return builder.compile()

    async def _entry(self, state: RightAgentState) -> RightAgentState:
        """入口节点

        空操作节点，仅用于路由判断。

        Args:
            state: 当前状态

        Returns:
            空状态更新
        """
        return {}

    def _route_entry(self, state: RightAgentState) -> str:
        """入口路由函数

        根据是否有 confirm_id 或 choice_id 决定下一步。

        Args:
            state: 当前状态

        Returns:
            下一个节点名称
        """
        if state.get("confirm_id"):
            return "execute_action"
        if state.get("choice_id"):
            return "continue_choice"
        return "understand"

    async def _understand(self, state: RightAgentState) -> RightAgentState:
        """意图理解节点

        使用规则引擎解析用户问题，提取意图和槽位信息。

        Args:
            state: 当前状态

        Returns:
            更新后的状态，包含意图、槽位和活动工具
        """
        question = (state.get("question") or "").strip()

        # 检查问题是否为空
        if not question:
            return {
                "status": "need_more_info",
                "answer": "请先描述你要查询或诊断的问题。",
                "follow_up_question": "请提供人员姓名、手机号或设备名称。",
                "needs_input": ["question"],
            }

        # 提取槽位和意图
        slots = self._extract_slots(question)
        intent = self._classify_intent(question)
        active_tool = "search_person" if intent == "person_lookup" else "diagnose_access_issue"

        # 检查是否有足够的信息
        if not slots:
            return {
                "intent": intent,
                "slots": {},
                "status": "need_more_info",
                "answer": "我还需要人员或设备信息才能继续处理。",
                "follow_up_question": "请提供姓名、手机号、卡号、设备名称或设备 SN。",
                "needs_input": ["personName", "telephone", "deviceName"],
            }

        return {"intent": intent, "slots": slots, "active_tool": active_tool}

    def _route_after_understand(self, state: RightAgentState) -> str:
        """意图理解后的路由函数

        Args:
            state: 当前状态

        Returns:
            下一个节点名称
        """
        if state.get("status") == "need_more_info":
            return "answer"
        return "call_tool"

    async def _call_tool(self, state: RightAgentState) -> RightAgentState:
        """工具执行节点

        根据活动工具调用对应的工具方法。

        Args:
            state: 当前状态

        Returns:
            更新后的状态，包含工具执行结果
        """
        project_id = state["project_id"]
        slots = state.get("slots") or {}
        active_tool = state.get("active_tool")

        if active_tool == "search_person":
            result = await self._tools.search_person(
                project_id,
                telephone=slots.get("telephone"),
                person_name=slots.get("personName"),
            )
        else:
            result = await self._tools.diagnose_access_issue(project_id, slots)

        return {"tool_result": result}

    async def _handle_tool_result(self, state: RightAgentState) -> RightAgentState:
        """工具结果处理节点

        根据工具执行结果进行处理，包括多人员选择、权限诊断等。

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        people = self._extract_people(state.get("tool_result") or {})

        # 处理多人员匹配情况
        if len(people) > 1:
            return {
                "status": "need_choice",
                "choices": self._person_choices(people),
                "active_tool": state.get("active_tool") or "diagnose_access_issue",
            }

        # 处理单人员查询结果（需要继续诊断）
        if len(people) == 1 and state.get("active_tool") == "search_person":
            person = people[0]
            slots = {**(state.get("slots") or {}), **self._person_slots(person)}
            return {
                "slots": slots,
                "active_tool": "diagnose_access_issue",
                "tool_result": {},
            }

        # 处理诊断结果
        diagnosis = self._extract_diagnosis(state.get("tool_result") or {})
        if diagnosis.get("mainCause") == PERMISSION_EXPIRED:
            return {"status": "need_confirm"}

        return {"status": "ok"}

    def _route_after_tool(self, state: RightAgentState) -> str:
        """工具结果处理后的路由函数

        Args:
            state: 当前状态

        Returns:
            下一个节点名称
        """
        if state.get("status") == "need_choice":
            return "need_choice"
        if state.get("status") == "need_confirm":
            return "need_confirm"
        if state.get("active_tool") == "diagnose_access_issue" and not state.get("tool_result"):
            return "call_tool"
        return "answer"

    async def _need_choice(self, state: RightAgentState) -> RightAgentState:
        """需要选择节点

        当查询到多个匹配人员时，保存快照并等待用户选择。

        Args:
            state: 当前状态

        Returns:
            更新后的状态，包含选择选项
        """
        choices = state.get("choices") or []

        # 保存会话快照，等待用户选择
        await self._repository.save_snapshot(
            session_id=state["session_id"],
            state="waiting_choice",
            pending_action="select_person",
            slots=state.get("slots") or {},
            choices=choices,
            metadata={
                "active_tool": state.get("active_tool") or "diagnose_access_issue",
                "question": state.get("question"),
            },
        )

        return {
            "answer": "找到多个匹配人员，请选择一个后继续。",
            "status": "need_choice",
            "follow_up_question": "请选择要查询的人员。",
            "choices": choices,
        }

    async def _continue_choice(self, state: RightAgentState) -> RightAgentState:
        """继续选择节点

        用户选择后恢复会话，加载快照并继续处理。

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        snapshot = await self._repository.load_snapshot(state["session_id"])

        # 验证快照是否有效
        if not snapshot or snapshot.get("state") != "waiting_choice":
            return {
                "status": "error",
                "answer": "当前没有待选择的操作，请重新发起查询。",
            }

        # 查找用户选择的选项
        choice = self._find_choice(snapshot.get("choices") or [], state.get("choice_id"))
        if choice is None:
            return {"status": "error", "answer": "无效选择，请重新选择。"}

        # 更新槽位并继续处理
        slots = {**(snapshot.get("slots") or {}), **self._person_slots(choice.get("data") or {})}
        metadata = snapshot.get("metadata") or {}
        active_tool = metadata.get("active_tool") or "diagnose_access_issue"

        # 如果只是人员查询，直接返回结果
        if active_tool == "search_person":
            await self._repository.clear_snapshot(state["session_id"], state="done")
            return {
                "slots": slots,
                "status": "ok",
                "answer": f"已选择：{choice.get('label', '该人员')}。",
                "data": {"person": choice.get("data") or {}},
            }

        # 继续诊断流程
        return {
            "slots": slots,
            "active_tool": active_tool,
            "status": "",
        }

    def _route_after_continue_choice(self, state: RightAgentState) -> str:
        """继续选择后的路由函数

        Args:
            state: 当前状态

        Returns:
            下一个节点名称
        """
        if state.get("status") in ("error", "ok") or state.get("answer"):
            return "answer"
        return "call_tool"

    async def _need_confirm(self, state: RightAgentState) -> RightAgentState:
        """需要确认节点

        当诊断结果为权限过期时，保存快照并等待用户确认续期。

        Args:
            state: 当前状态

        Returns:
            更新后的状态，包含确认信息
        """
        diagnosis = self._extract_diagnosis(state.get("tool_result") or {})

        # 构建确认信息
        confirm = {
            "id": RENEW_PERMISSION,
            "label": "确认续期",
            "description": "诊断结果为权限已过期，确认后将发起权限续期。",
            "risk_level": "medium",
        }

        # 保存会话快照，等待用户确认
        await self._repository.save_snapshot(
            session_id=state["session_id"],
            state="waiting_confirm",
            pending_action=RENEW_PERMISSION,
            slots=state.get("slots") or {},
            confirm=confirm,
            metadata={"diagnosis": diagnosis},
        )

        return {
            "status": "need_confirm",
            "answer": "诊断结果是权限已过期。是否需要发起权限续期？",
            "follow_up_question": "请确认是否续期。",
            "confirm": confirm,
            "data": {"diagnosis": diagnosis},
        }

    async def _execute_action(self, state: RightAgentState) -> RightAgentState:
        """执行操作节点

        用户确认后恢复会话，执行权限续期操作。

        Args:
            state: 当前状态

        Returns:
            更新后的状态，包含续期结果
        """
        snapshot = await self._repository.load_snapshot(state["session_id"])

        # 验证快照是否有效
        if not snapshot or snapshot.get("state") != "waiting_confirm":
            return {
                "status": "error",
                "answer": "当前没有待确认的操作，请重新发起诊断。",
            }

        # 验证确认操作
        if state.get("confirm_id") != RENEW_PERMISSION:
            return {"status": "error", "answer": "无效确认操作。"}
        if snapshot.get("pending_action") != RENEW_PERMISSION:
            return {"status": "error", "answer": "当前操作不允许续期。"}

        # 验证诊断结果
        diagnosis = (snapshot.get("metadata") or {}).get("diagnosis") or {}
        if diagnosis.get("mainCause") != PERMISSION_EXPIRED:
            return {"status": "error", "answer": "当前诊断原因不允许续期。"}

        # 执行权限续期
        result = await self._tools.renew_permission(
            state["project_id"],
            snapshot.get("slots") or {},
        )

        # 清理快照
        await self._repository.clear_snapshot(state["session_id"], state="done")

        return {
            "status": "ok",
            "answer": "已发起权限续期申请，请等待处理结果。",
            "data": {"renew_result": result},
        }

    async def _answer(self, state: RightAgentState) -> RightAgentState:
        """最终回答节点

        生成最终回答，清理会话快照。

        Args:
            state: 当前状态

        Returns:
            更新后的状态，包含最终回答
        """
        # 如果已有回答，直接返回
        if state.get("answer"):
            return {}

        tool_result = state.get("tool_result") or {}
        diagnosis = self._extract_diagnosis(tool_result)

        # 如果有诊断结果，生成诊断回答
        if diagnosis:
            await self._repository.clear_snapshot(state["session_id"], state="done")
            main_name = diagnosis.get("mainCauseName") or diagnosis.get("mainCause") or "未知原因"
            summary = diagnosis.get("summary") or ""
            return {
                "status": "ok",
                "answer": f"诊断结果：{main_name}。{summary}".strip(),
                "data": {"diagnosis": diagnosis},
            }

        # 清理快照并返回默认回答
        await self._repository.clear_snapshot(state["session_id"], state="done")
        return {
            "status": state.get("status") or "ok",
            "answer": "查询完成。",
            "data": {"result": tool_result},
        }

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
    def _classify_intent(question: str) -> str:
        """意图分类

        根据用户问题判断意图是人员查询还是权限问题。

        Args:
            question: 用户问题

        Returns:
            "person_lookup" 表示人员查询，"access_issue" 表示权限问题
        """
        if any(word in question for word in ("找", "查询", "查一下", "叫")) and not any(
            word in question for word in ("打不开", "开不了", "进不去", "刷不开")
        ):
            return "person_lookup"
        return "access_issue"

    @staticmethod
    def _extract_slots(question: str) -> dict[str, Any]:
        """从问题中提取槽位

        使用正则表达式提取手机号、姓名、设备名称等信息。

        Args:
            question: 用户问题

        Returns:
            槽位字典
        """
        slots: dict[str, Any] = {}

        # 提取手机号
        phone = re.search(r"1[3-9]\d{9}", question)
        if phone:
            slots["telephone"] = phone.group(0)

        # 提取姓名（支持多种模式）
        name = re.search(r"叫([\u4e00-\u9fa5]{2,4})", question)
        if not name:
            name = re.search(r"查询([\u4e00-\u9fa5]{2,4})", question)
        if not name:
            name = re.search(r"([\u4e00-\u9fa5]{2,4})(?:开不了|打不开|进不去|刷不开)", question)
        if name:
            slots["personName"] = name.group(1)

        # 提取设备名称（如"1号门"）
        device = re.search(r"([\u4e00-\u9fa5一二三四五六七八九十0-9]+号门)", question)
        if device:
            slots["deviceName"] = device.group(1)

        return slots

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
    def _extract_diagnosis(result: dict[str, Any]) -> dict[str, Any]:
        """从工具结果中提取诊断信息

        Args:
            result: 工具执行结果

        Returns:
            诊断信息字典
        """
        diagnosis = result.get("diagnosis")
        return diagnosis if isinstance(diagnosis, dict) else {}

    @staticmethod
    def _person_choices(people: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将人员列表转换为选择选项格式

        Args:
            people: 人员列表

        Returns:
            选择选项列表
        """
        choices = []
        for index, person in enumerate(people, start=1):
            person_id = str(person.get("personId") or person.get("id") or index)
            name = person.get("personName") or person.get("name") or "未知人员"
            phone = person.get("telephone") or person.get("phone") or ""
            choices.append(
                {
                    "id": person_id,
                    "label": f"{name} {phone}".strip(),
                    "description": person.get("description"),
                    "data": person,
                }
            )
        return choices

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
    def _find_choice(choices: list[dict[str, Any]], choice_id: str | None) -> dict[str, Any] | None:
        """从选项列表中查找指定ID的选项

        Args:
            choices: 选项列表
            choice_id: 选项ID

        Returns:
            匹配的选项，如果未找到则返回None
        """
        for choice in choices:
            if str(choice.get("id")) == str(choice_id):
                return choice
        return None

    def _to_response(self, state: dict[str, Any]) -> AgentResponseV2:
        """将状态转换为响应对象

        Args:
            state: 当前状态

        Returns:
            AgentResponseV2: 代理响应对象
        """
        return AgentResponseV2(
            answer=state.get("answer") or "",
            status=state.get("status") or "ok",
            session_id=state["session_id"],
            follow_up_question=state.get("follow_up_question"),
            choices=[AgentChoice.model_validate(item) for item in state.get("choices") or []],
            confirm=(
                AgentConfirm.model_validate(state["confirm"])
                if state.get("confirm")
                else None
            ),
            needs_input=state.get("needs_input") or [],
            data=state.get("data") or {},
        )

    @lru_cache
    def _get_system_prompt(self) -> str:
        """获取系统提示词

        从文件加载系统提示词，使用缓存避免重复读取。

        Returns:
            系统提示词内容
        """
        prompt_path = os.path.join(self._settings.base_dir, "prompts", "right_agent_system_v2.md")
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read()