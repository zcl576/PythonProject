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

PERMISSION_EXPIRED = "PERMISSION_EXPIRED"
RENEW_PERMISSION = "renew_permission"


class RightAgentState(TypedDict, total=False):
    session_id: str
    project_id: int
    question: str | None
    choice_id: str | None
    confirm_id: str | None
    intent: str
    slots: dict[str, Any]
    active_tool: str
    tool_result: dict[str, Any]
    status: str
    answer: str
    follow_up_question: str | None
    needs_input: list[str]
    choices: list[dict[str, Any]]
    confirm: dict[str, Any] | None
    data: dict[str, Any]


class RightAgentGraphService:
    def __init__(
        self,
        *,
        repository: AgentStateRepository | None = None,
        tools: RightToolsV2 | None = None,
    ) -> None:
        self._settings = get_settings()
        self._repository = repository or AgentStateRepository()
        self._tools = tools or RightToolsV2()
        self._graph = self._build_graph()

    async def do_execute(
        self, request: AgentRequestV2, project_id: str | None
    ) -> AgentResponseV2:
        effective_project_id = self._parse_project_id(project_id)
        session_id = request.session_id or uuid.uuid4().hex
        if effective_project_id is None:
            return AgentResponseV2(
                answer="缺少项目上下文，无法继续处理。",
                status="error",
                session_id=session_id,
                follow_up_question="请刷新页面或重新进入项目后再试。",
                needs_input=["Project-Id"],
            )

        await self._repository.ensure_thread(session_id, effective_project_id)
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
        builder = StateGraph(RightAgentState)
        builder.add_node("entry", self._entry)
        builder.add_node("understand", self._understand)
        builder.add_node("call_tool", self._call_tool)
        builder.add_node("handle_tool_result", self._handle_tool_result)
        builder.add_node("need_choice", self._need_choice)
        builder.add_node("continue_choice", self._continue_choice)
        builder.add_node("need_confirm", self._need_confirm)
        builder.add_node("execute_action", self._execute_action)
        builder.add_node("answer", self._answer)

        builder.add_edge(START, "entry")
        builder.add_conditional_edges(
            "entry",
            self._route_entry,
            {
                "continue_choice": "continue_choice",
                "execute_action": "execute_action",
                "understand": "understand",
            },
        )
        builder.add_conditional_edges(
            "understand",
            self._route_after_understand,
            {
                "answer": "answer",
                "call_tool": "call_tool",
            },
        )
        builder.add_edge("call_tool", "handle_tool_result")
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
        builder.add_conditional_edges(
            "continue_choice",
            self._route_after_continue_choice,
            {
                "call_tool": "call_tool",
                "answer": "answer",
            },
        )
        builder.add_edge("need_choice", END)
        builder.add_edge("need_confirm", END)
        builder.add_edge("execute_action", END)
        builder.add_edge("answer", END)
        return builder.compile()

    async def _entry(self, state: RightAgentState) -> RightAgentState:
        return {}

    def _route_entry(self, state: RightAgentState) -> str:
        if state.get("confirm_id"):
            return "execute_action"
        if state.get("choice_id"):
            return "continue_choice"
        return "understand"

    async def _understand(self, state: RightAgentState) -> RightAgentState:
        question = (state.get("question") or "").strip()
        if not question:
            return {
                "status": "need_more_info",
                "answer": "请先描述你要查询或诊断的问题。",
                "follow_up_question": "请提供人员姓名、手机号或设备名称。",
                "needs_input": ["question"],
            }

        slots = self._extract_slots(question)
        intent = self._classify_intent(question)
        if intent == "person_lookup":
            active_tool = "search_person"
        else:
            active_tool = "diagnose_access_issue"

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
        if state.get("status") == "need_more_info":
            return "answer"
        return "call_tool"

    async def _call_tool(self, state: RightAgentState) -> RightAgentState:
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
        people = self._extract_people(state.get("tool_result") or {})
        if len(people) > 1:
            return {
                "status": "need_choice",
                "choices": self._person_choices(people),
                "active_tool": state.get("active_tool") or "diagnose_access_issue",
            }
        if len(people) == 1 and state.get("active_tool") == "search_person":
            person = people[0]
            slots = {**(state.get("slots") or {}), **self._person_slots(person)}
            return {
                "slots": slots,
                "active_tool": "diagnose_access_issue",
                "tool_result": {},
            }

        diagnosis = self._extract_diagnosis(state.get("tool_result") or {})
        if diagnosis.get("mainCause") == PERMISSION_EXPIRED:
            return {"status": "need_confirm"}
        return {"status": "ok"}

    def _route_after_tool(self, state: RightAgentState) -> str:
        if state.get("status") == "need_choice":
            return "need_choice"
        if state.get("status") == "need_confirm":
            return "need_confirm"
        if state.get("active_tool") == "diagnose_access_issue" and not state.get("tool_result"):
            return "call_tool"
        return "answer"

    async def _need_choice(self, state: RightAgentState) -> RightAgentState:
        choices = state.get("choices") or []
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
        snapshot = await self._repository.load_snapshot(state["session_id"])
        if not snapshot or snapshot.get("state") != "waiting_choice":
            return {
                "status": "error",
                "answer": "当前没有待选择的操作，请重新发起查询。",
            }
        choice = self._find_choice(snapshot.get("choices") or [], state.get("choice_id"))
        if choice is None:
            return {"status": "error", "answer": "无效选择，请重新选择。"}

        slots = {**(snapshot.get("slots") or {}), **self._person_slots(choice.get("data") or {})}
        metadata = snapshot.get("metadata") or {}
        active_tool = metadata.get("active_tool") or "diagnose_access_issue"
        if active_tool == "search_person":
            await self._repository.clear_snapshot(state["session_id"], state="done")
            return {
                "slots": slots,
                "status": "ok",
                "answer": f"已选择：{choice.get('label', '该人员')}。",
                "data": {"person": choice.get("data") or {}},
            }
        return {
            "slots": slots,
            "active_tool": active_tool,
            "status": "",
        }

    def _route_after_continue_choice(self, state: RightAgentState) -> str:
        if state.get("status") in ("error", "ok") or state.get("answer"):
            return "answer"
        return "call_tool"

    async def _need_confirm(self, state: RightAgentState) -> RightAgentState:
        diagnosis = self._extract_diagnosis(state.get("tool_result") or {})
        confirm = {
            "id": RENEW_PERMISSION,
            "label": "确认续期",
            "description": "诊断结果为权限已过期，确认后将发起权限续期。",
            "risk_level": "medium",
        }
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
        snapshot = await self._repository.load_snapshot(state["session_id"])
        if not snapshot or snapshot.get("state") != "waiting_confirm":
            return {
                "status": "error",
                "answer": "当前没有待确认的操作，请重新发起诊断。",
            }
        if state.get("confirm_id") != RENEW_PERMISSION:
            return {"status": "error", "answer": "无效确认操作。"}
        if snapshot.get("pending_action") != RENEW_PERMISSION:
            return {"status": "error", "answer": "当前操作不允许续期。"}

        diagnosis = (snapshot.get("metadata") or {}).get("diagnosis") or {}
        if diagnosis.get("mainCause") != PERMISSION_EXPIRED:
            return {"status": "error", "answer": "当前诊断原因不允许续期。"}

        result = await self._tools.renew_permission(
            state["project_id"],
            snapshot.get("slots") or {},
        )
        await self._repository.clear_snapshot(state["session_id"], state="done")
        return {
            "status": "ok",
            "answer": "已发起权限续期申请，请等待处理结果。",
            "data": {"renew_result": result},
        }

    async def _answer(self, state: RightAgentState) -> RightAgentState:
        if state.get("answer"):
            return {}
        tool_result = state.get("tool_result") or {}
        diagnosis = self._extract_diagnosis(tool_result)
        if diagnosis:
            await self._repository.clear_snapshot(state["session_id"], state="done")
            main_name = diagnosis.get("mainCauseName") or diagnosis.get("mainCause") or "未知原因"
            summary = diagnosis.get("summary") or ""
            return {
                "status": "ok",
                "answer": f"诊断结果：{main_name}。{summary}".strip(),
                "data": {"diagnosis": diagnosis},
            }
        await self._repository.clear_snapshot(state["session_id"], state="done")
        return {
            "status": state.get("status") or "ok",
            "answer": "查询完成。",
            "data": {"result": tool_result},
        }

    @staticmethod
    def _parse_project_id(raw: str | None) -> int | None:
        if raw in (None, ""):
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _classify_intent(question: str) -> str:
        if any(word in question for word in ("找", "查询", "查一下", "叫")) and not any(
            word in question for word in ("打不开", "开不了", "进不去", "刷不开")
        ):
            return "person_lookup"
        return "access_issue"

    @staticmethod
    def _extract_slots(question: str) -> dict[str, Any]:
        slots: dict[str, Any] = {}
        phone = re.search(r"1[3-9]\d{9}", question)
        if phone:
            slots["telephone"] = phone.group(0)
        name = re.search(r"叫([\u4e00-\u9fa5]{2,4})", question)
        if not name:
            name = re.search(r"查询([\u4e00-\u9fa5]{2,4})", question)
        if not name:
            name = re.search(r"([\u4e00-\u9fa5]{2,4})(?:开不了|打不开|进不去|刷不开)", question)
        if name:
            slots["personName"] = name.group(1)
        device = re.search(r"([\u4e00-\u9fa5一二三四五六七八九十0-9]+号门)", question)
        if device:
            slots["deviceName"] = device.group(1)
        return slots

    @staticmethod
    def _extract_people(result: dict[str, Any]) -> list[dict[str, Any]]:
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
        diagnosis = result.get("diagnosis")
        return diagnosis if isinstance(diagnosis, dict) else {}

    @staticmethod
    def _person_choices(people: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        return {
            "personId": person.get("personId") or person.get("id"),
            "personName": person.get("personName") or person.get("name"),
            "telephone": person.get("telephone") or person.get("phone"),
        }

    @staticmethod
    def _find_choice(choices: list[dict[str, Any]], choice_id: str | None) -> dict[str, Any] | None:
        for choice in choices:
            if str(choice.get("id")) == str(choice_id):
                return choice
        return None

    def _to_response(self, state: dict[str, Any]) -> AgentResponseV2:
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
        prompt_path = os.path.join(self._settings.base_dir, "prompts", "right_agent_system_v2.md")
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read()
