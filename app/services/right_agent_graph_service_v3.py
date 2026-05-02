from __future__ import annotations

import os.path
import re
import uuid
from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.config import get_settings
from app.memory.langgraph_mysql_checkpoint_v3 import get_v3_mysql_checkpointer
from app.schemas.right_schema_v3 import (
    AgentChoiceV3,
    AgentConfirmV3,
    AgentRequestV3,
    AgentResponseV3,
)
from app.tools.right_tools_v3 import RightToolsV3

PERMISSION_EXPIRED = "PERMISSION_EXPIRED"
RENEW_PERMISSION = "renew_permission"
INTERRUPT_KEY = "__interrupt__"


class RightAgentStateV3(TypedDict, total=False):
    session_id: str
    project_id: int
    question: str | None
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


class RightAgentGraphServiceV3:
    def __init__(
        self,
        *,
        tools: RightToolsV3 | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self._settings = get_settings()
        self._tools = tools or RightToolsV3()
        self._checkpointer = checkpointer
        self._graph = None

    async def do_execute(
        self, request: AgentRequestV3, project_id: str | None
    ) -> AgentResponseV3:
        effective_project_id = self._parse_project_id(project_id)
        session_id = request.session_id or uuid.uuid4().hex
        if effective_project_id is None:
            return AgentResponseV3(
                answer="缺少项目上下文，无法继续处理。",
                status="error",
                session_id=session_id,
                follow_up_question="请刷新页面或重新进入项目后再试。",
                needs_input=["Project-Id"],
            )

        graph = await self._get_graph()
        config = {
            "configurable": {
                "thread_id": self._thread_id(effective_project_id, session_id)
            }
        }
        try:
            if request.resume is not None:
                state = await graph.ainvoke(
                    Command(resume=self._resume_payload(request.resume)),
                    config=config,
                )
            else:
                state = await graph.ainvoke(
                    {
                        "session_id": session_id,
                        "project_id": effective_project_id,
                        "question": request.question,
                        "slots": {},
                        "choices": [],
                        "data": {},
                        "needs_input": [],
                    },
                    config=config,
                )
        except ValueError as exc:
            return AgentResponseV3(
                answer="当前没有可继续的选择或确认，请重新发起查询。",
                status="error",
                session_id=session_id,
                data={"error": str(exc)},
            )
        return self._to_response(state, session_id)

    async def _get_graph(self):
        if self._graph is not None:
            return self._graph
        checkpointer = self._checkpointer or await get_v3_mysql_checkpointer()
        self._graph = self._build_graph().compile(checkpointer=checkpointer)
        return self._graph

    def _build_graph(self):
        builder = StateGraph(RightAgentStateV3)
        builder.add_node("understand", self._understand)
        builder.add_node("call_tool", self._call_tool)
        builder.add_node("handle_tool_result", self._handle_tool_result)
        builder.add_node("confirm_renew", self._confirm_renew)
        builder.add_node("answer", self._answer)

        builder.add_edge(START, "understand")
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
                "call_tool": "call_tool",
                "confirm_renew": "confirm_renew",
                "answer": "answer",
            },
        )
        builder.add_edge("confirm_renew", "answer")
        builder.add_edge("answer", END)
        return builder

    async def _understand(self, state: RightAgentStateV3) -> RightAgentStateV3:
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
        active_tool = "search_person" if intent == "person_lookup" else "diagnose_access_issue"
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

    def _route_after_understand(self, state: RightAgentStateV3) -> str:
        if state.get("status") == "need_more_info":
            return "answer"
        return "call_tool"

    async def _call_tool(self, state: RightAgentStateV3) -> RightAgentStateV3:
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

    async def _handle_tool_result(self, state: RightAgentStateV3) -> RightAgentStateV3:
        people = self._extract_people(state.get("tool_result") or {})
        if len(people) > 1:
            choices = self._person_choices(people)
            resume = interrupt(
                {
                    "type": "choice",
                    "answer": "找到多个匹配人员，请选择一个后继续。",
                    "follow_up_question": "请选择要查询的人员。",
                    "choices": choices,
                }
            )
            choice = self._choice_from_resume(resume, choices)
            if choice is None:
                return {"status": "error", "answer": "无效选择，请重新发起查询。"}
            return self._state_after_person_choice(state, choice)

        if len(people) == 1:
            choice = {"data": people[0], "label": self._person_label(people[0])}
            return self._state_after_person_choice(state, choice)

        diagnosis = self._extract_diagnosis(state.get("tool_result") or {})
        if diagnosis.get("mainCause") == PERMISSION_EXPIRED:
            return {"status": "need_confirm", "data": {"diagnosis": diagnosis}}
        if diagnosis:
            return {
                "status": "ok",
                "data": {"diagnosis": diagnosis},
            }
        return {
            "status": "ok",
            "data": {"result": state.get("tool_result") or {}},
        }

    def _state_after_person_choice(
        self,
        state: RightAgentStateV3,
        choice: dict[str, Any],
    ) -> RightAgentStateV3:
        person = choice.get("data") or {}
        slots = {**(state.get("slots") or {}), **self._person_slots(person)}
        if state.get("intent") == "person_lookup" and state.get("active_tool") == "search_person":
            return {
                "slots": slots,
                "status": "ok",
                "answer": f"已选择：{choice.get('label') or '该人员'}。",
                "data": {"person": person},
            }
        return {
            "slots": slots,
            "active_tool": "diagnose_access_issue",
            "tool_result": {},
            "status": "",
        }

    def _route_after_tool(self, state: RightAgentStateV3) -> str:
        if state.get("status") == "need_confirm":
            return "confirm_renew"
        if state.get("status") in ("error", "ok") or state.get("answer"):
            return "answer"
        if state.get("active_tool") == "diagnose_access_issue" and not state.get("tool_result"):
            return "call_tool"
        return "answer"

    async def _confirm_renew(self, state: RightAgentStateV3) -> RightAgentStateV3:
        diagnosis = (state.get("data") or {}).get("diagnosis") or {}
        confirm = {
            "id": RENEW_PERMISSION,
            "label": "确认续期",
            "description": "诊断结果为权限已过期，确认后将发起权限续期。",
            "risk_level": "medium",
        }
        resume = interrupt(
            {
                "type": "confirm",
                "answer": "诊断结果是权限已过期。是否需要发起权限续期？",
                "follow_up_question": "请确认是否续期。",
                "confirm": confirm,
                "data": {"diagnosis": diagnosis},
            }
        )
        if not self._is_renew_confirmed(resume):
            return {
                "status": "cancelled",
                "answer": "已取消权限续期。",
                "data": {"diagnosis": diagnosis},
            }
        if diagnosis.get("mainCause") != PERMISSION_EXPIRED:
            return {
                "status": "error",
                "answer": "当前诊断原因不允许续期。",
                "data": {"diagnosis": diagnosis},
            }

        result = await self._tools.renew_permission(
            state["project_id"],
            state.get("slots") or {},
        )
        return {
            "status": "ok",
            "answer": "已发起权限续期申请，请等待处理结果。",
            "data": {"diagnosis": diagnosis, "renew_result": result},
        }

    async def _answer(self, state: RightAgentStateV3) -> RightAgentStateV3:
        if state.get("answer"):
            return {}
        diagnosis = (state.get("data") or {}).get("diagnosis") or {}
        if diagnosis:
            main_name = diagnosis.get("mainCauseName") or diagnosis.get("mainCause") or "未知原因"
            summary = diagnosis.get("summary") or ""
            return {
                "status": state.get("status") or "ok",
                "answer": f"诊断结果：{main_name}。{summary}".strip(),
            }
        return {
            "status": state.get("status") or "ok",
            "answer": "查询完成。",
        }

    @staticmethod
    def _resume_payload(resume: Any) -> dict[str, Any]:
        if hasattr(resume, "model_dump"):
            return resume.model_dump(exclude_none=True)
        if isinstance(resume, dict):
            return {key: value for key, value in resume.items() if value is not None}
        return {"value": resume}

    @staticmethod
    def _parse_project_id(raw: str | None) -> int | None:
        if raw in (None, ""):
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _thread_id(project_id: int, session_id: str) -> str:
        return f"right-agent-v3:{project_id}:{session_id}"

    @staticmethod
    def _classify_intent(question: str) -> str:
        lookup_words = ("找", "查询", "查一下", "叫")
        issue_words = ("打不开", "开不了", "进不去", "刷不开")
        if any(word in question for word in lookup_words) and not any(
            word in question for word in issue_words
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

    @classmethod
    def _person_choices(cls, people: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

    @staticmethod
    def _person_label(person: dict[str, Any]) -> str:
        name = person.get("personName") or person.get("name") or "未知人员"
        phone = person.get("telephone") or person.get("phone") or ""
        return f"{name} {phone}".strip()

    @staticmethod
    def _person_slots(person: dict[str, Any]) -> dict[str, Any]:
        return {
            "personId": person.get("personId") or person.get("id"),
            "personName": person.get("personName") or person.get("name"),
            "telephone": person.get("telephone") or person.get("phone"),
        }

    @staticmethod
    def _choice_from_resume(
        resume: Any,
        choices: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not isinstance(resume, dict) or resume.get("type") != "choice":
            return None
        choice_id = resume.get("choice_id")
        for choice in choices:
            if str(choice.get("id")) == str(choice_id):
                return choice
        return None

    @staticmethod
    def _is_renew_confirmed(resume: Any) -> bool:
        return (
            isinstance(resume, dict)
            and resume.get("type") == "confirm"
            and resume.get("confirm_id") == RENEW_PERMISSION
            and resume.get("confirmed") is True
        )

    def _to_response(self, state: dict[str, Any], session_id: str) -> AgentResponseV3:
        interrupt_payload = self._interrupt_payload(state)
        if interrupt_payload:
            return self._interrupt_to_response(interrupt_payload, session_id)
        return AgentResponseV3(
            answer=state.get("answer") or "",
            status=state.get("status") or "ok",
            session_id=state.get("session_id") or session_id,
            follow_up_question=state.get("follow_up_question"),
            choices=[AgentChoiceV3.model_validate(item) for item in state.get("choices") or []],
            confirm=(
                AgentConfirmV3.model_validate(state["confirm"])
                if state.get("confirm")
                else None
            ),
            needs_input=state.get("needs_input") or [],
            data=state.get("data") or {},
        )

    @staticmethod
    def _interrupt_payload(state: dict[str, Any]) -> dict[str, Any] | None:
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
    ) -> AgentResponseV3:
        if payload.get("type") == "choice":
            return AgentResponseV3(
                answer=payload.get("answer") or "请选择一个选项后继续。",
                status="need_choice",
                session_id=session_id,
                follow_up_question=payload.get("follow_up_question"),
                choices=[
                    AgentChoiceV3.model_validate(item)
                    for item in payload.get("choices") or []
                ],
                data=payload.get("data") or {},
            )
        if payload.get("type") == "confirm":
            confirm = payload.get("confirm")
            return AgentResponseV3(
                answer=payload.get("answer") or "请确认是否继续。",
                status="need_confirm",
                session_id=session_id,
                follow_up_question=payload.get("follow_up_question"),
                confirm=AgentConfirmV3.model_validate(confirm) if confirm else None,
                data=payload.get("data") or {},
            )
        return AgentResponseV3(
            answer="流程已暂停，请提交续跑数据后继续。",
            status="interrupted",
            session_id=session_id,
            data=payload,
        )

    @lru_cache
    def _get_system_prompt(self) -> str:
        prompt_path = os.path.join(self._settings.base_dir, "prompts", "right_agent_system_v3.md")
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read()
