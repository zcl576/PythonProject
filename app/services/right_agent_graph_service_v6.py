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

INTERRUPT_KEY = "__interrupt__"
SLOT_NAMES_V6 = (
    "personId",
    "personName",
    "telephone",
    "deviceSn",
    "deviceId",
    "deviceName",
    "cardNo",
    "durationDays",
)


class RightAgentStateV6(TypedDict, total=False):
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
    def __init__(
        self,
        *,
        tools: RightToolsV6 | None = None,
        llm: LlmClient | None = None,
        policy: RightAgentPolicyCheckerV6 | None = None,
        checkpointer: Any | None = None,
    ) -> None:
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
        effective_project_id = self._parse_project_id(project_id)
        session_id = request.session_id or uuid.uuid4().hex
        if effective_project_id is None:
            return AgentResponseV6(
                answer="",
                status="error",
                session_id=session_id,
                follow_up_question="Project-Id is required.",
                needs_input=["Project-Id"],
            )

        graph = await self._get_graph()
        config = {"configurable": {"thread_id": self._thread_id(effective_project_id, session_id)}}
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
                        "tool_history": [],
                    },
                    config=config,
                )
        except ValueError as exc:
            return AgentResponseV6(answer="", status="error", session_id=session_id, data={"error": str(exc)})
        return self._to_response(state, session_id)

    async def _get_graph(self):
        if self._graph is not None:
            return self._graph
        checkpointer = self._checkpointer or await get_v6_mysql_checkpointer()
        self._graph = self._build_graph().compile(checkpointer=checkpointer)
        return self._graph

    def _build_graph(self):
        builder = StateGraph(RightAgentStateV6)
        builder.add_node("understand_with_llm", self._understand_with_llm)
        builder.add_node("normalize_plan", self._normalize_plan)
        builder.add_node("resolve_slots", self._resolve_slots)
        builder.add_node("execute_tool", self._execute_tool)
        builder.add_node("handle_tool_result", self._handle_tool_result)
        builder.add_node("confirm_write", self._confirm_write)
        builder.add_node("final_llm_answer", self._final_llm_answer)

        builder.add_edge(START, "understand_with_llm")
        builder.add_edge("understand_with_llm", "normalize_plan")
        builder.add_conditional_edges(
            "normalize_plan",
            self._route_after_normalize,
            {"resolve_slots": "resolve_slots", "final_llm_answer": "final_llm_answer"},
        )
        builder.add_conditional_edges(
            "resolve_slots",
            self._route_after_resolve_slots,
            {"execute_tool": "execute_tool", "final_llm_answer": "final_llm_answer"},
        )
        builder.add_edge("execute_tool", "handle_tool_result")
        builder.add_conditional_edges(
            "handle_tool_result",
            self._route_after_tool_result,
            {
                "resolve_slots": "resolve_slots",
                "confirm_write": "confirm_write",
                "final_llm_answer": "final_llm_answer",
            },
        )
        builder.add_conditional_edges(
            "confirm_write",
            self._route_after_confirm_write,
            {"execute_tool": "execute_tool", "final_llm_answer": "final_llm_answer"},
        )
        builder.add_edge("final_llm_answer", END)
        return builder

    async def _understand_with_llm(self, state: RightAgentStateV6) -> RightAgentStateV6:
        question = (state.get("question") or "").strip()
        if not question:
            return {
                "status": "need_more_info",
                "answer": "Please describe the permission request.",
                "follow_up_question": "Please provide a person, device, or permission question.",
                "needs_input": ["question"],
            }
        try:
            plan = await self._llm.understand_right_agent_v6(
                question=question,
                target_tools=tuple(TOOL_METADATA_V6.keys()),
                slot_names=SLOT_NAMES_V6,
            )
        except Exception as exc:
            return {"status": "error", "answer": "", "data": {"llm_error": str(exc), "stage": "planner"}}
        if not plan:
            return {"status": "error", "answer": "", "data": {"llm_error": "planner returned no plan"}}
        return {
            "intent": plan.get("intent"),
            "requested_tool": plan.get("target_tool") or "",
            "slots": self._clean_slots(plan.get("slots") or {}),
            "data": {"planner": plan},
        }

    async def _normalize_plan(self, state: RightAgentStateV6) -> RightAgentStateV6:
        if state.get("status") in ("need_more_info", "error"):
            return {}
        requested = state.get("requested_tool") or ""
        intent = state.get("intent")
        target = PLANNER_TARGET_TO_TOOL_V6.get(requested) or PLANNER_TARGET_TO_TOOL_V6.get(str(intent))
        data = state.get("data") or {}
        if target in WRITE_TOOLS_V6:
            return {
                "target_tool": "query_permission",
                "current_tool": "query_permission",
                "pending_write_tool": target,
                "status": "",
                "data": {**data, "requested_write_tool": target},
            }
        if target in READ_TOOLS_V6:
            return {
                "target_tool": target,
                "current_tool": target,
                "pending_write_tool": None,
                "status": "",
                "data": data,
            }
        return {
            "status": "error",
            "answer": "",
            "data": {**data, "error": "unsupported target tool", "target_tool": target},
        }

    def _route_after_normalize(self, state: RightAgentStateV6) -> str:
        if state.get("status") in ("need_more_info", "error"):
            return "final_llm_answer"
        return "resolve_slots"

    async def _resolve_slots(self, state: RightAgentStateV6) -> RightAgentStateV6:
        current_tool = state.get("current_tool") or state.get("target_tool")
        if not current_tool or current_tool not in TOOL_METADATA_V6:
            return {"status": "error", "answer": "", "data": {"error": "unknown current tool"}}
        metadata = TOOL_METADATA_V6[current_tool]
        slots = state.get("slots") or {}
        missing_any = self._missing_any_required(metadata.any_required_slots, slots)
        if missing_any:
            return {
                "status": "need_more_info",
                "answer": "Missing information for the next tool.",
                "follow_up_question": f"Please provide one of: {', '.join(missing_any)}.",
                "needs_input": missing_any,
            }
        for slot in metadata.required_slots:
            if slots.get(slot):
                continue
            resolver = metadata.slot_resolvers.get(slot)
            if resolver:
                return {
                    "current_tool": resolver,
                    "status": "",
                    "tool_result": {},
                    "data": {**(state.get("data") or {}), "after_resolver_tool": state.get("target_tool")},
                }
            return {
                "status": "need_more_info",
                "answer": "Missing required information.",
                "follow_up_question": f"Please provide {slot}.",
                "needs_input": [slot],
            }
        return {"current_tool": current_tool, "status": ""}

    def _route_after_resolve_slots(self, state: RightAgentStateV6) -> str:
        if state.get("status") in ("need_more_info", "error"):
            return "final_llm_answer"
        return "execute_tool"

    async def _execute_tool(self, state: RightAgentStateV6) -> RightAgentStateV6:
        result = await self._tools.execute(
            state["current_tool"],
            state["project_id"],
            state.get("slots") or {},
        )
        return {"tool_result": result}

    async def _handle_tool_result(self, state: RightAgentStateV6) -> RightAgentStateV6:
        result = state.get("tool_result") or {}
        current_tool = state["current_tool"]
        history = self._append_tool_history(state, current_tool, result)
        data = {**(state.get("data") or {}), "tool_history": history}
        if result.get("status") == "not_configured":
            return {"status": "error", "answer": "", "tool_history": history, "data": {**data, "result": result}}

        target = state.get("target_tool") or current_tool
        if current_tool != target:
            return self._handle_resolver_result(state, current_tool, result, history, data)

        if current_tool == "search_person":
            return self._handle_person_lookup(state, result, history, data)
        if current_tool == "search_device":
            return self._handle_device_lookup(state, result, history, data)
        if current_tool == "query_permission":
            permission_status = self._policy.permission_status(result).value
            data = {**data, "permission_result": result, "permission_status": permission_status}
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
        if current_tool == "search_person":
            people = self._extract_people(result)
            selected = self._select_candidate(
                kind="person",
                candidates=people,
                choices=self._person_choices(people),
                empty_answer="No matching person was found.",
            )
            if selected.get("status"):
                return {**selected, "tool_history": history, "data": {**data, **(selected.get("data") or {})}}
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
            if selected.get("status"):
                return {**selected, "tool_history": history, "data": {**data, **(selected.get("data") or {})}}
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
        people = self._extract_people(result)
        selected = self._select_candidate(
            kind="person",
            candidates=people,
            choices=self._person_choices(people),
            empty_answer="No matching person was found.",
        )
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
        devices = self._extract_devices(result)
        selected = self._select_candidate(
            kind="device",
            candidates=devices,
            choices=self._device_choices(devices),
            empty_answer="No matching device was found.",
        )
        if selected.get("status"):
            return {**selected, "tool_history": history, "data": {**data, **(selected.get("data") or {})}}
        return {
            "status": "ok",
            "slots": {**(state.get("slots") or {}), **self._device_slots(selected["data"])},
            "tool_history": history,
            "data": {**data, "device": selected["data"]},
        }

    def _route_after_tool_result(self, state: RightAgentStateV6) -> str:
        if state.get("status") == "continue_resolve":
            return "resolve_slots"
        if state.get("status") == "need_confirm":
            return "confirm_write"
        return "final_llm_answer"

    async def _confirm_write(self, state: RightAgentStateV6) -> RightAgentStateV6:
        write_tool = state.get("pending_write_tool")
        permission_result = state.get("permission_result") or (state.get("data") or {}).get("permission_result") or {}
        if not write_tool or write_tool not in WRITE_TOOLS_V6:
            return {"status": "error", "answer": "", "data": {**(state.get("data") or {}), "error": "missing write tool"}}

        metadata = TOOL_METADATA_V6[write_tool]
        decision = self._policy.check_write(tool=metadata, permission_result=permission_result, confirmed=False)
        if decision.get("reason") != "requires_confirm":
            return {
                "status": decision.get("status") or "policy_denied",
                "answer": "",
                "policy": dict(decision),
                "data": {**(state.get("data") or {}), "policy": dict(decision)},
            }
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
        if not self._is_confirmed(resume, confirm_id):
            return {
                "status": "cancelled",
                "data": {**(state.get("data") or {}), "permission_result": permission_result, "policy": dict(decision)},
            }
        confirmed_decision = self._policy.check_write(tool=metadata, permission_result=permission_result, confirmed=True)
        if not confirmed_decision.allowed:
            return {
                "status": confirmed_decision.get("status") or "policy_denied",
                "answer": "",
                "policy": dict(confirmed_decision),
                "data": {**(state.get("data") or {}), "policy": dict(confirmed_decision)},
            }
        return {
            "current_tool": write_tool,
            "target_tool": write_tool,
            "status": "",
            "policy": dict(confirmed_decision),
            "data": {**(state.get("data") or {}), "policy": dict(confirmed_decision)},
        }

    def _route_after_confirm_write(self, state: RightAgentStateV6) -> str:
        if state.get("status") in ("cancelled", "error", "policy_denied"):
            return "final_llm_answer"
        return "execute_tool"

    async def _final_llm_answer(self, state: RightAgentStateV6) -> RightAgentStateV6:
        if state.get("answer"):
            return {}
        if state.get("status") == "need_more_info" and state.get("follow_up_question"):
            return {}
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
        if not answer:
            return {"status": "error", "answer": "", "data": {**(state.get("data") or {}), "llm_error": "empty final answer"}}
        return {"status": state.get("status") or "ok", "answer": answer}

    @staticmethod
    def _append_tool_history(state: RightAgentStateV6, tool_name: str, result: dict[str, Any]) -> list[dict[str, Any]]:
        return [*(state.get("tool_history") or []), {"tool": tool_name, "result": result}]

    @staticmethod
    def _missing_any_required(required: tuple[str, ...], slots: dict[str, Any]) -> list[str]:
        if not required:
            return []
        if any(slots.get(slot) for slot in required):
            return []
        return list(required)

    @staticmethod
    def _clean_slots(slots: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in slots.items() if key in SLOT_NAMES_V6 and value not in (None, "")}

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
    def _extract_devices(result: dict[str, Any]) -> list[dict[str, Any]]:
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
        return f"{person.get('personName') or person.get('name') or 'Unknown person'} {person.get('telephone') or person.get('phone') or ''}".strip()

    @staticmethod
    def _device_label(device: dict[str, Any]) -> str:
        return f"{device.get('deviceName') or device.get('name') or 'Unknown device'} {device.get('deviceSn') or device.get('sn') or device.get('deviceId') or ''}".strip()

    @staticmethod
    def _person_slots(person: dict[str, Any]) -> dict[str, Any]:
        return {
            "personId": person.get("personId") or person.get("id"),
            "personName": person.get("personName") or person.get("name"),
            "telephone": person.get("telephone") or person.get("phone"),
        }

    @staticmethod
    def _device_slots(device: dict[str, Any]) -> dict[str, Any]:
        return {
            "deviceSn": device.get("deviceSn") or device.get("sn"),
            "deviceId": device.get("deviceId") or device.get("id"),
            "deviceName": device.get("deviceName") or device.get("name"),
        }

    def _select_candidate(self, *, kind: str, candidates: list[dict[str, Any]], choices: list[dict[str, Any]], empty_answer: str) -> dict[str, Any]:
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
    def _choice_from_resume(resume: Any, choices: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not isinstance(resume, dict) or resume.get("type") != "choice":
            return None
        choice_id = resume.get("choice_id")
        for choice in choices:
            if str(choice.get("id")) == str(choice_id):
                return choice
        return None

    @staticmethod
    def _is_confirmed(resume: Any, confirm_id: str) -> bool:
        return (
            isinstance(resume, dict)
            and resume.get("type") == "confirm"
            and resume.get("confirm_id") == confirm_id
            and resume.get("confirmed") is True
        )

    def _to_response(self, state: dict[str, Any], session_id: str) -> AgentResponseV6:
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
        interrupts = state.get(INTERRUPT_KEY)
        if not interrupts:
            return None
        first = interrupts[0] if isinstance(interrupts, (list, tuple)) else interrupts
        value = getattr(first, "value", first)
        return value if isinstance(value, dict) else {"value": value}

    def _interrupt_to_response(self, payload: dict[str, Any], session_id: str) -> AgentResponseV6:
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
        return f"right-agent-v6:{project_id}:{session_id}"
