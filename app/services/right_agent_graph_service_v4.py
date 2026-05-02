from __future__ import annotations

import os.path
import re
import uuid
from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.clients.llm_client import LlmClient
from app.config import get_settings
from app.memory.langgraph_mysql_checkpoint_v4 import get_v4_mysql_checkpointer
from app.schemas.right_schema_v4 import (
    AgentChoiceV4,
    AgentConfirmV4,
    AgentRequestV4,
    AgentResponseV4,
)
from app.services.right_agent_metadata_v4 import (
    INTENT_TO_TOOL,
    PERMISSION_EXPIRED,
    RENEW_PERMISSION,
    SUPPORTED_CAPABILITIES,
    TOOL_METADATA,
)
from app.services.right_agent_policy_v4 import RightAgentPolicyCheckerV4
from app.tools.right_tools_v4 import RightToolsV4

INTERRUPT_KEY = "__interrupt__"


class RightAgentStateV4(TypedDict, total=False):
    session_id: str
    project_id: int
    question: str | None
    intent: str
    target_tool: str
    current_tool: str
    slots: dict[str, Any]
    tool_result: dict[str, Any]
    status: str
    answer: str
    follow_up_question: str | None
    needs_input: list[str]
    choices: list[dict[str, Any]]
    confirm: dict[str, Any] | None
    data: dict[str, Any]


class RightAgentGraphServiceV4:
    def __init__(
        self,
        *,
        tools: RightToolsV4 | None = None,
        policy: RightAgentPolicyCheckerV4 | None = None,
        checkpointer: Any | None = None,
        llm: LlmClient | None = None,
    ) -> None:
        self._settings = get_settings()
        self._tools = tools or RightToolsV4()
        self._policy = policy or RightAgentPolicyCheckerV4()
        self._checkpointer = checkpointer
        self._llm = llm or LlmClient()
        self._graph = None

    async def do_execute(
        self, request: AgentRequestV4, project_id: str | None
    ) -> AgentResponseV4:
        effective_project_id = self._parse_project_id(project_id)
        session_id = request.session_id or uuid.uuid4().hex
        if effective_project_id is None:
            return AgentResponseV4(
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
            return AgentResponseV4(
                answer="当前没有可继续的选择或确认，请重新发起查询。",
                status="error",
                session_id=session_id,
                data={"error": str(exc)},
            )
        return self._to_response(state, session_id)

    async def _get_graph(self):
        if self._graph is not None:
            return self._graph
        checkpointer = self._checkpointer or await get_v4_mysql_checkpointer()
        self._graph = self._build_graph().compile(checkpointer=checkpointer)
        return self._graph

    def _build_graph(self):
        builder = StateGraph(RightAgentStateV4)
        builder.add_node("understand", self._understand)
        builder.add_node("plan_task", self._plan_task)
        builder.add_node("resolve_slots", self._resolve_slots)
        builder.add_node("check_policy", self._check_policy)
        builder.add_node("confirm_action", self._confirm_action)
        builder.add_node("execute_tool", self._execute_tool)
        builder.add_node("handle_tool_result", self._handle_tool_result)
        builder.add_node("answer", self._answer)

        builder.add_edge(START, "understand")
        builder.add_edge("understand", "plan_task")
        builder.add_conditional_edges(
            "plan_task",
            self._route_after_plan,
            {"resolve_slots": "resolve_slots", "answer": "answer"},
        )
        builder.add_conditional_edges(
            "resolve_slots",
            self._route_after_resolve_slots,
            {"check_policy": "check_policy", "answer": "answer"},
        )
        builder.add_conditional_edges(
            "check_policy",
            self._route_after_policy,
            {
                "confirm_action": "confirm_action",
                "execute_tool": "execute_tool",
                "answer": "answer",
            },
        )
        builder.add_conditional_edges(
            "confirm_action",
            self._route_after_confirm,
            {"execute_tool": "execute_tool", "answer": "answer"},
        )
        builder.add_edge("execute_tool", "handle_tool_result")
        builder.add_conditional_edges(
            "handle_tool_result",
            self._route_after_tool_result,
            {
                "resolve_slots": "resolve_slots",
                "check_policy": "check_policy",
                "answer": "answer",
            },
        )
        builder.add_edge("answer", END)
        return builder

    async def _understand(self, state: RightAgentStateV4) -> RightAgentStateV4:
        question = (state.get("question") or "").strip()
        if not question:
            return {
                "status": "need_more_info",
                "answer": "请先描述你要查询、诊断或办理的门禁问题。",
                "follow_up_question": "请提供姓名、手机号、卡号、设备名称或门禁点。",
                "needs_input": ["question"],
            }
        understood = await self._understand_with_llm(question)
        if understood:
            return understood
        return {
            "intent": self._classify_intent(question),
            "slots": self._extract_slots(question),
        }

    async def _understand_with_llm(self, question: str) -> RightAgentStateV4 | None:
        if not self._llm.enabled:
            return None
        try:
            understood = await self._llm.understand_right_agent_v4(
                question=question,
                intents=tuple(INTENT_TO_TOOL.keys()),
                slot_names=(
                    "personId",
                    "telephone",
                    "cardNo",
                    "deviceId",
                    "deviceName",
                    "deviceSn",
                    "personName",
                ),
            )
        except Exception:
            return None
        if not understood:
            return None

        intent = understood.get("intent")
        if not isinstance(intent, str) or intent not in INTENT_TO_TOOL:
            intent = "unsupported"
        raw_slots = understood.get("slots")
        slots = self._clean_slots(raw_slots if isinstance(raw_slots, dict) else {})
        return {
            "intent": intent,
            "slots": slots,
            "data": {"understanding": {"source": "llm"}},
        }

    async def _plan_task(self, state: RightAgentStateV4) -> RightAgentStateV4:
        if state.get("status") == "need_more_info":
            return {}
        intent = state.get("intent") or "unsupported"
        target_tool = INTENT_TO_TOOL.get(intent)
        if not target_tool:
            return {
                "status": "unsupported",
                "answer": "当前还不能处理这个需求。",
                "follow_up_question": "可以换成查询人员、设备、卡号、权限、刷卡记录，或诊断开不了门原因。",
                "data": {"supported_capabilities": SUPPORTED_CAPABILITIES},
            }
        return {"target_tool": target_tool, "current_tool": target_tool}

    def _route_after_plan(self, state: RightAgentStateV4) -> str:
        if state.get("status") in ("need_more_info", "unsupported"):
            return "answer"
        return "resolve_slots"

    async def _resolve_slots(self, state: RightAgentStateV4) -> RightAgentStateV4:
        current_tool = state["current_tool"]
        metadata = TOOL_METADATA[current_tool]
        slots = state.get("slots") or {}
        if current_tool == state.get("target_tool") and metadata.risk == "write" and not metadata.configured:
            return {
                "status": "not_configured",
                "answer": f"{metadata.description}能力尚未配置后端接口，当前不能执行。",
            }

        missing_any = self._missing_any_required(metadata.any_required_slots, slots)
        if missing_any:
            return {
                "status": "need_more_info",
                "answer": f"{metadata.description}还缺少必要信息。",
                "follow_up_question": f"请提供以下任一信息：{', '.join(missing_any)}。",
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
                    "data": {
                        **(state.get("data") or {}),
                        "resolving_slot": slot,
                        "after_resolver_tool": state.get("target_tool"),
                    },
                }
            return {
                "status": "need_more_info",
                "answer": f"{metadata.description}还缺少 {slot}。",
                "follow_up_question": f"请提供 {slot}。",
                "needs_input": [slot],
            }
        return {"status": ""}

    def _route_after_resolve_slots(self, state: RightAgentStateV4) -> str:
        if state.get("status") in ("need_more_info", "not_configured"):
            return "answer"
        return "check_policy"

    async def _check_policy(self, state: RightAgentStateV4) -> RightAgentStateV4:
        tool = TOOL_METADATA[state["current_tool"]]
        decision = self._policy.check(tool=tool, state=dict(state), confirmed=False)
        if decision.allowed:
            return {"status": ""}
        if decision.get("reason") == "requires_confirm":
            return {"status": "need_confirm"}
        return {
            "status": decision.get("status") or "policy_denied",
            "answer": decision.get("message") or "当前策略不允许执行该操作。",
            "data": {**(state.get("data") or {}), "policy": dict(decision)},
        }

    def _route_after_policy(self, state: RightAgentStateV4) -> str:
        if state.get("status") == "need_confirm":
            return "confirm_action"
        if state.get("status") in ("not_configured", "policy_denied"):
            return "answer"
        return "execute_tool"

    async def _confirm_action(self, state: RightAgentStateV4) -> RightAgentStateV4:
        tool = TOOL_METADATA[state["current_tool"]]
        confirm_id = tool.confirm_id or state["current_tool"]
        confirm = {
            "id": confirm_id,
            "label": f"确认{tool.description}",
            "description": f"{tool.description}会修改业务数据，确认后将继续执行。",
            "risk_level": "medium",
        }
        resume = interrupt(
            {
                "type": "confirm",
                "answer": f"是否确认执行：{tool.description}？",
                "follow_up_question": "请确认是否继续。",
                "confirm": confirm,
                "data": state.get("data") or {},
            }
        )
        if not self._is_confirmed(resume, confirm_id):
            return {
                "status": "cancelled",
                "answer": f"已取消{tool.description}。",
            }
        decision = self._policy.check(tool=tool, state=dict(state), confirmed=True)
        if not decision.allowed:
            return {
                "status": decision.get("status") or "policy_denied",
                "answer": decision.get("message") or "当前策略不允许执行该操作。",
                "data": {**(state.get("data") or {}), "policy": dict(decision)},
            }
        return {"status": ""}

    def _route_after_confirm(self, state: RightAgentStateV4) -> str:
        if state.get("status") in ("cancelled", "not_configured", "policy_denied"):
            return "answer"
        return "execute_tool"

    async def _execute_tool(self, state: RightAgentStateV4) -> RightAgentStateV4:
        result = await self._tools.execute(
            state["current_tool"],
            state["project_id"],
            state.get("slots") or {},
        )
        return {"tool_result": result}

    async def _handle_tool_result(self, state: RightAgentStateV4) -> RightAgentStateV4:
        result = state.get("tool_result") or {}
        if result.get("status") == "not_configured":
            return {
                "status": "not_configured",
                "answer": result.get("message") or "当前能力尚未配置后端接口。",
                "data": {"result": result},
            }

        current_tool = state["current_tool"]
        target_tool = state.get("target_tool") or current_tool
        if current_tool != target_tool:
            return self._handle_resolver_result(state)

        if current_tool == "search_person":
            return self._handle_person_lookup_result(state)

        diagnosis = self._extract_diagnosis(result)
        if current_tool == "diagnose_access_issue" and diagnosis.get("mainCause") == PERMISSION_EXPIRED:
            return {
                "current_tool": "renew_permission",
                "target_tool": "renew_permission",
                "status": "",
                "data": {"diagnosis": diagnosis},
                "tool_result": {},
            }
        if diagnosis:
            return {"status": "ok", "data": {"diagnosis": diagnosis}}
        return {"status": "ok", "data": {"result": result}}

    def _handle_resolver_result(self, state: RightAgentStateV4) -> RightAgentStateV4:
        current_tool = state["current_tool"]
        if current_tool == "search_person":
            people = self._extract_people(state.get("tool_result") or {})
            person = self._select_candidate(
                kind="person",
                candidates=people,
                choices=self._person_choices(people),
                empty_answer="未找到匹配人员。",
            )
            if person.get("status"):
                return person
            slots = {**(state.get("slots") or {}), **self._person_slots(person["data"])}
            return {
                "slots": slots,
                "current_tool": state.get("target_tool") or "diagnose_access_issue",
                "status": "continue_resolve",
                "tool_result": {},
            }
        return {
            "status": "not_configured",
            "answer": f"{TOOL_METADATA[current_tool].description}能力尚未配置后端接口。",
        }

    def _handle_person_lookup_result(self, state: RightAgentStateV4) -> RightAgentStateV4:
        people = self._extract_people(state.get("tool_result") or {})
        person = self._select_candidate(
            kind="person",
            candidates=people,
            choices=self._person_choices(people),
            empty_answer="未找到匹配人员。",
        )
        if person.get("status"):
            return person
        return {
            "status": "ok",
            "answer": f"已找到人员：{self._person_label(person['data'])}。",
            "data": {"person": person["data"]},
        }

    def _select_candidate(
        self,
        *,
        kind: str,
        candidates: list[dict[str, Any]],
        choices: list[dict[str, Any]],
        empty_answer: str,
    ) -> dict[str, Any]:
        if not candidates:
            return {"status": "ok", "answer": empty_answer, "data": {"result": []}}
        if len(candidates) == 1:
            return {"data": candidates[0]}
        resume = interrupt(
            {
                "type": "choice",
                "kind": kind,
                "answer": "找到多个匹配对象，请选择一个后继续。",
                "follow_up_question": "请选择要处理的对象。",
                "choices": choices,
            }
        )
        choice = self._choice_from_resume(resume, choices)
        if choice is None:
            return {"status": "error", "answer": "无效选择，请重新发起查询。"}
        return {"data": choice.get("data") or {}}

    def _route_after_tool_result(self, state: RightAgentStateV4) -> str:
        if state.get("status") in ("ok", "error", "not_configured") or state.get("answer"):
            return "answer"
        if state.get("status") == "continue_resolve":
            return "resolve_slots"
        if state.get("current_tool") == state.get("target_tool"):
            return "check_policy"
        return "resolve_slots"

    async def _answer(self, state: RightAgentStateV4) -> RightAgentStateV4:
        if state.get("answer"):
            return {}
        if state.get("status") == "unsupported":
            return {}
        diagnosis = (state.get("data") or {}).get("diagnosis") or {}
        if diagnosis:
            main_name = diagnosis.get("mainCauseName") or diagnosis.get("mainCause") or "未知原因"
            summary = diagnosis.get("summary") or ""
            return {
                "status": state.get("status") or "ok",
                "answer": f"诊断结果：{main_name}。{summary}".strip(),
            }
        return {"status": state.get("status") or "ok", "answer": "处理完成。"}

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
        return f"right-agent-v4:{project_id}:{session_id}"

    @staticmethod
    def _classify_intent(question: str) -> str:
        if any(word in question for word in ("续期", "延期")):
            return "renew_permission"
        if any(word in question for word in ("开通", "授权", "添加权限")):
            return "open_permission"
        if any(word in question for word in ("挂失", "禁用卡", "停用卡")):
            return "disable_card"
        if any(word in question for word in ("刷卡记录", "通行记录", "开门记录")):
            return "access_record_lookup"
        if any(word in question for word in ("卡号", "门禁卡")):
            return "card_lookup"
        if any(word in question for word in ("凭证", "二维码", "人脸")):
            return "credential_lookup"
        if any(word in question for word in ("权限", "能不能进", "能否进")):
            return "permission_lookup"
        if any(word in question for word in ("设备", "门禁机", "在线", "离线")):
            return "device_lookup"
        if any(word in question for word in ("打不开", "开不了", "进不去", "刷不开")):
            return "access_issue"
        if any(word in question for word in ("找", "查询", "查一下", "叫")):
            return "person_lookup"
        return "unsupported"

    @staticmethod
    def _extract_slots(question: str) -> dict[str, Any]:
        slots: dict[str, Any] = {}
        phone = re.search(r"1[3-9]\d{9}", question)
        if phone:
            slots["telephone"] = phone.group(0)
        card = re.search(r"(?:cardNo|卡号|门禁卡)[:：\s]*([A-Za-z0-9_-]+)", question, re.IGNORECASE)
        if card:
            slots["cardNo"] = card.group(1)
        person_id = re.search(r"(?:personId|人员ID)[:：\s]*([A-Za-z0-9_-]+)", question, re.IGNORECASE)
        if person_id:
            slots["personId"] = person_id.group(1)
        device_id = re.search(r"(?:deviceId|设备ID)[:：\s]*([A-Za-z0-9_-]+)", question, re.IGNORECASE)
        if device_id:
            slots["deviceId"] = device_id.group(1)
        device_sn = re.search(r"(?:deviceSn|设备SN|sn)[:：\s]*([A-Za-z0-9_-]+)", question, re.IGNORECASE)
        if device_sn:
            slots["deviceSn"] = device_sn.group(1)
        name = re.search(r"叫([\u4e00-\u9fa5]{2,4})", question)
        if not name:
            name = re.search(r"查询([\u4e00-\u9fa5]{2,4})", question)
        if not name:
            name = re.search(r"([\u4e00-\u9fa5]{2,4})(?:开不了|打不开|进不去|刷不开|权限|卡号)", question)
        if not name:
            name = re.search(r"^([\u4e00-\u9fa5]{2,4})", question)
        if name:
            slots["personName"] = name.group(1)
        device = re.search(r"([\u4e00-\u9fa5一二三四五六七八九十0-9]+号门)", question)
        if device:
            slots["deviceName"] = device.group(1)
        return slots

    @staticmethod
    def _clean_slots(slots: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "personId",
            "telephone",
            "cardNo",
            "deviceId",
            "deviceName",
            "deviceSn",
            "personName",
        }
        cleaned: dict[str, Any] = {}
        for key, value in slots.items():
            if key not in allowed or value in (None, ""):
                continue
            cleaned[key] = value
        return cleaned

    @staticmethod
    def _missing_any_required(required: tuple[str, ...], slots: dict[str, Any]) -> list[str]:
        if not required:
            return []
        if any(slots.get(slot) for slot in required):
            return []
        return list(required)

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
    def _is_confirmed(resume: Any, confirm_id: str) -> bool:
        return (
            isinstance(resume, dict)
            and resume.get("type") == "confirm"
            and resume.get("confirm_id") == confirm_id
            and resume.get("confirmed") is True
        )

    def _to_response(self, state: dict[str, Any], session_id: str) -> AgentResponseV4:
        interrupt_payload = self._interrupt_payload(state)
        if interrupt_payload:
            return self._interrupt_to_response(interrupt_payload, session_id)
        return AgentResponseV4(
            answer=state.get("answer") or "",
            status=state.get("status") or "ok",
            session_id=state.get("session_id") or session_id,
            follow_up_question=state.get("follow_up_question"),
            choices=[AgentChoiceV4.model_validate(item) for item in state.get("choices") or []],
            confirm=(
                AgentConfirmV4.model_validate(state["confirm"])
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
    ) -> AgentResponseV4:
        if payload.get("type") == "choice":
            return AgentResponseV4(
                answer=payload.get("answer") or "请选择一个选项后继续。",
                status="need_choice",
                session_id=session_id,
                follow_up_question=payload.get("follow_up_question"),
                choices=[
                    AgentChoiceV4.model_validate(item)
                    for item in payload.get("choices") or []
                ],
                data=payload.get("data") or {},
            )
        if payload.get("type") == "confirm":
            confirm = payload.get("confirm")
            return AgentResponseV4(
                answer=payload.get("answer") or "请确认是否继续。",
                status="need_confirm",
                session_id=session_id,
                follow_up_question=payload.get("follow_up_question"),
                confirm=AgentConfirmV4.model_validate(confirm) if confirm else None,
                data=payload.get("data") or {},
            )
        return AgentResponseV4(
            answer="流程已暂停，请提交续跑数据后继续。",
            status="interrupted",
            session_id=session_id,
            data=payload,
        )

    @lru_cache
    def _get_system_prompt(self) -> str:
        prompt_path = os.path.join(self._settings.base_dir, "prompts", "right_agent_system_v4.md")
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read()
