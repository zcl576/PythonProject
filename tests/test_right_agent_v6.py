from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.api import right_routes_v6
from app.schemas.right_schema_v6 import AgentRequestV6, AgentResponseV6
from app.services.right_agent_graph_service_v6 import RightAgentGraphServiceV6
from app.tools.right_tools_v6 import RightToolsV6

ZHANG_SAN = "\u5f20\u4e09"
DOOR_3 = "\u4e09\u53f7\u95e8"


class FakeToolsV6:
    def __init__(self) -> None:
        self.results: dict[str, list[dict[str, Any]]] = {
            "search_person": [],
            "search_device": [],
            "query_permission": [],
            "extend_permission": [],
            "enable_permission": [],
            "disable_permission": [],
            "grant_permission": [],
            "revoke_permission": [],
        }
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute(
        self,
        tool_name: str,
        project_id: int,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append((tool_name, {"project_id": project_id, "slots": dict(slots)}))
        queued = self.results.get(tool_name) or []
        if queued:
            return queued.pop(0)
        return {"ok": True}


class FakeLlmV6:
    def __init__(
        self,
        *,
        plan: dict[str, Any] | None,
        answer: str | None = "final answer",
        fail_answer: bool = False,
    ) -> None:
        self.plan = plan
        self.answer = answer
        self.fail_answer = fail_answer
        self.plan_calls: list[dict[str, Any]] = []
        self.answer_calls: list[dict[str, Any]] = []

    async def understand_right_agent_v6(
        self,
        *,
        question: str,
        target_tools: tuple[str, ...],
        slot_names: tuple[str, ...],
    ) -> dict[str, Any] | None:
        self.plan_calls.append(
            {
                "question": question,
                "target_tools": target_tools,
                "slot_names": slot_names,
            }
        )
        return self.plan

    async def answer_right_agent_v6(
        self,
        *,
        question: str | None,
        slots: dict[str, Any],
        tool_history: list[dict[str, Any]],
        permission_status: str | None = None,
        permission_result: dict[str, Any] | None = None,
        write_result: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
    ) -> str | None:
        if self.fail_answer:
            raise RuntimeError("final llm down")
        self.answer_calls.append(
            {
                "question": question,
                "slots": dict(slots),
                "tool_history": tool_history,
                "permission_status": permission_status,
                "permission_result": permission_result,
                "write_result": write_result,
                "policy": policy,
            }
        )
        return self.answer


def make_service(tools: FakeToolsV6, llm: FakeLlmV6) -> RightAgentGraphServiceV6:
    return RightAgentGraphServiceV6(
        tools=tools,
        llm=llm,
        checkpointer=InMemorySaver(),
    )


def plan(target_tool: str, slots: dict[str, Any], intent: str | None = None) -> dict[str, Any]:
    return {
        "intent": intent or target_tool,
        "target_tool": target_tool,
        "slots": slots,
    }


def test_right_tools_v6_exposes_registry_and_read_langchain_tools() -> None:
    tools = RightToolsV6()

    assert set(tools.registry) == {
        "search_person",
        "search_device",
        "query_permission",
        "extend_permission",
        "enable_permission",
        "disable_permission",
        "grant_permission",
        "revoke_permission",
    }
    assert {tool.name for tool in tools.read_langchain_tools()} == {
        "search_person",
        "search_device",
        "query_permission",
    }
    assert "extend_permission" not in {tool.name for tool in tools.read_langchain_tools()}
    assert "enable_permission" not in {tool.name for tool in tools.read_langchain_tools()}


@pytest.mark.asyncio
async def test_query_permission_with_ids_calls_query_directly() -> None:
    tools = FakeToolsV6()
    tools.results["query_permission"].append({"permission": {"status": "ACTIVE"}})
    llm = FakeLlmV6(
        plan=plan("query_permission", {"personId": "p1", "deviceSn": "D001"}),
        answer="active answer",
    )
    service = make_service(tools, llm)

    response = await service.do_execute(
        AgentRequestV6(question="p1 D001"),
        "100",
    )

    assert response.status == "ok"
    assert response.answer == "active answer"
    assert [name for name, _ in tools.calls] == ["query_permission"]
    assert llm.answer_calls[-1]["permission_status"] == "ACTIVE"


@pytest.mark.asyncio
async def test_extend_expired_permission_requires_confirm_then_executes() -> None:
    tools = FakeToolsV6()
    tools.results["query_permission"].append({"permission": {"status": "EXPIRED"}})
    tools.results["extend_permission"].append({"extended": True})
    llm = FakeLlmV6(
        plan=plan(
            "extend_permission",
            {"personId": "p1", "deviceSn": "D001", "durationDays": 30},
        ),
        answer="extended answer",
    )
    service = make_service(tools, llm)

    first = await service.do_execute(
        AgentRequestV6(question="extend p1 D001", session_id="s-extend-v6"),
        "100",
    )

    assert first.status == "need_confirm"
    assert first.confirm is not None
    assert first.confirm.id == "extend_permission"
    assert [name for name, _ in tools.calls] == ["query_permission"]

    second = await service.do_execute(
        AgentRequestV6(
            session_id="s-extend-v6",
            resume={"type": "confirm", "confirm_id": "extend_permission", "confirmed": True},
        ),
        "100",
    )

    assert second.status == "ok"
    assert second.answer == "extended answer"
    assert [name for name, _ in tools.calls] == ["query_permission", "extend_permission"]
    assert tools.calls[-1][1]["slots"]["durationDays"] == 30
    assert llm.answer_calls[-1]["write_result"] == {"extended": True}


@pytest.mark.asyncio
async def test_enable_disabled_permission_confirms_then_executes() -> None:
    tools = FakeToolsV6()
    tools.results["query_permission"].append({"permission": {"status": "DISABLED"}})
    tools.results["enable_permission"].append({"enabled": True})
    llm = FakeLlmV6(
        plan=plan("enable_permission", {"personId": "p1", "deviceSn": "D001"}),
        answer="enabled answer",
    )
    service = make_service(tools, llm)

    first = await service.do_execute(
        AgentRequestV6(question="enable p1 D001", session_id="s-enable-v6"),
        "100",
    )

    assert first.status == "need_confirm"
    assert first.confirm is not None
    assert first.confirm.id == "enable_permission"

    second = await service.do_execute(
        AgentRequestV6(
            session_id="s-enable-v6",
            resume={"type": "confirm", "confirm_id": "enable_permission", "confirmed": True},
        ),
        "100",
    )

    assert second.status == "ok"
    assert [name for name, _ in tools.calls] == ["query_permission", "enable_permission"]


@pytest.mark.asyncio
async def test_enable_active_permission_is_policy_denied() -> None:
    tools = FakeToolsV6()
    tools.results["query_permission"].append({"permission": {"status": "ACTIVE"}})
    llm = FakeLlmV6(
        plan=plan("enable_permission", {"personId": "p1", "deviceSn": "D001"}),
        answer="policy denied answer",
    )
    service = make_service(tools, llm)

    response = await service.do_execute(
        AgentRequestV6(question="enable active p1 D001"),
        "100",
    )

    assert response.status == "policy_denied"
    assert response.answer == "policy denied answer"
    assert [name for name, _ in tools.calls] == ["query_permission"]
    assert llm.answer_calls[-1]["policy"]["reason"] == "status_not_allowed"


@pytest.mark.asyncio
async def test_write_plan_resolves_person_and_device_before_policy() -> None:
    tools = FakeToolsV6()
    tools.results["search_person"].append(
        {"context": {"personId": "p1", "personName": ZHANG_SAN}}
    )
    tools.results["search_device"].append(
        {"context": {"deviceSn": "D001", "deviceName": DOOR_3}}
    )
    tools.results["query_permission"].append({"permission": {"status": "EXPIRED"}})
    llm = FakeLlmV6(
        plan=plan(
            "extend_permission",
            {"personName": ZHANG_SAN, "deviceName": DOOR_3},
        ),
        answer="need confirm",
    )
    service = make_service(tools, llm)

    response = await service.do_execute(
        AgentRequestV6(question=f"{ZHANG_SAN}{DOOR_3}\u5ef6\u671f", session_id="s-resolve-write-v6"),
        "100",
    )

    assert response.status == "need_confirm"
    assert [name for name, _ in tools.calls] == [
        "search_person",
        "search_device",
        "query_permission",
    ]
    assert tools.calls[-1][1]["slots"]["personId"] == "p1"
    assert tools.calls[-1][1]["slots"]["deviceSn"] == "D001"


@pytest.mark.asyncio
async def test_final_llm_failure_returns_error_without_template_answer() -> None:
    tools = FakeToolsV6()
    tools.results["query_permission"].append({"permission": {"status": "ACTIVE"}})
    llm = FakeLlmV6(
        plan=plan("query_permission", {"personId": "p1", "deviceSn": "D001"}),
        fail_answer=True,
    )
    service = make_service(tools, llm)

    response = await service.do_execute(
        AgentRequestV6(question="p1 D001"),
        "100",
    )

    assert response.status == "error"
    assert response.answer == ""
    assert response.data["stage"] == "final_answer"


def test_v6_route_returns_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubService:
        async def do_execute(
            self,
            request: AgentRequestV6,
            project_id: str | None,
        ) -> AgentResponseV6:
            return AgentResponseV6(
                answer="ok",
                status="ok",
                session_id=request.session_id or "generated",
            )

    monkeypatch.setattr(right_routes_v6, "agent_service_v6", StubService())
    app = FastAPI()
    app.include_router(right_routes_v6.router)
    client = TestClient(app)

    response = client.post(
        "/api/web/agent/v6",
        headers={"Project-Id": "100"},
        json={"question": "hello", "session_id": "s-api-v6"},
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == "s-api-v6"
