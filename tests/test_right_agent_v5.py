from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.api import right_routes_v5
from app.schemas.right_schema_v5 import AgentRequestV5, AgentResponseV5
from app.services.right_agent_graph_service_v5 import RightAgentGraphServiceV5
from app.tools.right_tools_v5 import RightToolsV5

ZHANG_SAN = "\u5f20\u4e09"


class FakeToolsV5:
    def __init__(self) -> None:
        self.results: dict[str, list[dict[str, Any]]] = {
            "search_person": [],
            "search_device": [],
            "query_permission": [],
            "renew_permission": [],
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


class FakeLlmV5:
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

    async def understand_right_agent_v5(
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

    async def answer_right_agent_v5(
        self,
        *,
        question: str | None,
        slots: dict[str, Any],
        tool_history: list[dict[str, Any]],
        permission_result: dict[str, Any] | None = None,
        renew_result: dict[str, Any] | None = None,
    ) -> str | None:
        if self.fail_answer:
            raise RuntimeError("final llm down")
        self.answer_calls.append(
            {
                "question": question,
                "slots": dict(slots),
                "tool_history": tool_history,
                "permission_result": permission_result,
                "renew_result": renew_result,
            }
        )
        return self.answer


def make_service(tools: FakeToolsV5, llm: FakeLlmV5) -> RightAgentGraphServiceV5:
    return RightAgentGraphServiceV5(
        tools=tools,
        llm=llm,
        checkpointer=InMemorySaver(),
    )


def query_permission_plan(slots: dict[str, Any]) -> dict[str, Any]:
    return {
        "intent": "access_issue",
        "target_tool": "query_permission",
        "slots": slots,
    }


def test_right_tools_v5_exposes_registry_and_read_langchain_tools() -> None:
    tools = RightToolsV5()

    assert set(tools.registry) == {
        "search_person",
        "search_device",
        "query_permission",
        "renew_permission",
    }
    assert {tool.name for tool in tools.read_langchain_tools()} == {
        "search_person",
        "search_device",
        "query_permission",
    }
    assert "renew_permission" not in {tool.name for tool in tools.read_langchain_tools()}


@pytest.mark.asyncio
async def test_person_name_and_device_sn_skips_search_device() -> None:
    tools = FakeToolsV5()
    tools.results["search_person"].append(
        {"context": {"personId": "p1", "personName": ZHANG_SAN}}
    )
    tools.results["query_permission"].append({"permission": {"status": "ACTIVE"}})
    llm = FakeLlmV5(
        plan=query_permission_plan({"personName": ZHANG_SAN, "deviceSn": "D001"}),
        answer="ok from llm",
    )
    service = make_service(tools, llm)

    response = await service.do_execute(
        AgentRequestV5(question=f"{ZHANG_SAN} D001 \u5f00\u4e0d\u4e86\u95e8"),
        "100",
    )

    assert response.status == "ok"
    assert response.answer == "ok from llm"
    assert [name for name, _ in tools.calls] == ["search_person", "query_permission"]
    assert tools.calls[-1][1]["slots"]["personId"] == "p1"
    assert tools.calls[-1][1]["slots"]["deviceSn"] == "D001"


@pytest.mark.asyncio
async def test_person_id_and_device_sn_calls_query_permission_directly() -> None:
    tools = FakeToolsV5()
    tools.results["query_permission"].append({"permission": {"status": "ACTIVE"}})
    llm = FakeLlmV5(
        plan=query_permission_plan({"personId": "p1", "deviceSn": "D001"}),
        answer="direct answer",
    )
    service = make_service(tools, llm)

    response = await service.do_execute(
        AgentRequestV5(question="personId p1 deviceSn D001"),
        "100",
    )

    assert response.status == "ok"
    assert response.answer == "direct answer"
    assert [name for name, _ in tools.calls] == ["query_permission"]


@pytest.mark.asyncio
async def test_person_name_and_device_name_resolves_person_and_device() -> None:
    tools = FakeToolsV5()
    tools.results["search_person"].append(
        {"context": {"personId": "p1", "personName": ZHANG_SAN}}
    )
    tools.results["search_device"].append(
        {"context": {"deviceSn": "D001", "deviceName": "\u4e09\u53f7\u95e8"}}
    )
    tools.results["query_permission"].append({"permission": {"status": "ACTIVE"}})
    llm = FakeLlmV5(
        plan=query_permission_plan({"personName": ZHANG_SAN, "deviceName": "\u4e09\u53f7\u95e8"}),
        answer="resolved answer",
    )
    service = make_service(tools, llm)

    response = await service.do_execute(
        AgentRequestV5(question=f"{ZHANG_SAN}\u4e09\u53f7\u95e8\u5f00\u4e0d\u4e86"),
        "100",
    )

    assert response.status == "ok"
    assert [name for name, _ in tools.calls] == [
        "search_person",
        "search_device",
        "query_permission",
    ]
    assert tools.calls[-1][1]["slots"]["personId"] == "p1"
    assert tools.calls[-1][1]["slots"]["deviceSn"] == "D001"


@pytest.mark.asyncio
async def test_multiple_people_interrupts_and_resume_continues() -> None:
    tools = FakeToolsV5()
    tools.results["search_person"].append(
        {
            "people": [
                {"personId": "p1", "personName": ZHANG_SAN},
                {"personId": "p2", "personName": ZHANG_SAN},
            ]
        }
    )
    tools.results["query_permission"].append({"permission": {"status": "ACTIVE"}})
    llm = FakeLlmV5(
        plan=query_permission_plan({"personName": ZHANG_SAN, "deviceSn": "D001"}),
        answer="choice answer",
    )
    service = make_service(tools, llm)

    first = await service.do_execute(
        AgentRequestV5(
            question=f"{ZHANG_SAN} D001 \u5f00\u4e0d\u4e86",
            session_id="s-person-choice-v5",
        ),
        "100",
    )

    assert first.status == "need_choice"
    assert [choice.id for choice in first.choices] == ["p1", "p2"]

    second = await service.do_execute(
        AgentRequestV5(
            session_id="s-person-choice-v5",
            resume={"type": "choice", "choice_id": "p1"},
        ),
        "100",
    )

    assert second.status == "ok"
    assert [name for name, _ in tools.calls] == ["search_person", "query_permission"]


@pytest.mark.asyncio
async def test_multiple_devices_interrupts_and_resume_continues() -> None:
    tools = FakeToolsV5()
    tools.results["search_device"].append(
        {
            "devices": [
                {"deviceSn": "D001", "deviceName": "\u4e09\u53f7\u95e8"},
                {"deviceSn": "D002", "deviceName": "\u4e09\u53f7\u95e8"},
            ]
        }
    )
    tools.results["query_permission"].append({"permission": {"status": "ACTIVE"}})
    llm = FakeLlmV5(
        plan=query_permission_plan({"personId": "p1", "deviceName": "\u4e09\u53f7\u95e8"}),
        answer="device choice answer",
    )
    service = make_service(tools, llm)

    first = await service.do_execute(
        AgentRequestV5(
            question="p1 device",
            session_id="s-device-choice-v5",
        ),
        "100",
    )

    assert first.status == "need_choice"
    assert [choice.id for choice in first.choices] == ["D001", "D002"]

    second = await service.do_execute(
        AgentRequestV5(
            session_id="s-device-choice-v5",
            resume={"type": "choice", "choice_id": "D001"},
        ),
        "100",
    )

    assert second.status == "ok"
    assert [name for name, _ in tools.calls] == ["search_device", "query_permission"]
    assert tools.calls[-1][1]["slots"]["deviceSn"] == "D001"


@pytest.mark.asyncio
async def test_permission_expired_confirms_then_renews() -> None:
    tools = FakeToolsV5()
    tools.results["query_permission"].append(
        {"diagnosis": {"mainCause": "PERMISSION_EXPIRED", "mainCauseName": "\u6743\u9650\u8fc7\u671f"}}
    )
    tools.results["renew_permission"].append({"renewed": True})
    llm = FakeLlmV5(
        plan=query_permission_plan({"personId": "p1", "deviceSn": "D001"}),
        answer="\u5df2\u7eed\u671f",
    )
    service = make_service(tools, llm)

    first = await service.do_execute(
        AgentRequestV5(
            question="p1 D001 expired",
            session_id="s-renew-v5",
        ),
        "100",
    )

    assert first.status == "need_confirm"
    assert first.confirm is not None
    assert first.confirm.id == "renew_permission"
    assert [name for name, _ in tools.calls] == ["query_permission"]

    second = await service.do_execute(
        AgentRequestV5(
            session_id="s-renew-v5",
            resume={"type": "confirm", "confirm_id": "renew_permission", "confirmed": True},
        ),
        "100",
    )

    assert second.status == "ok"
    assert second.answer == "\u5df2\u7eed\u671f"
    assert [name for name, _ in tools.calls] == ["query_permission", "renew_permission"]
    assert llm.answer_calls[-1]["renew_result"] == {"renewed": True}


@pytest.mark.asyncio
async def test_non_expired_permission_does_not_renew() -> None:
    tools = FakeToolsV5()
    tools.results["query_permission"].append({"permission": {"status": "ACTIVE"}})
    llm = FakeLlmV5(
        plan=query_permission_plan({"personId": "p1", "deviceSn": "D001"}),
        answer="not expired",
    )
    service = make_service(tools, llm)

    response = await service.do_execute(
        AgentRequestV5(question="p1 D001", session_id="s-no-renew-v5"),
        "100",
    )

    assert response.status == "ok"
    assert response.confirm is None
    assert [name for name, _ in tools.calls] == ["query_permission"]


@pytest.mark.asyncio
async def test_direct_renew_plan_is_normalized_to_permission_query() -> None:
    tools = FakeToolsV5()
    tools.results["query_permission"].append({"permission": {"status": "ACTIVE"}})
    llm = FakeLlmV5(
        plan={
            "intent": "renew_permission",
            "target_tool": "renew_permission",
            "slots": {"personId": "p1", "deviceSn": "D001"},
        },
        answer="checked first",
    )
    service = make_service(tools, llm)

    response = await service.do_execute(
        AgentRequestV5(question="\u5e2e p1 D001 \u7eed\u671f"),
        "100",
    )

    assert response.status == "ok"
    assert [name for name, _ in tools.calls] == ["query_permission"]
    assert response.data["planner_rejected_tool"] == "renew_permission"


@pytest.mark.asyncio
async def test_final_llm_failure_returns_error_without_template_answer() -> None:
    tools = FakeToolsV5()
    tools.results["query_permission"].append({"permission": {"status": "ACTIVE"}})
    llm = FakeLlmV5(
        plan=query_permission_plan({"personId": "p1", "deviceSn": "D001"}),
        fail_answer=True,
    )
    service = make_service(tools, llm)

    response = await service.do_execute(
        AgentRequestV5(question="p1 D001"),
        "100",
    )

    assert response.status == "error"
    assert response.answer == ""
    assert response.data["stage"] == "final_answer"


def test_v5_route_returns_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubService:
        async def do_execute(
            self,
            request: AgentRequestV5,
            project_id: str | None,
        ) -> AgentResponseV5:
            return AgentResponseV5(
                answer="ok",
                status="ok",
                session_id=request.session_id or "generated",
            )

    monkeypatch.setattr(right_routes_v5, "agent_service_v5", StubService())
    app = FastAPI()
    app.include_router(right_routes_v5.router)
    client = TestClient(app)

    response = client.post(
        "/api/web/agent/v5",
        headers={"Project-Id": "100"},
        json={"question": "hello", "session_id": "s-api-v5"},
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == "s-api-v5"
