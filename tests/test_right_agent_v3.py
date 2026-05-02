from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.api import right_routes_v3
from app.schemas.right_schema_v3 import AgentRequestV3
from app.schemas.right_schema_v3 import AgentResponseV3
from app.services.right_agent_graph_service_v3 import RightAgentGraphServiceV3

ZHANG_SAN = "\u5f20\u4e09"


class FakeToolsV3:
    def __init__(self) -> None:
        self.search_person_results: list[dict[str, Any]] = []
        self.diagnose_results: list[dict[str, Any]] = []
        self.renew_result: dict[str, Any] = {"ok": True}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def search_person(
        self,
        project_id: int,
        *,
        telephone: str | None = None,
        person_name: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "search_person",
                {
                    "project_id": project_id,
                    "telephone": telephone,
                    "person_name": person_name,
                },
            )
        )
        return self.search_person_results.pop(0)

    async def diagnose_access_issue(
        self,
        project_id: int,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(("diagnose_access_issue", {"project_id": project_id, "slots": slots}))
        return self.diagnose_results.pop(0)

    async def renew_permission(
        self,
        project_id: int,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(("renew_permission", {"project_id": project_id, "slots": slots}))
        return self.renew_result


def make_service(tools: FakeToolsV3) -> RightAgentGraphServiceV3:
    return RightAgentGraphServiceV3(tools=tools, checkpointer=InMemorySaver())


@pytest.mark.asyncio
async def test_missing_project_id_is_rejected() -> None:
    service = make_service(FakeToolsV3())

    response = await service.do_execute(
        AgentRequestV3(question=f"{ZHANG_SAN}\u4e09\u53f7\u95e8\u6253\u4e0d\u5f00"),
        None,
    )

    assert response.status == "error"
    assert response.needs_input == ["Project-Id"]


@pytest.mark.asyncio
async def test_multiple_people_interrupts_for_choice() -> None:
    tools = FakeToolsV3()
    tools.diagnose_results.append(
        {
            "people": [
                {"personId": "p1", "personName": ZHANG_SAN, "telephone": "13800000001"},
                {"personId": "p2", "personName": ZHANG_SAN, "telephone": "13800000002"},
            ]
        }
    )
    service = make_service(tools)

    response = await service.do_execute(
        AgentRequestV3(
            question=f"{ZHANG_SAN}\u4e09\u53f7\u95e8\u6253\u4e0d\u5f00",
            session_id="s-choice",
        ),
        "100",
    )

    assert response.status == "need_choice"
    assert [choice.id for choice in response.choices] == ["p1", "p2"]


@pytest.mark.asyncio
async def test_invalid_choice_resume_is_rejected() -> None:
    tools = FakeToolsV3()
    tools.diagnose_results.append(
        {
            "people": [
                {"personId": "p1", "personName": ZHANG_SAN},
                {"personId": "p2", "personName": ZHANG_SAN},
            ]
        }
    )
    service = make_service(tools)
    await service.do_execute(
        AgentRequestV3(
            question=f"{ZHANG_SAN}\u4e09\u53f7\u95e8\u6253\u4e0d\u5f00",
            session_id="s-invalid-choice",
        ),
        "100",
    )

    response = await service.do_execute(
        AgentRequestV3(
            session_id="s-invalid-choice",
            resume={"type": "choice", "choice_id": "missing"},
        ),
        "100",
    )

    assert response.status == "error"
    assert not any(name == "renew_permission" for name, _ in tools.calls)


@pytest.mark.asyncio
async def test_choice_resume_continues_to_diagnosis() -> None:
    tools = FakeToolsV3()
    tools.diagnose_results.extend(
        [
            {
                "people": [
                    {"personId": "p1", "personName": ZHANG_SAN},
                    {"personId": "p2", "personName": ZHANG_SAN},
                ]
            },
            {"diagnosis": {"mainCause": "DEVICE_OFFLINE", "mainCauseName": "\u8bbe\u5907\u79bb\u7ebf"}},
        ]
    )
    service = make_service(tools)
    await service.do_execute(
        AgentRequestV3(
            question=f"{ZHANG_SAN}\u4e09\u53f7\u95e8\u6253\u4e0d\u5f00",
            session_id="s-choice-continue",
        ),
        "100",
    )

    response = await service.do_execute(
        AgentRequestV3(
            session_id="s-choice-continue",
            resume={"type": "choice", "choice_id": "p1"},
        ),
        "100",
    )

    assert response.status == "ok"
    assert response.confirm is None
    assert response.data["diagnosis"]["mainCause"] == "DEVICE_OFFLINE"
    assert [name for name, _ in tools.calls].count("diagnose_access_issue") == 2


@pytest.mark.asyncio
async def test_permission_expired_interrupts_for_confirm() -> None:
    tools = FakeToolsV3()
    tools.diagnose_results.append(
        {"diagnosis": {"mainCause": "PERMISSION_EXPIRED", "mainCauseName": "\u6743\u9650\u8fc7\u671f"}}
    )
    service = make_service(tools)

    response = await service.do_execute(
        AgentRequestV3(
            question=f"{ZHANG_SAN}\u4e09\u53f7\u95e8\u6253\u4e0d\u5f00",
            session_id="s-confirm",
        ),
        "100",
    )

    assert response.status == "need_confirm"
    assert response.confirm is not None
    assert response.confirm.id == "renew_permission"


@pytest.mark.asyncio
async def test_confirm_resume_renews_only_after_permission_expired() -> None:
    tools = FakeToolsV3()
    tools.diagnose_results.append(
        {"diagnosis": {"mainCause": "PERMISSION_EXPIRED", "mainCauseName": "\u6743\u9650\u8fc7\u671f"}}
    )
    service = make_service(tools)
    await service.do_execute(
        AgentRequestV3(
            question=f"{ZHANG_SAN}\u4e09\u53f7\u95e8\u6253\u4e0d\u5f00",
            session_id="s-renew",
        ),
        "100",
    )

    response = await service.do_execute(
        AgentRequestV3(
            session_id="s-renew",
            resume={"type": "confirm", "confirm_id": "renew_permission", "confirmed": True},
        ),
        "100",
    )

    assert response.status == "ok"
    assert any(name == "renew_permission" for name, _ in tools.calls)


@pytest.mark.asyncio
async def test_non_permission_expired_does_not_confirm_or_renew() -> None:
    tools = FakeToolsV3()
    tools.diagnose_results.append(
        {"diagnosis": {"mainCause": "NO_CREDENTIAL", "mainCauseName": "\u65e0\u51ed\u8bc1"}}
    )
    service = make_service(tools)

    response = await service.do_execute(
        AgentRequestV3(
            question=f"{ZHANG_SAN}\u4e09\u53f7\u95e8\u6253\u4e0d\u5f00",
            session_id="s-no-renew",
        ),
        "100",
    )

    assert response.status == "ok"
    assert response.confirm is None
    assert not any(name == "renew_permission" for name, _ in tools.calls)
    assert "Project-Id" not in response.answer
    assert "project_id" not in response.answer


def test_v3_route_returns_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubService:
        async def do_execute(
            self,
            request: AgentRequestV3,
            project_id: str | None,
        ) -> AgentResponseV3:
            return AgentResponseV3(
                answer="ok",
                status="ok",
                session_id=request.session_id or "generated",
            )

    monkeypatch.setattr(right_routes_v3, "agent_service_v3", StubService())
    app = FastAPI()
    app.include_router(right_routes_v3.router)
    client = TestClient(app)

    response = client.post(
        "/api/web/agent/v3",
        headers={"Project-Id": "100"},
        json={"question": "hello", "session_id": "s-api"},
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == "s-api"
