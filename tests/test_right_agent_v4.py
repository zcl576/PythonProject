from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.api import right_routes_v4
from app.schemas.right_schema_v4 import AgentRequestV4, AgentResponseV4
from app.services.right_agent_metadata_v4 import TOOL_METADATA
from app.services.right_agent_graph_service_v4 import RightAgentGraphServiceV4

ZHANG_SAN = "\u5f20\u4e09"


class FakeToolsV4:
    def __init__(self) -> None:
        self.results: dict[str, list[dict[str, Any]]] = {
            "search_person": [],
            "diagnose_access_issue": [],
            "renew_permission": [],
        }
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute(
        self,
        tool_name: str,
        project_id: int,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append((tool_name, {"project_id": project_id, "slots": slots}))
        metadata = TOOL_METADATA[tool_name]
        if not metadata.configured:
            return {
                "status": "not_configured",
                "tool": tool_name,
                "message": f"{metadata.description}\u80fd\u529b\u5c1a\u672a\u914d\u7f6e\u540e\u7aef\u63a5\u53e3\u3002",
            }
        queued = self.results.get(tool_name) or []
        if queued:
            return queued.pop(0)
        return {"ok": True}


def make_service(tools: FakeToolsV4) -> RightAgentGraphServiceV4:
    return RightAgentGraphServiceV4(tools=tools, checkpointer=InMemorySaver())


@pytest.mark.asyncio
async def test_missing_project_id_is_rejected() -> None:
    service = make_service(FakeToolsV4())

    response = await service.do_execute(
        AgentRequestV4(question=f"{ZHANG_SAN}\u4e09\u53f7\u95e8\u6253\u4e0d\u5f00"),
        None,
    )

    assert response.status == "error"
    assert response.needs_input == ["Project-Id"]


@pytest.mark.asyncio
async def test_unknown_intent_is_unsupported_without_tool_call() -> None:
    tools = FakeToolsV4()
    service = make_service(tools)

    response = await service.do_execute(
        AgentRequestV4(question="\u5e2e\u6211\u8ba2\u4e00\u4efd\u5916\u5356"),
        "100",
    )

    assert response.status == "unsupported"
    assert tools.calls == []
    assert response.data["supported_capabilities"]


@pytest.mark.asyncio
async def test_existing_person_id_does_not_call_search_person_resolver() -> None:
    tools = FakeToolsV4()
    service = make_service(tools)

    response = await service.do_execute(
        AgentRequestV4(question="personId P1 \u4e09\u53f7\u95e8\u6743\u9650"),
        "100",
    )

    assert response.status == "not_configured"
    assert not any(name == "search_person" for name, _ in tools.calls)


@pytest.mark.asyncio
async def test_missing_person_id_uses_search_person_resolver() -> None:
    tools = FakeToolsV4()
    tools.results["search_person"].append(
        {"context": {"personId": "p1", "personName": ZHANG_SAN}}
    )
    service = make_service(tools)

    response = await service.do_execute(
        AgentRequestV4(question=f"{ZHANG_SAN} deviceId D1 \u6743\u9650"),
        "100",
    )

    assert response.status == "not_configured"
    assert any(name == "search_person" for name, _ in tools.calls)


@pytest.mark.asyncio
async def test_multiple_people_interrupts_for_choice() -> None:
    tools = FakeToolsV4()
    tools.results["search_person"].append(
        {
            "people": [
                {"personId": "p1", "personName": ZHANG_SAN},
                {"personId": "p2", "personName": ZHANG_SAN},
            ]
        }
    )
    service = make_service(tools)

    response = await service.do_execute(
        AgentRequestV4(
            question=f"{ZHANG_SAN} deviceId D1 \u6743\u9650",
            session_id="s-choice-v4",
        ),
        "100",
    )

    assert response.status == "need_choice"
    assert [choice.id for choice in response.choices] == ["p1", "p2"]


@pytest.mark.asyncio
async def test_choice_resume_continues_after_resolver() -> None:
    tools = FakeToolsV4()
    tools.results["search_person"].append(
        {
            "people": [
                {"personId": "p1", "personName": ZHANG_SAN},
                {"personId": "p2", "personName": ZHANG_SAN},
            ]
        }
    )
    service = make_service(tools)
    await service.do_execute(
        AgentRequestV4(
            question=f"{ZHANG_SAN} deviceId D1 \u6743\u9650",
            session_id="s-choice-resume-v4",
        ),
        "100",
    )

    response = await service.do_execute(
        AgentRequestV4(
            session_id="s-choice-resume-v4",
            resume={"type": "choice", "choice_id": "p1"},
        ),
        "100",
    )

    assert response.status == "not_configured"
    assert response.data["policy"]["reason"] == "not_configured"


@pytest.mark.asyncio
async def test_invalid_choice_resume_returns_error() -> None:
    tools = FakeToolsV4()
    tools.results["search_person"].append(
        {
            "people": [
                {"personId": "p1", "personName": ZHANG_SAN},
                {"personId": "p2", "personName": ZHANG_SAN},
            ]
        }
    )
    service = make_service(tools)
    await service.do_execute(
        AgentRequestV4(
            question=f"{ZHANG_SAN} deviceId D1 \u6743\u9650",
            session_id="s-invalid-choice-v4",
        ),
        "100",
    )

    response = await service.do_execute(
        AgentRequestV4(
            session_id="s-invalid-choice-v4",
            resume={"type": "choice", "choice_id": "missing"},
        ),
        "100",
    )

    assert response.status == "error"


@pytest.mark.asyncio
async def test_permission_expired_interrupts_for_confirm() -> None:
    tools = FakeToolsV4()
    tools.results["diagnose_access_issue"].append(
        {"diagnosis": {"mainCause": "PERMISSION_EXPIRED", "mainCauseName": "\u6743\u9650\u8fc7\u671f"}}
    )
    service = make_service(tools)

    response = await service.do_execute(
        AgentRequestV4(
            question=f"{ZHANG_SAN}\u4e09\u53f7\u95e8\u6253\u4e0d\u5f00",
            session_id="s-confirm-v4",
        ),
        "100",
    )

    assert response.status == "need_confirm"
    assert response.confirm is not None
    assert response.confirm.id == "renew_permission"


@pytest.mark.asyncio
async def test_permission_expired_confirm_calls_renew() -> None:
    tools = FakeToolsV4()
    tools.results["diagnose_access_issue"].append(
        {"diagnosis": {"mainCause": "PERMISSION_EXPIRED", "mainCauseName": "\u6743\u9650\u8fc7\u671f"}}
    )
    tools.results["renew_permission"].append({"renewed": True})
    service = make_service(tools)
    await service.do_execute(
        AgentRequestV4(
            question=f"{ZHANG_SAN}\u4e09\u53f7\u95e8\u6253\u4e0d\u5f00",
            session_id="s-renew-v4",
        ),
        "100",
    )

    response = await service.do_execute(
        AgentRequestV4(
            session_id="s-renew-v4",
            resume={"type": "confirm", "confirm_id": "renew_permission", "confirmed": True},
        ),
        "100",
    )

    assert response.status == "ok"
    assert any(name == "renew_permission" for name, _ in tools.calls)


@pytest.mark.asyncio
async def test_non_permission_expired_does_not_confirm_or_renew() -> None:
    tools = FakeToolsV4()
    tools.results["diagnose_access_issue"].append(
        {"diagnosis": {"mainCause": "DEVICE_OFFLINE", "mainCauseName": "\u8bbe\u5907\u79bb\u7ebf"}}
    )
    service = make_service(tools)

    response = await service.do_execute(
        AgentRequestV4(
            question=f"{ZHANG_SAN}\u4e09\u53f7\u95e8\u6253\u4e0d\u5f00",
            session_id="s-no-renew-v4",
        ),
        "100",
    )

    assert response.status == "ok"
    assert response.confirm is None
    assert not any(name == "renew_permission" for name, _ in tools.calls)


@pytest.mark.asyncio
async def test_unconfigured_tool_returns_not_configured() -> None:
    service = make_service(FakeToolsV4())

    response = await service.do_execute(
        AgentRequestV4(question="\u4e09\u53f7\u95e8\u8bbe\u5907\u5728\u7ebf\u5417"),
        "100",
    )

    assert response.status == "not_configured"


@pytest.mark.asyncio
async def test_write_action_without_valid_condition_is_policy_denied() -> None:
    tools = FakeToolsV4()
    service = make_service(tools)

    response = await service.do_execute(
        AgentRequestV4(question=f"\u5e2e{ZHANG_SAN}\u7eed\u671f"),
        "100",
    )

    assert response.status == "policy_denied"
    assert not any(name == "renew_permission" for name, _ in tools.calls)


def test_v4_route_returns_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubService:
        async def do_execute(
            self,
            request: AgentRequestV4,
            project_id: str | None,
        ) -> AgentResponseV4:
            return AgentResponseV4(
                answer="ok",
                status="ok",
                session_id=request.session_id or "generated",
            )

    monkeypatch.setattr(right_routes_v4, "agent_service_v4", StubService())
    app = FastAPI()
    app.include_router(right_routes_v4.router)
    client = TestClient(app)

    response = client.post(
        "/api/web/agent/v4",
        headers={"Project-Id": "100"},
        json={"question": "hello", "session_id": "s-api-v4"},
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == "s-api-v4"
