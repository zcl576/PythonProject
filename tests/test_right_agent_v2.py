from typing import Any

import pytest

from app.schemas.right_schema_v2 import AgentRequestV2
from app.services.right_agent_graph_service import RightAgentGraphService

ZHANG_SAN = "\u5f20\u4e09"


class FakeRepository:
    def __init__(self) -> None:
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.threads: list[tuple[str, int]] = []
        self.cleared: list[tuple[str, str]] = []

    async def ensure_thread(self, session_id: str, project_id: int) -> None:
        self.threads.append((session_id, project_id))

    async def save_snapshot(self, **kwargs: Any) -> None:
        self.snapshots[kwargs["session_id"]] = kwargs

    async def load_snapshot(self, session_id: str) -> dict[str, Any] | None:
        return self.snapshots.get(session_id)

    async def clear_snapshot(self, session_id: str, state: str = "done") -> None:
        self.cleared.append((session_id, state))
        self.snapshots.pop(session_id, None)


class FakeTools:
    def __init__(
        self,
        *,
        people: list[dict[str, Any]] | None = None,
        main_cause: str = "DEVICE_OFFLINE",
    ) -> None:
        self.people = people or []
        self.main_cause = main_cause
        self.renew_calls: list[tuple[int, dict[str, Any]]] = []

    async def search_person(
        self,
        project_id: int,
        *,
        telephone: str | None = None,
        person_name: str | None = None,
    ) -> dict[str, Any]:
        return {"people": self.people}

    async def diagnose_access_issue(
        self,
        project_id: int,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        if self.people and not slots.get("personId"):
            return {"people": self.people}
        return {
            "diagnosis": {
                "summary": "diagnosis summary",
                "mainCause": self.main_cause,
                "mainCauseName": "permission expired"
                if self.main_cause == "PERMISSION_EXPIRED"
                else "device offline",
            }
        }

    async def renew_permission(
        self,
        project_id: int,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        self.renew_calls.append((project_id, slots))
        return {"ok": True}


@pytest.mark.asyncio
async def test_missing_project_id_is_rejected() -> None:
    service = RightAgentGraphService(repository=FakeRepository(), tools=FakeTools())

    response = await service.do_execute(
        AgentRequestV2(question=f"\u67e5\u8be2{ZHANG_SAN}"),
        project_id=None,
    )

    assert response.status == "error"
    assert response.needs_input == ["Project-Id"]


@pytest.mark.asyncio
async def test_multiple_people_returns_choice_and_saves_snapshot() -> None:
    repo = FakeRepository()
    tools = FakeTools(
        people=[
            {"personId": "p1", "personName": ZHANG_SAN, "telephone": "13800000001"},
            {"personId": "p2", "personName": ZHANG_SAN, "telephone": "13800000002"},
        ]
    )
    service = RightAgentGraphService(repository=repo, tools=tools)

    response = await service.do_execute(
        AgentRequestV2(session_id="s1", question=f"\u67e5\u8be2{ZHANG_SAN}"),
        project_id="100",
    )

    assert response.status == "need_choice"
    assert [item.id for item in response.choices] == ["p1", "p2"]
    assert repo.snapshots["s1"]["state"] == "waiting_choice"


@pytest.mark.asyncio
async def test_invalid_choice_is_rejected() -> None:
    repo = FakeRepository()
    repo.snapshots["s1"] = {
        "state": "waiting_choice",
        "choices": [{"id": "p1", "data": {"personId": "p1"}}],
        "slots": {},
        "metadata": {"active_tool": "diagnose_access_issue"},
    }
    service = RightAgentGraphService(repository=repo, tools=FakeTools())

    response = await service.do_execute(
        AgentRequestV2(session_id="s1", choice_id="bad"),
        project_id="100",
    )

    assert response.status == "error"
    assert response.answer


@pytest.mark.asyncio
async def test_permission_expired_returns_confirm() -> None:
    repo = FakeRepository()
    service = RightAgentGraphService(
        repository=repo,
        tools=FakeTools(main_cause="PERMISSION_EXPIRED"),
    )

    response = await service.do_execute(
        AgentRequestV2(session_id="s1", question=f"{ZHANG_SAN}\u5f00\u4e0d\u4e86\u95e8"),
        project_id="100",
    )

    assert response.status == "need_confirm"
    assert response.confirm
    assert response.confirm.id == "renew_permission"
    assert repo.snapshots["s1"]["state"] == "waiting_confirm"


@pytest.mark.asyncio
async def test_non_permission_expired_does_not_return_confirm() -> None:
    tools = FakeTools(main_cause="DEVICE_OFFLINE")
    service = RightAgentGraphService(repository=FakeRepository(), tools=tools)

    response = await service.do_execute(
        AgentRequestV2(session_id="s1", question=f"{ZHANG_SAN}\u5f00\u4e0d\u4e86\u95e8"),
        project_id="100",
    )

    assert response.status == "ok"
    assert response.confirm is None
    assert tools.renew_calls == []


@pytest.mark.asyncio
async def test_confirm_requires_pending_permission_expired_snapshot() -> None:
    repo = FakeRepository()
    repo.snapshots["s1"] = {
        "state": "waiting_confirm",
        "pending_action": "renew_permission",
        "slots": {"personId": "p1"},
        "metadata": {"diagnosis": {"mainCause": "DEVICE_OFFLINE"}},
    }
    tools = FakeTools()
    service = RightAgentGraphService(repository=repo, tools=tools)

    response = await service.do_execute(
        AgentRequestV2(session_id="s1", confirm_id="renew_permission"),
        project_id="100",
    )

    assert response.status == "error"
    assert tools.renew_calls == []


@pytest.mark.asyncio
async def test_confirm_renews_when_snapshot_allows_it() -> None:
    repo = FakeRepository()
    repo.snapshots["s1"] = {
        "state": "waiting_confirm",
        "pending_action": "renew_permission",
        "slots": {"personId": "p1"},
        "metadata": {"diagnosis": {"mainCause": "PERMISSION_EXPIRED"}},
    }
    tools = FakeTools()
    service = RightAgentGraphService(repository=repo, tools=tools)

    response = await service.do_execute(
        AgentRequestV2(session_id="s1", confirm_id="renew_permission"),
        project_id="100",
    )

    assert response.status == "ok"
    assert tools.renew_calls == [(100, {"personId": "p1"})]
