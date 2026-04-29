from typing import Any

import httpx
import pytest

from app.memory.local_memory import get_inmemory_session
from app.schemas.diagnosis import DiagnosisAgentRequest
from app.services.access_diagnosis_agent import AccessDiagnosisAgentService


class FakeEstateAiClient:
    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[int, dict[str, Any]]] = []
        self.result = result or {
            "context": {"personName": "张三"},
            "diagnosis": {
                "summary": "人员权限已过期",
                "mainCause": "PERMISSION_EXPIRED",
                "mainCauseName": "权限过期",
                "confidence": 0.91,
                "evidences": ["权限截止时间早于当前时间"],
                "suggestedActions": [
                    {
                        "action": "RENEW_PERMISSION",
                        "actionName": "续期权限",
                        "description": "为该人员续期门禁权限",
                        "riskLevel": "medium",
                    }
                ],
            },
        }

    async def get_result(self, project_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((project_id, payload))
        return self.result


class DisabledLlm:
    enabled = False

    async def extract_diagnosis_fields(self, question: str) -> dict[str, Any] | None:
        return None

    async def explain_diagnosis(
        self,
        question: str | None,
        normalized: dict[str, Any],
        result: dict[str, Any],
    ) -> str | None:
        return None


class FailingLlm:
    enabled = True

    async def extract_diagnosis_fields(self, question: str) -> dict[str, Any] | None:
        raise httpx.HTTPError("llm down")

    async def explain_diagnosis(
        self,
        question: str | None,
        normalized: dict[str, Any],
        result: dict[str, Any],
    ) -> str | None:
        raise httpx.HTTPError("llm down")


@pytest.mark.asyncio
async def test_need_more_info_when_request_has_no_identifiers() -> None:
    service = AccessDiagnosisAgentService(client=FakeEstateAiClient(), llm=DisabledLlm())

    response = await service.diagnose(DiagnosisAgentRequest(project_id=1, question="帮我看看为什么打不开门"))

    assert response.status == "need_more_info"
    assert response.main_cause == "CONTEXT_INSUFFICIENT"
    assert response.follow_up_question
    assert response.trace is not None
    assert [event.stage for event in response.trace.events] == ["intent", "extract", "memory", "clarify"]


@pytest.mark.asyncio
async def test_rule_extracts_phone_and_calls_diagnosis_tool() -> None:
    client = FakeEstateAiClient()
    service = AccessDiagnosisAgentService(client=client, llm=DisabledLlm())

    response = await service.diagnose(
        DiagnosisAgentRequest(project_id=7, question="手机号13800138000为什么打不开门")
    )

    assert response.status == "done"
    assert response.normalized_request["telephone"] == "13800138000"
    assert response.main_cause == "PERMISSION_EXPIRED"
    assert response.available_actions[0]["need_confirm"] is True
    assert client.calls == [(7, response.normalized_request)]


@pytest.mark.asyncio
async def test_session_context_is_merged_across_turns() -> None:
    client = FakeEstateAiClient()
    service = AccessDiagnosisAgentService(
        client=client,
        llm=DisabledLlm(),
        session_store=get_inmemory_session(),
    )

    first = await service.diagnose(
        DiagnosisAgentRequest(project_id=1, session_id="s1", question="手机号13800138000打不开门")
    )
    second = await service.diagnose(
        DiagnosisAgentRequest(project_id=1, session_id="s1", question="deviceId D001", device_id="D001")
    )

    assert first.normalized_request["telephone"] == "13800138000"
    assert second.normalized_request["telephone"] == "13800138000"
    assert second.normalized_request["deviceId"] == "D001"


@pytest.mark.asyncio
async def test_llm_failures_fall_back_to_rules() -> None:
    service = AccessDiagnosisAgentService(client=FakeEstateAiClient(), llm=FailingLlm())

    response = await service.diagnose(
        DiagnosisAgentRequest(project_id=1, question="手机号13800138000为什么打不开门")
    )

    assert response.status == "done"
    assert response.llm_used is False
    assert response.agent_mode == "rule"
    assert len(response.warnings) == 2
    assert response.answer.startswith("诊断结论：权限过期。")
