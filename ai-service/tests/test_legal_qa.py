import unittest
from unittest.mock import AsyncMock, Mock, patch

import pytest
from app.agent.legal_qa.agent import LegalQaAgent
from app.agent.registry import AgentRegistry
from app.agent.runtime import AgentRuntime
from app.core.config import Settings
from app.main import create_app
from app.schemas.contracts import AgentContext, AgentValidationErrorResponse, Citation, LegalQaInput
from app.tools.law import FakeLawTool
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage


class _LawTool:
    async def search(self, query: str, *, request_id: str, trace_id: str | None) -> list[Citation]:
        return [
            Citation(
                document_id="law-1",
                kb_chunk_id="chunk-1",
                law_name="测试法",
                article_no="第一条",
                quoted_text="测试法条内容",
                relevance_score=0.9,
            ),
            Citation(document_id="untraceable", kb_chunk_id="chunk-2", quoted_text=" "),
        ]


class LegalQaAgentTest(unittest.IsolatedAsyncioTestCase):
    async def test_registry_and_runtime_invoke_legal_qa_with_no_results(self):
        registry = AgentRegistry()
        registry.register("legal_qa", LegalQaAgent(FakeLawTool()))
        result = await AgentRuntime(registry).invoke(
            agent_code="legal_qa",
            input={"question": "无命中样例"},
            context=AgentContext(requestId="legal-qa-runtime"),
        )
        self.assertIn("未找到直接法律依据", result.answer)
        self.assertEqual(result.citations, [])

    async def test_graph_validates_untraceable_citations_before_generation(self):
        model = Mock(ainvoke=AsyncMock(return_value=AIMessage(content=" 模拟回答 ")))
        law_tool = _LawTool()
        with patch.object(law_tool, "search", AsyncMock(wraps=law_tool.search)) as search:
            result = await LegalQaAgent(law_tool, model).invoke(
                LegalQaInput(question="测试问题"),
                AgentContext(requestId="legal-qa-graph", traceId="trace-legal-qa"),
            )

        search.assert_awaited_once_with("测试问题", request_id="legal-qa-graph", trace_id="trace-legal-qa")
        self.assertEqual(result.answer, "模拟回答")
        self.assertEqual([citation.document_id for citation in result.citations], ["law-1"])
        messages = model.ainvoke.await_args.args[0]
        self.assertIn("测试法条内容", messages[-1].content)
        self.assertNotIn("untraceable", messages[-1].content)

    async def test_empty_retrieval_skips_model(self):
        model = Mock(ainvoke=AsyncMock())
        result = await LegalQaAgent(FakeLawTool(), model).invoke(
            {"question": "无命中样例"}, AgentContext(requestId="empty-retrieval")
        )
        self.assertEqual(result.citations, [])
        model.ainvoke.assert_not_awaited()


@pytest.fixture
def payload():
    return {
        "agentCode": "legal_qa",
        "input": {"question": "合同违约金如何处理？"},
        "context": {"requestId": "legal-qa-api", "userId": "1", "sessionId": "2"},
    }


@pytest.fixture
def client():
    with TestClient(create_app(Settings(mode="fake", _env_file=None))) as client:
        yield client


def test_invoke_legal_qa_uses_common_agent_api(client, payload):
    with patch("app.main.create_chat_model") as create_model:
        response = client.post("/api/v1/agent/invoke", json=payload)

    create_model.assert_not_called()
    assert response.status_code == 200
    body = response.json()
    assert body["requestId"] == "legal-qa-api"
    assert body["data"]["answer"].startswith("离线测试")
    assert body["data"]["citations"][0]["kb_chunk_id"] == "fake-chunk-1"


def test_invoke_legal_qa_empty_retrieval(client, payload):
    payload["input"] = {"question": "无命中样例"}
    response = client.post("/api/v1/agent/invoke", json=payload)
    assert response.status_code == 200
    assert response.json()["data"] == {"answer": "未找到直接法律依据，无法给出确定性回答。", "citations": []}


def test_openapi_documents_agent_validation_response():
    document = create_app().openapi()
    operation = document["paths"]["/api/v1/agent/invoke"]["post"]
    response_schema = operation["responses"]["422"]["content"]["application/json"]["schema"]
    assert response_schema == {"$ref": "#/components/schemas/AgentValidationErrorResponse"}

    schemas = document["components"]["schemas"]
    envelope = schemas["AgentValidationErrorResponse"]
    assert set(envelope["required"]) == {"success", "error"}
    assert envelope["properties"]["success"]["const"] is False
    assert envelope["properties"]["error"]["$ref"] == "#/components/schemas/AgentValidationError"

    error = schemas["AgentValidationError"]
    assert set(error["required"]) == {"code", "message", "details"}
    assert set(error["properties"]["code"]["enum"]) == {"INVALID_REQUEST", "INVALID_AGENT_INPUT"}
    assert error["properties"]["details"]["items"]["$ref"] == "#/components/schemas/ValidationErrorDetail"
    assert set(schemas["ValidationErrorDetail"]["required"]) == {"loc", "type"}


@pytest.mark.parametrize(
    "invalid_input", [{}, {"question": " \t "}, {"question": 42}, {"question": "字" * 4001}]
)
def test_rejects_invalid_legal_qa_input(client, payload, invalid_input):
    payload["input"] = invalid_input
    response = client.post("/api/v1/agent/invoke", json=payload)

    assert response.status_code == 422
    body = AgentValidationErrorResponse.model_validate(response.json())
    assert body.model_dump() == response.json()
    assert body.error.code == "INVALID_AGENT_INPUT"
    assert body.error.details[0].loc == ["input", "question"]


@pytest.mark.parametrize(
    ("field", "value"), [("agentCode", " "), ("input", []), ("context", {"requestId": ""})]
)
def test_rejects_invalid_api_envelope(client, payload, field, value):
    payload[field] = value
    response = client.post("/api/v1/agent/invoke", json=payload)

    assert response.status_code == 422
    body = AgentValidationErrorResponse.model_validate(response.json())
    assert body.model_dump() == response.json()
    assert body.error.code == "INVALID_REQUEST"


def test_rejects_malformed_json(client):
    response = client.post(
        "/api/v1/agent/invoke", content='{"private":', headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
    assert "private" not in response.text


def test_unknown_agent_returns_404(client, payload):
    payload["agentCode"] = "contract_review"
    response = client.post("/api/v1/agent/invoke", json=payload)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "AGENT_NOT_FOUND"


def test_live_mode_requires_real_retrieval():
    for law_tool in (None, FakeLawTool()):
        with (
            pytest.raises(RuntimeError, match="live 模式需要注入真实法律检索实现"),
            TestClient(create_app(Settings(mode="live", _env_file=None), law_tool=law_tool)),
        ):
            pass


@pytest.mark.parametrize(
    "model_call",
    [
        AsyncMock(side_effect=TimeoutError("upstream private details")),
        AsyncMock(return_value=AIMessage(content=" ")),
    ],
)
def test_model_failure_has_structured_response(payload, model_call):
    model = Mock(ainvoke=model_call)
    settings = Settings(mode="live", _env_file=None)
    with (
        patch("app.main.create_chat_model", return_value=model),
        TestClient(create_app(settings, law_tool=_LawTool())) as client,
    ):
        response = client.post("/api/v1/agent/invoke", json=payload)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "LLM_UNAVAILABLE"
    assert "upstream private details" not in response.text
    model.ainvoke.assert_awaited_once()
