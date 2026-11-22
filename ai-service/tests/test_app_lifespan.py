"""验证应用装配与关闭，模型和数据库均由替身提供。"""

import asyncio
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest
from app import main
from app.core.config import Settings
from app.schemas.contracts import AgentContext, Citation
from app.tools.citation_validator import CitationReferenceError
from app.tools.law import FakeLawTool
from app.tools.law_splitter import split_law_articles
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

ENDPOINT = "/api/v1/agent/invoke"


def _request(request_id: str) -> dict:
    return {
        "agentCode": "legal_qa",
        "input": {"question": "合同违约金如何处理？"},
        "context": {"requestId": request_id},
    }


def _vector_settings(resources, **overrides) -> Settings:
    values = {
        "mode": "fake",
        "law_search_backend": "vector",
        "bge_model_path": "models/local-bge",
        "milvus_db_path": str(resources.db_path),
        "milvus_collection_name": "legal_chunks",
        "law_search_top_k": 5,
        "citation_reference_source_path": str(resources.source_path),
        "citation_reference_metadata_path": str(resources.metadata_path),
    }
    return Settings(_env_file=None, **(values | overrides))


@pytest.fixture
def resources(monkeypatch, tmp_path):
    db_path = tmp_path / "legal.db"
    # 用空的 collections 目录模拟已初始化的目录布局，所有 SDK 调用仍经过替身。
    (db_path / "collections").mkdir(parents=True)
    # 服务从正式样本独立读取参考快照。这里构造 Milvus 替身的命中记录。
    fixture = Path(__file__).with_name("fixtures") / "civil_code_2020_articles_577_584"
    source_path = fixture.with_suffix(".txt")
    metadata_path = fixture.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    chunks = split_law_articles(
        source_path.read_bytes().decode("utf-8"),
        document_id=metadata["document_id"],
        law_name=metadata["law_name"],
        source_url=metadata["source_url"],
    )
    chunk = next(chunk for chunk in chunks if chunk.article_no == "第五百八十四条")
    encoder = Mock(spec_set=["encode_query"])
    encoder.encode_query.return_value = [1.0] + [0.0] * 511
    store = Mock(spec_set=["search", "close"])
    store.search.return_value = [(chunk, 0.85)]
    create_embedding = Mock(return_value=encoder)
    open_store = Mock(return_value=store)
    create_model = Mock(side_effect=AssertionError("此用例不应配置 DeepSeek"))
    monkeypatch.setattr(main, "BgeEmbedding", create_embedding)
    monkeypatch.setattr(main, "MilvusStore", open_store)
    monkeypatch.setattr(main, "create_chat_model", create_model)
    return SimpleNamespace(
        db_path=db_path,
        source_path=source_path,
        metadata_path=metadata_path,
        chunk=chunk,
        other_chunk=chunks[0],
        encoder=encoder,
        store=store,
        embed=create_embedding,
        open=open_store,
        llm=create_model,
    )


def test_default_fake_startup_uses_no_model_or_store(monkeypatch, resources):
    monkeypatch.delenv("AI_SERVICE_MODE", raising=False)
    monkeypatch.delenv("AI_SERVICE_LAW_SEARCH_BACKEND", raising=False)
    monkeypatch.delenv("AI_SERVICE_LAW_SEARCH_TOP_K", raising=False)
    load_validator = Mock(side_effect=AssertionError("默认 Fake 不应读取法律快照"))
    monkeypatch.setattr(main, "_load_citation_validator", load_validator)
    settings = Settings(_env_file=None)
    assert (settings.mode, settings.law_search_backend, settings.law_search_top_k) == ("fake", "fake", 3)

    with TestClient(main.create_app(settings)) as client:
        response = client.post(ENDPOINT, json=_request("fake-default"))

    assert response.status_code == 200
    assert response.json()["data"]["citations"][0]["kb_chunk_id"] == "fake-chunk-1"
    resources.embed.assert_not_called()
    resources.open.assert_not_called()
    resources.llm.assert_not_called()
    load_validator.assert_not_called()


def test_vector_settings_use_existing_environment_prefix(monkeypatch):
    settings_env = {
        "AI_SERVICE_MODE": "fake",
        "AI_SERVICE_LAW_SEARCH_BACKEND": "vector",
        "AI_SERVICE_BGE_MODEL_PATH": "bge-snapshot",
        "AI_SERVICE_MILVUS_DB_PATH": "legal.db",
        "AI_SERVICE_MILVUS_COLLECTION_NAME": "legal",
        "AI_SERVICE_LAW_SEARCH_TOP_K": "4",
        "AI_SERVICE_CITATION_REFERENCE_SOURCE_PATH": "verified-law.txt",
        "AI_SERVICE_CITATION_REFERENCE_METADATA_PATH": "verified-law.json",
    }
    for name, value in settings_env.items():
        monkeypatch.setenv(name, value)

    settings = Settings(_env_file=None)

    assert settings.mode == "fake"
    assert settings.law_search_backend == "vector"
    assert settings.bge_model_path == "bge-snapshot"
    assert settings.milvus_db_path == "legal.db"
    assert settings.milvus_collection_name == "legal"
    assert settings.law_search_top_k == 4
    assert settings.citation_reference_source_path == "verified-law.txt"
    assert settings.citation_reference_metadata_path == "verified-law.json"


def test_vector_lifespan_reuses_dependencies_and_does_not_create_llm(resources):
    app = main.create_app(_vector_settings(resources))
    resources.embed.assert_not_called()
    resources.open.assert_not_called()

    with TestClient(app) as client:
        for request_id in ("vector-1", "vector-2"):
            response = client.post(ENDPOINT, json=_request(request_id))
            assert response.status_code == 200
            body = response.json()
            assert body["requestId"] == request_id
            assert body["data"]["answer"].startswith("离线测试")
            citation = body["data"]["citations"][0]
            assert citation["kb_chunk_id"] == resources.chunk.kb_chunk_id
            assert citation["quoted_text"] == resources.chunk.text
        resources.store.close.assert_not_called()

    resources.embed.assert_called_once_with("models/local-bge")
    resources.open.assert_called_once_with(str(resources.db_path), "legal_chunks", create=False)
    assert resources.encoder.encode_query.call_count == 2
    expected = call(resources.encoder.encode_query.return_value, top_k=5)
    assert resources.store.search.call_args_list == [expected, expected]
    resources.llm.assert_not_called()
    resources.store.close.assert_called_once_with()
    assert not hasattr(app.state, "agent_runtime")


def _supported_decision(chunk_id: str) -> AIMessage:
    return AIMessage(content=json.dumps({"status": "supported", "citation_ids": [chunk_id]}))


def test_live_vector_generates_with_only_the_selected_citation(resources):
    chunk = resources.chunk
    other_chunk = resources.other_chunk
    resources.store.search.return_value = [(other_chunk, 0.9), (chunk, 0.8)]
    llm = Mock(
        ainvoke=AsyncMock(side_effect=[_supported_decision(chunk.kb_chunk_id), AIMessage(content="模拟回答")])
    )
    resources.llm.side_effect = None
    resources.llm.return_value = llm
    settings = _vector_settings(resources, mode="live")
    request = _request("vector-live")
    request["input"]["question"] = "合同违约造成的损失包括可获得的利益吗？"

    with TestClient(main.create_app(settings)) as client:
        response = client.post(ENDPOINT, json=request)

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["answer"] == "模拟回答"
    assert [citation["kb_chunk_id"] for citation in data["citations"]] == [chunk.kb_chunk_id]
    assert data["citations"][0]["quoted_text"] == chunk.text
    resources.llm.assert_called_once_with(settings)
    assert llm.ainvoke.await_count == 2
    judge_call, generation_call = llm.ainvoke.await_args_list
    # 判断看原问题和两条真实候选。生成只看到选中的引用，即使它的检索分数更低。
    judge_input = json.loads(judge_call.args[0][-1].content)
    assert judge_input["question"] == request["input"]["question"]
    assert [candidate["kb_chunk_id"] for candidate in judge_input["candidates"]] == [
        other_chunk.kb_chunk_id,
        chunk.kb_chunk_id,
    ]
    assert judge_input["candidates"][1]["quoted_text"] == chunk.text
    assert all("relevance_score" not in candidate for candidate in judge_input["candidates"])
    generation_prompt = generation_call.args[0][-1].content
    assert chunk.kb_chunk_id in generation_prompt
    assert other_chunk.kb_chunk_id not in generation_prompt
    resources.store.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("status", "answer"),
    [
        ("insufficient", "当前核验通过的法条不足以直接回答该问题。"),
        ("uncertain", "无法可靠判断现有法条是否足以回答该问题。"),
    ],
)
def test_live_vector_skips_generation_without_sufficient_evidence(resources, status, answer):
    decision = AIMessage(content=json.dumps({"status": status, "citation_ids": []}))
    llm = Mock(ainvoke=AsyncMock(return_value=decision))
    resources.llm.side_effect = None
    resources.llm.return_value = llm

    with TestClient(main.create_app(_vector_settings(resources, mode="live"))) as client:
        response = client.post(ENDPOINT, json=_request("evidence-" + status))

    assert response.status_code == 200
    assert response.json()["data"] == {"answer": answer, "citations": []}
    llm.ainvoke.assert_awaited_once()


@pytest.mark.parametrize(
    "raw_decision",
    [
        pytest.param('{"status":"supported","citation_ids":["<known>","forged"]}', id="unknown-id"),
        pytest.param('{"status":"supported","citation_ids":[]}', id="empty-selection"),
        pytest.param('{"status":"uncertain","citation_ids":["<known>"]}', id="contradictory-selection"),
        pytest.param('{"status":"supported","citation_ids":["<known>","<known>"]}', id="duplicate-id"),
        pytest.param(
            '{"status":"insufficient","status":"supported","citation_ids":["<known>"]}', id="duplicate-key"
        ),
        pytest.param('{"status":"supported","citation_ids":[1]}', id="invalid-id-type"),
        pytest.param('{"status":"yes","citation_ids":["<known>"]}', id="invalid-status"),
        pytest.param(
            '{"status":"supported","citation_ids":["<known>"],"quoted_text":"模型补写的法条"}',
            id="unexpected-field",
        ),
        pytest.param("结论为 supported", id="not-json"),
    ],
)
def test_live_vector_fails_closed_on_invalid_evidence_decision(resources, raw_decision):
    # 即使某个 ID 合法，整份判断无效时也不传给生成模型。
    response_text = raw_decision.replace("<known>", resources.chunk.kb_chunk_id)
    llm = Mock(ainvoke=AsyncMock(return_value=AIMessage(content=response_text)))
    resources.llm.side_effect = None
    resources.llm.return_value = llm

    with TestClient(main.create_app(_vector_settings(resources, mode="live"))) as client:
        response = client.post(ENDPOINT, json=_request("invalid-evidence-decision"))

    assert response.status_code == 200
    assert response.json()["data"] == {"answer": "无法可靠判断现有法条是否足以回答该问题。", "citations": []}
    llm.ainvoke.assert_awaited_once()


def test_live_vector_evidence_service_failure_is_503(resources):
    llm = Mock(ainvoke=AsyncMock(side_effect=TimeoutError("private upstream details")))
    resources.llm.side_effect = None
    resources.llm.return_value = llm

    with TestClient(main.create_app(_vector_settings(resources, mode="live"))) as client:
        response = client.post(ENDPOINT, json=_request("evidence-service-failure"))

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "LLM_UNAVAILABLE"
    assert "private upstream details" not in response.text
    llm.ainvoke.assert_awaited_once()
    resources.store.close.assert_called_once_with()


@pytest.mark.parametrize("include_valid", [True, False], ids=["mixed-citations", "all-rejected"])
def test_vector_citation_filter_applies_before_llm_and_response(resources, caplog, include_valid):
    chunk = resources.chunk
    forged_text = "篡改后的引用正文"
    forged_id = "forged-chunk-id"
    hits = [
        (chunk.model_copy(update={"text": forged_text}), 0.99),
        (chunk.model_copy(update={"kb_chunk_id": forged_id}), 0.98),
    ]
    if include_valid:
        hits.insert(1, (chunk, 0.8))
    resources.store.search.return_value = hits
    llm = Mock(
        ainvoke=AsyncMock(side_effect=[_supported_decision(chunk.kb_chunk_id), AIMessage(content="模拟回答")])
    )
    resources.llm.side_effect = None
    resources.llm.return_value = llm
    request = _request("citation-filter")

    with TestClient(main.create_app(_vector_settings(resources, mode="live"))) as client:
        response = client.post(ENDPOINT, json=request)

    assert response.status_code == 200
    data = response.json()["data"]
    if include_valid:
        assert [citation["kb_chunk_id"] for citation in data["citations"]] == [chunk.kb_chunk_id]
        assert data["citations"][0]["quoted_text"] == chunk.text
        assert data["answer"] == "模拟回答"
        assert llm.ainvoke.await_count == 2
        for model_call in llm.ainvoke.await_args_list:
            prompt = model_call.args[0][-1].content
            assert chunk.text.strip() in prompt
            assert forged_text not in prompt and forged_id not in prompt
    else:
        assert data == {"answer": "未找到直接法律依据，无法给出确定性回答。", "citations": []}
        llm.ainvoke.assert_not_awaited()

    records = [record for record in caplog.records if record.name == "app.agent.legal_qa.nodes"]
    assert [record.args for record in records] == [(2, {"quoted_text_mismatch": 1, "unknown_chunk": 1})]
    assert forged_text not in caplog.text and forged_id not in caplog.text
    assert chunk.text.strip() not in caplog.text
    assert request["input"]["question"] not in caplog.text
    resources.store.close.assert_called_once_with()


def test_explicit_law_tool_overrides_vector_settings_and_is_not_closed(resources):
    tool = Mock(
        search=AsyncMock(
            return_value=[Citation(document_id="external", kb_chunk_id="external-1", quoted_text="原文")]
        )
    )
    settings = _vector_settings(
        resources,
        bge_model_path="",
        milvus_db_path="",
        milvus_collection_name="",
        citation_reference_source_path="",
        citation_reference_metadata_path="",
    )

    with TestClient(main.create_app(settings, law_tool=tool)) as client:
        response = client.post(ENDPOINT, json=_request("injected"))

    assert response.status_code == 200
    assert response.json()["data"]["citations"][0]["kb_chunk_id"] == "external-1"
    tool.search.assert_awaited_once_with("合同违约金如何处理？", request_id="injected", trace_id=None)
    tool.close.assert_not_called()
    resources.embed.assert_not_called()
    resources.open.assert_not_called()
    resources.llm.assert_not_called()


def test_explicit_vector_tool_requires_snapshot_and_remains_externally_owned(resources):
    tool = main.VectorLawSearch(resources.encoder, resources.store)
    settings = Settings(
        mode="fake", law_search_backend="fake", citation_reference_source_path="", _env_file=None
    )
    with (
        pytest.raises(CitationReferenceError, match="AI_SERVICE_CITATION_REFERENCE_SOURCE_PATH"),
        TestClient(main.create_app(settings, law_tool=tool)),
    ):
        pytest.fail("显式注入的向量检索器也应校验引用")

    settings = _vector_settings(resources, law_search_backend="fake")
    resources.store.search.return_value = [
        (resources.chunk.model_copy(update={"kb_chunk_id": "forged-chunk"}), 0.99)
    ]
    with TestClient(main.create_app(settings, law_tool=tool)) as client:
        response = client.post(ENDPOINT, json=_request("external-vector"))

    assert response.status_code == 200
    assert response.json()["data"]["citations"] == []
    resources.embed.assert_not_called()
    resources.open.assert_not_called()
    resources.store.close.assert_not_called()


def test_live_still_rejects_explicit_fake_tool_when_vector_is_configured(resources):
    with (
        pytest.raises(RuntimeError, match="live 模式需要注入真实法律检索实现"),
        TestClient(main.create_app(_vector_settings(resources, mode="live"), law_tool=FakeLawTool())),
    ):
        pytest.fail("live 模式不应启动")
    resources.embed.assert_not_called()
    resources.open.assert_not_called()
    resources.llm.assert_not_called()


@pytest.mark.parametrize(
    "field",
    ["bge_model_path", "milvus_db_path", "milvus_collection_name"],
)
def test_missing_vector_configuration_rejects_startup_before_initialization(resources, field):
    with (
        pytest.raises(RuntimeError, match=f"AI_SERVICE_{field.upper()}"),
        TestClient(main.create_app(_vector_settings(resources, **{field: " "}))),
    ):
        pytest.fail("配置缺失时不应启动")
    resources.embed.assert_not_called()
    resources.open.assert_not_called()
    resources.llm.assert_not_called()


@pytest.mark.parametrize("field", ["citation_reference_source_path", "citation_reference_metadata_path"])
def test_missing_snapshot_configuration_rejects_startup_before_opening_dependencies(resources, field):
    settings = _vector_settings(resources, **{field: " "})
    app = main.create_app(settings)

    with pytest.raises(CitationReferenceError, match=f"AI_SERVICE_{field.upper()}"), TestClient(app):
        pytest.fail("快照配置缺失时不应启动")

    resources.embed.assert_not_called()
    resources.open.assert_not_called()
    resources.llm.assert_not_called()
    assert not hasattr(app.state, "agent_runtime")


@pytest.mark.parametrize(
    ("field", "name", "content", "cause"),
    [
        ("citation_reference_source_path", "missing.txt", None, FileNotFoundError),
        ("citation_reference_metadata_path", "missing.json", None, FileNotFoundError),
        ("citation_reference_metadata_path", "malformed.json", "{", json.JSONDecodeError),
        ("citation_reference_metadata_path", "unversioned.json", "{}", ValueError),
        ("citation_reference_source_path", "empty.txt", "", ValueError),
    ],
)
def test_unreadable_snapshot_fails_before_opening_dependencies(
    resources, tmp_path, field, name, content, cause
):
    path = tmp_path / name
    if content is not None:
        path.write_text(content, encoding="utf-8")
    settings = _vector_settings(resources, **{field: str(path)})

    with (
        pytest.raises(CitationReferenceError, match="无法加载") as error,
        TestClient(main.create_app(settings)),
    ):
        pytest.fail("无法加载可信快照时不应启动")

    assert isinstance(error.value.__cause__, cause)
    resources.embed.assert_not_called()
    resources.open.assert_not_called()
    resources.llm.assert_not_called()


@pytest.mark.parametrize("create_empty_directory", [False, True])
def test_vector_never_initializes_an_unprepared_database_path(tmp_path, resources, create_empty_directory):
    db_path = tmp_path / "unprepared.db"
    if create_empty_directory:
        db_path.mkdir()

    with (
        pytest.raises(FileNotFoundError, match="未找到已初始化的 Milvus Lite 数据库目录"),
        TestClient(main.create_app(_vector_settings(resources, milvus_db_path=str(db_path)))),
    ):
        pytest.fail("数据库未入库时不应启动")

    assert db_path.exists() == create_empty_directory
    if create_empty_directory:
        assert not any(db_path.iterdir())
    resources.embed.assert_not_called()
    resources.open.assert_not_called()


@pytest.mark.parametrize(
    ("factory", "error"),
    [
        ("embed", FileNotFoundError("BGE model snapshot does not exist")),
        ("open", FileNotFoundError("数据库不存在")),
        ("open", LookupError("Collection 不存在")),
    ],
)
def test_backend_initialization_errors_abort_without_fallback(resources, factory, error):
    getattr(resources, factory).side_effect = error
    with (
        pytest.raises(type(error)) as failure,
        TestClient(main.create_app(_vector_settings(resources))),
    ):
        pytest.fail("依赖初始化失败时不应启动")
    assert failure.value is error
    resources.store.close.assert_not_called()
    resources.llm.assert_not_called()
    if factory == "embed":
        resources.open.assert_not_called()
    else:
        resources.open.assert_called_once_with(str(resources.db_path), "legal_chunks", create=False)


@pytest.mark.parametrize("stage", ["adapter", "llm"])
def test_startup_failure_after_opening_store_closes_it(monkeypatch, resources, stage):
    error = RuntimeError("装配失败")
    settings = _vector_settings(resources)
    if stage == "adapter":
        monkeypatch.setattr(main, "VectorLawSearch", Mock(side_effect=error))
    else:
        settings = _vector_settings(resources, mode="live")
        resources.llm.side_effect = error
    app = main.create_app(settings)

    with pytest.raises(RuntimeError, match="装配失败"), TestClient(app):
        pytest.fail("装配失败时不应启动")
    resources.store.close.assert_called_once_with()
    assert not hasattr(app.state, "agent_runtime")


def test_request_error_still_closes_owned_store(resources):
    resources.store.search.side_effect = RuntimeError("存储查询失败")
    app = main.create_app(_vector_settings(resources))
    with pytest.raises(RuntimeError, match="存储查询失败"), TestClient(app) as client:
        client.post(ENDPOINT, json=_request("failed-request"))
    resources.store.close.assert_called_once_with()


def test_reference_failure_during_validation_is_a_server_error(monkeypatch, resources):
    error = CitationReferenceError("参考数据不可用")
    monkeypatch.setattr(main.CitationValidator, "validate", Mock(side_effect=error))
    llm = Mock(ainvoke=AsyncMock())
    resources.llm.side_effect = None
    resources.llm.return_value = llm
    app = main.create_app(_vector_settings(resources, mode="live"))

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(ENDPOINT, json=_request("reference-failure"))

    assert response.status_code == 500
    llm.ainvoke.assert_not_awaited()
    resources.store.close.assert_called_once_with()


def test_shutdown_waits_for_worker_after_request_cancellation(resources):
    app = main.create_app(_vector_settings(resources))

    async def exercise():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        shutting_down = asyncio.Event()
        release = threading.Event()

        def block_search(*args, **kwargs):
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5)
            return resources.store.search.return_value

        resources.store.search.side_effect = block_search

        async def run_lifespan():
            async with app.router.lifespan_context(app):
                # 直接取消本循环的请求协程，模拟请求已结束而同步线程尚未退出的情况。
                runtime = app.state.agent_runtime
                request = asyncio.create_task(
                    runtime.invoke(
                        "legal_qa", _request("cancelled")["input"], AgentContext(requestId="cancelled")
                    )
                )
                await asyncio.wait_for(started.wait(), timeout=5)
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
                shutting_down.set()
                return runtime

        lifespan = asyncio.create_task(run_lifespan())
        try:
            await asyncio.wait_for(shutting_down.wait(), timeout=5)
            assert not lifespan.done()
            resources.store.close.assert_not_called()
        finally:
            release.set()
        runtime = await asyncio.wait_for(lifespan, timeout=5)
        with pytest.raises(RuntimeError, match="法律向量检索已停止"):
            await runtime.invoke("legal_qa", _request("stopped")["input"], AgentContext(requestId="stopped"))

    asyncio.run(exercise())
    resources.store.close.assert_called_once_with()
    resources.store.search.assert_called_once()


def test_importing_app_does_not_load_rag_dependencies():
    # 用新进程排除其他用例已加载可选依赖的影响。
    code = (
        "import sys; import app.main; "
        "assert not any(name in sys.modules for name in "
        "('sentence_transformers', 'torch', 'pymilvus', 'milvus_lite'))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
