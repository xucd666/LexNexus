import asyncio
import threading
from unittest.mock import Mock, call, patch

import pytest
from app.schemas.law_chunk import LawChunk
from app.tools.law import LawSearch
from app.tools.vector_law_search import VectorLawSearch

VECTOR = [1.0] + [0.0] * 511
QUERY = "  合同约定不能履行，如何处理？ "


@pytest.fixture
def backend():
    embedding = Mock(spec_set=["encode_query"])
    embedding.encode_query.return_value = VECTOR
    store = Mock(spec_set=["search", "close"])
    store.search.return_value = []
    return embedding, store


def _chunk(chunk_id: str, article_no: str, text: str) -> LawChunk:
    return LawChunk(
        document_id="civil-code-2020-fixture",
        kb_chunk_id=chunk_id,
        law_name="中华人民共和国民法典",
        article_no=article_no,
        text=text,
        source_url="https://example.invalid/civil-code",
    )


def test_search_maps_all_citation_fields_and_keeps_store_order(backend):
    embedding, store = backend
    first = _chunk("chunk-580", "第五百八十条", "  第一款测试正文。\n　　第二款测试正文。 \n")
    second = _chunk("chunk-581", "第五百八十一条", "另一条测试正文。")
    # 分数顺序刻意不同于常规降序，也包含负分，以验证适配器直接沿用存储结果。
    store.search.return_value = [(first, -0.25), (second, 0.875)]
    searcher: LawSearch = VectorLawSearch(embedding, store, top_k=2)

    citations = asyncio.run(searcher.search(QUERY, request_id="req-1", trace_id="trace-1"))

    embedding.encode_query.assert_called_once_with(QUERY)
    store.search.assert_called_once_with(VECTOR, top_k=2)
    assert [citation.model_dump() for citation in citations] == [
        {
            "document_id": "civil-code-2020-fixture",
            "kb_chunk_id": "chunk-580",
            "law_name": "中华人民共和国民法典",
            "article_no": "第五百八十条",
            "quoted_text": "  第一款测试正文。\n　　第二款测试正文。 \n",
            "relevance_score": -0.25,
        },
        {
            "document_id": "civil-code-2020-fixture",
            "kb_chunk_id": "chunk-581",
            "law_name": "中华人民共和国民法典",
            "article_no": "第五百八十一条",
            "quoted_text": "另一条测试正文。",
            "relevance_score": 0.875,
        },
    ]


def test_search_returns_empty_list_when_store_has_no_hits(backend):
    embedding, store = backend
    searcher = VectorLawSearch(embedding, store)
    assert asyncio.run(searcher.search(QUERY, request_id="req-1", trace_id=None)) == []
    store.search.assert_called_once_with(VECTOR, top_k=3)


@pytest.mark.parametrize(
    ("query", "error"), [("", ValueError), (" \t\n　", ValueError), (None, TypeError), (42, TypeError)]
)
def test_search_rejects_invalid_input_before_calling_the_backends(backend, query, error):
    embedding, store = backend
    searcher = VectorLawSearch(embedding, store)

    with pytest.raises(error, match="query"):
        asyncio.run(searcher.search(query, request_id="req-1", trace_id=None))

    embedding.encode_query.assert_not_called()
    store.search.assert_not_called()


@pytest.mark.parametrize("top_k", [0, 16_385, True, 1.5])
def test_rejects_invalid_top_k_at_construction(backend, top_k):
    with pytest.raises(ValueError, match="top_k"):
        VectorLawSearch(*backend, top_k=top_k)


@pytest.mark.parametrize("phase", ["encoding", "milvus"])
def test_backend_error_propagates_and_does_not_keep_the_search_slot(backend, phase):
    embedding, store = backend
    operation = embedding.encode_query if phase == "encoding" else store.search
    cause = RuntimeError(f"{phase} failed")
    operation.side_effect = cause
    searcher = VectorLawSearch(embedding, store)

    async def exercise():
        with pytest.raises(RuntimeError) as failure:
            await searcher.search(QUERY, request_id="req-1", trace_id=None)
        assert failure.value is cause
        if phase == "encoding":
            store.search.assert_not_called()

        operation.side_effect = None
        assert (
            await asyncio.wait_for(searcher.search(QUERY, request_id="req-2", trace_id=None), timeout=5) == []
        )

    asyncio.run(exercise())
    store.close.assert_not_called()


@pytest.mark.parametrize("phase", ["encoding", "milvus"])
def test_event_loop_can_progress_while_sync_backend_is_waiting(backend, phase):
    embedding, store = backend
    operation = embedding.encode_query if phase == "encoding" else store.search
    searcher = VectorLawSearch(embedding, store)

    async def exercise():
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        started = asyncio.Event()
        release = threading.Event()

        def blocked_operation(*args, **kwargs):
            loop.call_soon_threadsafe(started.set)
            assert threading.get_ident() != loop_thread
            assert release.wait(5)
            return operation.return_value

        operation.side_effect = blocked_operation
        task = asyncio.create_task(searcher.search(QUERY, request_id="req-1", trace_id=None))
        try:
            # 只有循环仍能处理事件，才能在同步调用结束前收到此通知。
            await asyncio.wait_for(started.wait(), timeout=5)
            assert not task.done()
        finally:
            release.set()
        assert await task == []

    asyncio.run(exercise())


def test_cancelled_request_holds_slot_until_its_worker_finishes(backend):
    embedding, store = backend
    searcher = VectorLawSearch(embedding, store)

    async def exercise():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        queued = asyncio.Event()
        release = threading.Event()

        def encode(query):
            if query == "第一条":
                loop.call_soon_threadsafe(started.set)
                assert release.wait(5)
            return VECTOR

        async def search_next():
            queued.set()
            return await searcher.search("第二条", request_id="req-2", trace_id=None)

        embedding.encode_query.side_effect = encode
        with patch.object(loop, "run_in_executor", wraps=loop.run_in_executor) as submit:
            first = asyncio.create_task(searcher.search("第一条", request_id="req-1", trace_id=None))
            try:
                await asyncio.wait_for(started.wait(), timeout=5)
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first

                second = asyncio.create_task(search_next())
                await asyncio.wait_for(queued.wait(), timeout=5)
                # 观察线程池提交次数可以检测重叠，无需靠 sleep 猜测线程调度。
                assert submit.call_count == 1
                assert not second.done()
            finally:
                release.set()
            assert await asyncio.wait_for(second, timeout=5) == []
            assert submit.call_count == 2

    asyncio.run(exercise())
    assert embedding.encode_query.call_args_list == [call("第一条"), call("第二条")]
    assert store.search.call_count == 2
    store.close.assert_not_called()


def test_multiple_searches_reuse_injected_resources_and_keep_owner_in_control(backend):
    embedding, store = backend
    searcher = VectorLawSearch(embedding, store)
    embedding.encode_query.assert_not_called()
    store.search.assert_not_called()

    async def exercise():
        assert await searcher.search("第一条", request_id="req-1", trace_id="trace-1") == []
        assert await searcher.search("第二条", request_id="req-2", trace_id=None) == []

    asyncio.run(exercise())
    assert embedding.encode_query.call_args_list == [call("第一条"), call("第二条")]
    assert store.search.call_args_list == [call(VECTOR, top_k=3), call(VECTOR, top_k=3)]
    store.close.assert_not_called()
