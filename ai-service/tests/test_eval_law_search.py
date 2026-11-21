import asyncio
import json
from collections import Counter
from unittest.mock import AsyncMock, Mock

import pytest
from app.schemas.contracts import Citation
from app.tools.law_splitter import split_law_articles
from scripts import eval_law_search, ingest_laws

GENERAL = "第五百七十七条"
PAYMENT = "第五百七十九条"
REPAIR = "第五百八十二条"


def _case(case_id, category="in_scope", expected=(PAYMENT,)):
    return {
        "case_id": case_id,
        "question": f"用于统计的测试问题 {case_id}",
        "category": category,
        "expected_article_nos": list(expected),
        "annotation_reason": "用于验证评测逻辑",
    }


def _json_bytes(rows):
    return json.dumps(rows, ensure_ascii=False).encode("utf-8")


def test_fixed_dataset_annotations_cover_all_eight_real_articles():
    source = eval_law_search.CORPUS
    metadata = json.loads(source.with_suffix(".json").read_bytes())
    chunks = split_law_articles(
        source.with_suffix(".txt").read_bytes().decode("utf-8"),
        document_id=metadata["document_id"],
        law_name=metadata["law_name"],
        source_url=metadata["source_url"],
    )
    corpus = {chunk.article_no for chunk in chunks}
    cases = eval_law_search.load_cases(eval_law_search.DEFAULT_CASES.read_bytes(), corpus)
    assert len(corpus) == 8
    assert Counter(case.category for case in cases) == {"in_scope": 14, "unrelated": 7, "out_of_scope": 7}
    assert {article for case in cases for article in case.expected_article_nos} == corpus


@pytest.mark.parametrize(
    "rows",
    [
        [_case("no-label", expected=())],
        [_case("unknown-article", expected=("第一条",))],
        [_case("relabeled-negative", category="unrelated")],
        [_case("duplicate"), _case("duplicate")],
    ],
)
def test_rejects_invalid_annotations(rows):
    with pytest.raises(ValueError):
        eval_law_search.load_cases(_json_bytes(rows), {PAYMENT})


def test_ranked_hits_and_score_distribution_preserve_real_returns():
    dataset = [
        _case("first"),
        _case("second", expected=(PAYMENT, REPAIR)),
        _case("miss"),
        _case("unrelated", category="unrelated", expected=()),
        _case("outside", category="out_of_scope", expected=()),
        _case("outside-empty", category="out_of_scope", expected=()),
    ]
    cases = eval_law_search.load_cases(_json_bytes(dataset), {GENERAL, PAYMENT, REPAIR})

    def hit(article, score):
        return Citation(article_no=article, relevance_score=score)

    searcher = Mock(
        search=AsyncMock(
            side_effect=[
                [hit(PAYMENT, 0.8)],
                [hit(GENERAL, 0.9), hit(REPAIR, 0.7)],
                [hit(GENERAL, 0.85)],
                [hit(GENERAL, 0.25)],
                [hit(PAYMENT, 0.82)],
                [],
            ]
        )
    )

    rows = asyncio.run(eval_law_search.retrieve_cases(searcher, cases))
    summary = eval_law_search.summarize(rows)

    assert [row["first_relevant_rank"] for row in rows] == [1, 2, None, None, None, None]
    assert rows[1]["top_3"] == [{"article_no": GENERAL, "score": 0.9}, {"article_no": REPAIR, "score": 0.7}]
    assert summary["retrieval_hits"] == {
        "hit_at_1": {"hits": 1, "total": 3, "rate": 1 / 3},
        "hit_at_3": {"hits": 2, "total": 3, "rate": 2 / 3},
    }
    assert summary["by_category"]["in_scope"]["top_1_score"]["median"] == 0.85
    assert summary["by_category"]["unrelated"]["top_1_score"]["max"] == 0.25
    assert summary["by_category"]["out_of_scope"] == {
        "questions": 2,
        "returned": 1,
        "top_1_score": {"count": 1, "min": 0.82, "median": 0.82, "max": 0.82},
    }
    assert summary["query_latency_ms"]["count"] == 6
    assert all(row["elapsed_ms"] >= 0 for row in rows)


def test_non_finite_score_is_a_retrieval_error():
    cases = eval_law_search.load_cases(_json_bytes([_case("bad-score")]), {PAYMENT})
    searcher = Mock(
        search=AsyncMock(return_value=[Citation(article_no=PAYMENT, relevance_score=float("nan"))])
    )
    with pytest.raises(ValueError, match="有限分数"):
        asyncio.run(eval_law_search.retrieve_cases(searcher, cases))


def test_one_embedding_is_shared_by_ingestion_and_retrieval(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    workspace = tmp_path / "evaluation"
    workspace.mkdir()
    dataset = tmp_path / "cases.json"
    dataset.write_bytes(_json_bytes([_case("miss")]))
    vector = [1.0] + [0.0] * 511
    embedding = Mock()
    embedding.encode_documents.return_value = [vector] * 8
    embedding.encode_query.return_value = vector
    load_model = Mock(return_value=embedding)
    unexpected_load = Mock(side_effect=AssertionError("入库应复用已加载的模型"))
    write_store = Mock()
    write_store.upsert.return_value = 8
    read_store = Mock()
    read_store.search.return_value = []
    monkeypatch.setattr(eval_law_search, "BgeEmbedding", load_model)
    monkeypatch.setattr(eval_law_search, "MilvusStore", Mock(return_value=read_store))
    monkeypatch.setattr(ingest_laws, "BgeEmbedding", unexpected_load)
    monkeypatch.setattr(ingest_laws, "MilvusStore", Mock(return_value=write_store))
    monkeypatch.setattr(ingest_laws, "_verify_written_chunks", Mock())

    report = eval_law_search._evaluate(tmp_path / "model", dataset, workspace)

    load_model.assert_called_once_with(tmp_path / "model")
    unexpected_load.assert_not_called()
    embedding.encode_documents.assert_called_once()
    embedding.encode_query.assert_called_once_with("用于统计的测试问题 miss")
    read_store.search.assert_called_once_with(vector, top_k=3)
    read_store.close.assert_called_once_with()
    write_store.close.assert_called_once_with()
    assert report["corpus"]["chunk_count"] == 8
    assert report["summary"]["retrieval_hits"]["hit_at_3"]["hits"] == 0
