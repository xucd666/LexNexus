"""用本地 BGE 和全新 Milvus Lite 样本库验证真实语义检索，默认跳过。"""

import asyncio
import json
import math
import os
import subprocess
import sys
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import pytest
from app.schemas.contracts import Citation
from app.schemas.law_chunk import LawChunk
from app.tools.bge_embedding import BgeEmbedding
from app.tools.law_splitter import split_law_articles
from app.tools.milvus_store import MilvusStore
from app.tools.vector_law_search import VectorLawSearch

pytestmark = pytest.mark.skipif(
    os.environ.get("LEXNEXUS_RUN_VECTOR_LAW_SEARCH_INTEGRATION") != "1",
    reason="设置 LEXNEXUS_RUN_VECTOR_LAW_SEARCH_INTEGRATION=1 以启用真实语义检索测试",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "civil_code_2020_articles_577_584"
COLLECTION = "legal_vector_search_integration"
TOP_K = 3
QUESTIONS = (
    ("承租人一直不支付租金怎么办？", "第五百七十九条"),
    ("合同违约造成的损失包括可获得的利益吗？", "第五百八十四条"),
    ("交付质量不符合合同约定，可以修理或更换吗？", "第五百八十二条"),
    ("室内绿植换盆后需要每天浇水吗？", None),
)


def _ingest_fixture(tmp_path: Path, snapshot: Path) -> Path:
    """运行正式入库命令，子进程退出后再把库交给检索进程。"""
    db_path = tmp_path / "civil-code.db"
    manifest_path = tmp_path / "manifest.json"
    assert not db_path.exists()
    # Lite 嵌入服务在进程退出时释放目录锁。subprocess.run 会等待退出，避免并行打开同一库。
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "scripts.ingest_laws",
            "--source",
            str(FIXTURE.with_suffix(".txt")),
            "--metadata",
            str(FIXTURE.with_suffix(".json")),
            "--model-path",
            str(snapshot),
            "--db-path",
            str(db_path),
            "--collection",
            COLLECTION,
            "--manifest",
            str(manifest_path),
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, f"入库子进程退出码 {result.returncode}\n{result.stdout}\n{result.stderr}"
    print(result.stdout, end="")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["chunk_count"] == 8
    assert manifest["embedding_dimension"] == 512
    assert manifest["source_sha256"] == sha256(FIXTURE.with_suffix(".txt").read_bytes()).hexdigest()
    return db_path


def _fixture_by_id() -> dict[str, LawChunk]:
    # 与入库一样读取原始字节并解码，按项目的切分规则保留全部正文和换行。
    metadata = json.loads(FIXTURE.with_suffix(".json").read_text(encoding="utf-8"))
    chunks = split_law_articles(
        FIXTURE.with_suffix(".txt").read_bytes().decode("utf-8"),
        document_id=metadata["document_id"],
        law_name=metadata["law_name"],
        source_url=metadata["source_url"],
    )
    assert len(chunks) == 8
    return {chunk.kb_chunk_id: chunk for chunk in chunks}


async def _run_questions(searcher: VectorLawSearch, by_id: dict[str, LawChunk]) -> None:
    rankings = []
    misses = []
    for index, (query, expected) in enumerate((*QUESTIONS, QUESTIONS[0]), start=1):
        start = perf_counter()
        hits = await searcher.search(query, request_id=f"vector-test-{index}", trace_id=None)
        elapsed_ms = (perf_counter() - start) * 1000

        assert isinstance(hits, list)
        assert len(hits) <= TOP_K
        if expected is not None:
            assert len(hits) == TOP_K
        articles = []
        ids = []
        scores = []
        for hit in hits:
            assert isinstance(hit, Citation)
            # 用返回的主键定位同一原始 LawChunk，逐项验证引用的身份和原文。
            assert hit.kb_chunk_id in by_id
            chunk = by_id[hit.kb_chunk_id]
            assert hit.document_id == chunk.document_id
            assert hit.law_name == chunk.law_name
            assert hit.article_no == chunk.article_no
            assert hit.quoted_text == chunk.text
            assert hit.relevance_score is not None and math.isfinite(hit.relevance_score)
            articles.append(hit.article_no)
            ids.append(hit.kb_chunk_id)
            scores.append(hit.relevance_score)

        assert len(ids) == len(set(ids))
        rankings.append((ids, scores))
        ranking = "、".join(
            f"{article} ({score:.8f})" for article, score in zip(articles, scores, strict=True)
        )
        label = "重复查询" if index > len(QUESTIONS) else f"问题 {index}"
        print(f"{label} {query}\n  Top-{TOP_K} {ranking or '[]'}\n  耗时 {elapsed_ms:.2f} ms")
        if expected is not None:
            if expected in articles:
                print(f"  预期 {expected} 实际排名 {articles.index(expected) + 1}")
            else:
                print(f"  预期 {expected} 未进入 Top-{TOP_K}")
                misses.append(label)
        else:
            # 无关问题仅观察结果。向量 Top-K 当前没有相关性阈值。
            print("  无关问题，仅观察实际返回")

    assert rankings[-1][0] == rankings[0][0]
    assert rankings[-1][1] == pytest.approx(rankings[0][1], abs=1e-6, rel=0)
    assert not misses, f"预期法条未进入 Top-{TOP_K}：{', '.join(misses)}"


def test_vector_law_search_with_offline_bge_and_milvus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """用同一个模型、存储和适配器执行完整检索链路。"""
    model_path = os.environ.get("LEXNEXUS_BGE_MODEL_PATH")
    if not model_path:
        pytest.fail("启用真实检索测试时须设置 LEXNEXUS_BGE_MODEL_PATH，指向本地 BGE 快照目录")
    snapshot = Path(model_path).expanduser().resolve()
    assert snapshot.is_dir(), f"BGE 本地快照目录不存在：{snapshot}"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    by_id = _fixture_by_id()
    db_path = _ingest_fixture(tmp_path, snapshot)

    # 记录实际执行查询的对象，并调用原方法，确认重复请求没有另起模型或存储对象。
    encoded_by = []
    searched_by = []
    encode_query = BgeEmbedding.encode_query
    search_vectors = MilvusStore.search

    def observe_encoding(instance, query):
        encoded_by.append(instance)
        return encode_query(instance, query)

    def observe_search(instance, embedding, **options):
        searched_by.append(instance)
        return search_vectors(instance, embedding, **options)

    monkeypatch.setattr(BgeEmbedding, "encode_query", observe_encoding)
    monkeypatch.setattr(MilvusStore, "search", observe_search)
    embedding = BgeEmbedding(snapshot)
    with closing(MilvusStore(db_path, COLLECTION)) as store:
        searcher = VectorLawSearch(embedding, store, top_k=TOP_K)
        # asyncio.run 会先处理完默认线程池，再由 closing 调用 store.close()。
        asyncio.run(_run_questions(searcher, by_id))
        query_count = len(QUESTIONS) + 1
        assert len(encoded_by) == len(searched_by) == query_count
        assert all(instance is embedding for instance in encoded_by)
        assert all(instance is store for instance in searched_by)
    print(f"已验证 {query_count} 次查询复用同一模型和存储，重复查询的顺序与分数一致。")
