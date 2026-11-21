"""评测当前无过滤的法律向量检索，将逐题结果和统计量输出为 JSON。"""

import argparse
import asyncio
import json
import math
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Annotated, Literal

from app.tools.bge_embedding import BgeEmbedding
from app.tools.law import LawSearch
from app.tools.law_splitter import split_law_articles
from app.tools.milvus_store import MilvusStore
from app.tools.vector_law_search import VectorLawSearch
from pydantic import BaseModel, ConfigDict, StringConstraints, TypeAdapter
from scripts.ingest_laws import ingest_laws

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
DEFAULT_CASES = FIXTURES / "law_search_relevance.json"
CORPUS = FIXTURES / "civil_code_2020_articles_577_584"
COLLECTION = "legal_relevance_eval"
TOP_K = 3
Category = Literal["in_scope", "unrelated", "out_of_scope"]
NonEmptyText = Annotated[str, StringConstraints(min_length=1)]


class EvaluationCase(BaseModel):
    """一条事先标注的问题及其语料覆盖范围。"""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    case_id: NonEmptyText
    question: NonEmptyText
    category: Category
    expected_article_nos: list[NonEmptyText]
    annotation_reason: NonEmptyText


def load_cases(data: bytes, corpus_article_nos: set[str]) -> list[EvaluationCase]:
    """校验标注引用的法条存在于本轮语料，避免指标误用语料外法条。"""
    cases = TypeAdapter(list[EvaluationCase]).validate_json(data)
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("评测题目须非空，且 case_id 不能重复")
    for case in cases:
        expected = set(case.expected_article_nos)
        if case.category == "in_scope":
            if not expected or not expected <= corpus_article_nos:
                raise ValueError(f"{case.case_id} 须标注当前语料中的预期条号")
        elif expected:
            raise ValueError(f"{case.case_id} 的无关或库外问题不能标注预期条号")
    return cases


async def retrieve_cases(searcher: LawSearch, cases: list[EvaluationCase]) -> list[dict]:
    """逐题保留原始命中顺序，计时从异步检索调用开始。"""
    rows = []
    for case in cases:
        start = perf_counter()
        citations = await searcher.search(case.question, request_id=case.case_id, trace_id=None)
        elapsed_ms = (perf_counter() - start) * 1000
        if len(citations) > TOP_K:
            raise ValueError(f"{case.case_id} 返回的引用数量超过 Top-{TOP_K}")
        hits = []
        for citation in citations:
            article, score = citation.article_no, citation.relevance_score
            if not article or score is None or not math.isfinite(score):
                raise ValueError(f"{case.case_id} 返回缺少条号或有限分数的引用")
            hits.append({"article_no": article, "score": score})

        # 多条标注时取首个命中的排名。无命中记为 null，仍保留实际 Top-3。
        relevant_rank = next(
            (
                rank
                for rank, hit in enumerate(hits, start=1)
                if hit["article_no"] in case.expected_article_nos
            ),
            None,
        )
        rows.append(
            case.model_dump()
            | {"top_3": hits, "first_relevant_rank": relevant_rank, "elapsed_ms": elapsed_ms}
        )
    return rows


def _distribution(values: list[float]) -> dict:
    return {
        "count": len(values),
        "min": min(values, default=None),
        "median": median(values) if values else None,
        "max": max(values, default=None),
    }


def summarize(rows: list[dict]) -> dict:
    """Hit@K 只以库内问题为分母。空返回计入题数，不计为零分。"""
    ranks = [row["first_relevant_rank"] for row in rows if row["category"] == "in_scope"]
    categories = {}
    for category in ("in_scope", "unrelated", "out_of_scope"):
        group = [row for row in rows if row["category"] == category]
        scores = [row["top_3"][0]["score"] for row in group if row["top_3"]]
        categories[category] = {
            "questions": len(group),
            "returned": len(scores),
            "top_1_score": _distribution(scores),
        }

    hits = {
        1: sum(rank == 1 for rank in ranks),
        TOP_K: sum(rank is not None and rank <= TOP_K for rank in ranks),
    }
    return {
        "retrieval_hits": {
            f"hit_at_{k}": {"hits": count, "total": len(ranks), "rate": count / len(ranks) if ranks else None}
            for k, count in hits.items()
        },
        "by_category": categories,
        "query_latency_ms": _distribution([row["elapsed_ms"] for row in rows]),
    }


def _evaluate(model_path: Path, dataset: Path, workspace: Path) -> dict:
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    source = CORPUS.with_suffix(".txt")
    metadata_path = CORPUS.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw_source = source.read_bytes()
    article_nos = {
        chunk.article_no
        for chunk in split_law_articles(
            raw_source.decode("utf-8"),
            document_id=metadata["document_id"],
            law_name=metadata["law_name"],
            source_url=metadata["source_url"],
        )
    }
    raw_cases = dataset.read_bytes()
    cases = load_cases(raw_cases, article_nos)

    # 入库和查询共用一次模型加载。新库经过现有入库流程的主键与原文核查。
    embedding = BgeEmbedding(model_path)
    db_path = workspace / "civil-code.db"
    manifest = ingest_laws(
        source=source,
        metadata=metadata_path,
        model_path=model_path,
        db_path=db_path,
        collection=COLLECTION,
        manifest=workspace / "manifest.json",
        embedding=embedding,
    )
    if manifest["source_sha256"] != sha256(raw_source).hexdigest():
        raise RuntimeError("评测准备期间语料发生变化")
    with closing(MilvusStore(db_path, COLLECTION, create=False)) as store:
        searcher = VectorLawSearch(embedding, store, top_k=TOP_K)
        rows = asyncio.run(retrieve_cases(searcher, cases))
    return {
        "dataset_sha256": sha256(raw_cases).hexdigest(),
        "model_snapshot": model_path.name,
        "corpus": manifest,
        "top_k": TOP_K,
        "summary": summarize(rows),
        "cases": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """创建临时语料库并运行基线，退出码只反映评测是否完成。"""
    parser = argparse.ArgumentParser(description="评测民法典第 577—584 条样本的法律检索相关性")
    parser.add_argument("--model-path", type=Path, required=True, help="本地 BGE 模型快照目录")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CASES, help="带人工标注的 JSON 评测题目")
    args = parser.parse_args(argv)
    snapshot = args.model_path.expanduser().resolve()
    if not snapshot.is_dir():
        parser.error(f"BGE 本地快照目录不存在：{snapshot}")

    # Lite 按进程持有数据库目录锁。先等唯一的工作进程退出，再删除它创建的临时库。
    with (
        TemporaryDirectory(prefix="lexnexus-law-eval-") as directory,
        ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as worker,
    ):
        report = worker.submit(_evaluate, snapshot, args.dataset, Path(directory)).result()
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
