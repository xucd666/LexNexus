"""使用独立临时数据库验证正式 MilvusStore，默认跳过。"""

import json
import math
import os
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest
from app.schemas.law_chunk import LawChunk
from app.tools.law_splitter import split_law_articles
from app.tools.milvus_store import MilvusStore

pytestmark = pytest.mark.skipif(
    os.environ.get("LEXNEXUS_RUN_MILVUS_INTEGRATION") != "1",
    reason="设置 LEXNEXUS_RUN_MILVUS_INTEGRATION=1 以启用真实 Milvus Lite 测试",
)

COLLECTION = "legal_chunks_integration"
FIXTURE = Path(__file__).resolve().with_name("fixtures") / "civil_code_2020_articles_577_584"
FIELD_BYTES = {
    "kb_chunk_id": 512,
    "document_id": 512,
    "law_name": 1024,
    "article_no": 256,
    "text": 65_535,
    "source_url": 4096,
}


def _fixture_chunks() -> list[LawChunk]:
    metadata = json.loads(FIXTURE.with_suffix(".json").read_text(encoding="utf-8"))
    chunks = split_law_articles(
        FIXTURE.with_suffix(".txt").read_text(encoding="utf-8"),
        document_id=metadata["document_id"],
        law_name=metadata["law_name"],
        source_url=metadata["source_url"],
    )
    assert len(chunks) == 8
    # 复用样本原文，另设仅供测试的文档 ID，验证过滤会排除原本靠前的命中。
    control = LawChunk.model_validate(
        {
            **chunks[-1].model_dump(),
            "document_id": '过滤对照"条文\\版本\n第二行',
            "kb_chunk_id": "integration-filter-control",
        }
    )
    return [*chunks, control]


def _vector(cosine: float) -> list[float]:
    """构造单位向量，其与首个基向量的余弦分数等于 cosine。"""
    return [cosine, math.sqrt(1 - cosine**2), *([0.0] * 510)]


def _check_round_trip(store: MilvusStore, client, chunks: list[LawChunk]) -> list[float]:
    field_names = list(LawChunk.model_fields)
    rows = client.get(COLLECTION, ids=[chunk.kb_chunk_id for chunk in chunks], output_fields=field_names)
    by_key = {row["kb_chunk_id"]: {name: row[name] for name in field_names} for row in rows}
    assert len(rows) == len(chunks)
    assert by_key == {chunk.kb_chunk_id: chunk.model_dump(mode="json") for chunk in chunks}
    assert client.get_collection_stats(COLLECTION)["row_count"] == len(chunks)

    # 查询以可计算的固定向量为基准，检查 SDK 顺序和数值，不依赖模型或网络。
    hits = store.search(_vector(1.0), top_k=3)
    assert [chunk for chunk, _ in hits] == [chunks[0], chunks[-1], chunks[1]]
    scores = [score for _, score in hits]
    assert all(type(score) is float and math.isfinite(score) for score in scores)
    assert scores == pytest.approx([1.0, 0.9, 0.8], abs=1e-6)

    filtered = store.search(_vector(1.0), top_k=3, document_id=chunks[0].document_id)
    assert [chunk for chunk, _ in filtered] == chunks[:3]
    assert [score for _, score in filtered] == pytest.approx([1.0, 0.8, 0.6], abs=1e-6)
    control = store.search(_vector(1.0), document_id=chunks[-1].document_id)
    assert [chunk for chunk, _ in control] == [chunks[-1]]
    assert control[0][1] == pytest.approx(0.9, abs=1e-6)
    assert store.search(_vector(1.0), document_id="integration-absent-doc") == []
    return scores


def _write(db_path: Path) -> None:
    from pymilvus import DataType, MilvusClient

    assert not db_path.exists()
    chunks = _fixture_chunks()
    scores = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, 0.9)
    with (
        closing(MilvusStore(db_path, COLLECTION, create=True)) as store,
        closing(MilvusClient(uri=str(db_path))) as client,
    ):
        assert db_path.is_dir()
        schema = client.describe_collection(COLLECTION)
        fields = {field["name"]: field for field in schema["fields"]}
        assert schema["auto_id"] is False
        assert schema["enable_dynamic_field"] is False
        assert set(fields) == {*FIELD_BYTES, "embedding"}
        for name, max_bytes in FIELD_BYTES.items():
            assert fields[name]["type"] == DataType.VARCHAR
            assert fields[name]["params"]["max_length"] == max_bytes
            assert fields[name].get("is_primary", False) == (name == "kb_chunk_id")
        assert fields["embedding"]["type"] == DataType.FLOAT_VECTOR
        assert fields["embedding"]["params"]["dim"] == 512
        index = client.describe_index(COLLECTION, index_name="embedding")
        assert (index["field_name"], index["index_type"], index["metric_type"]) == (
            "embedding",
            "FLAT",
            "COSINE",
        )
        assert store.upsert(chunks, map(_vector, scores)) == len(chunks)
        assert store.upsert(chunks[:1], [_vector(1.0)]) == 1
        actual = _check_round_trip(store, client, chunks)
    print(f"write pid={os.getpid()} rows={len(chunks)} dim=512 index=FLAT/COSINE top3={actual}")


def _reopen(db_path: Path) -> None:
    from pymilvus import MilvusClient

    chunks = _fixture_chunks()
    with (
        closing(MilvusStore(db_path, COLLECTION)) as store,
        closing(MilvusClient(uri=str(db_path))) as client,
    ):
        actual = _check_round_trip(store, client, chunks)
    print(f"reopen pid={os.getpid()} persisted_rows={len(chunks)} top3={actual}")


def _check_errors(db_path: Path) -> None:
    from pymilvus import DataType, MilvusClient
    from pymilvus.exceptions import MilvusException

    assert not db_path.exists()
    with pytest.raises(FileNotFoundError, match="数据库不存在"):
        MilvusStore(db_path, COLLECTION)
    assert not db_path.exists()
    with (
        closing(MilvusStore(db_path, COLLECTION, create=True)) as store,
        closing(MilvusClient(uri=str(db_path))) as client,
    ):
        missing = "integration_missing_collection"
        with pytest.raises(LookupError, match="Collection 不存在"):
            MilvusStore(db_path, missing)
        assert not client.has_collection(missing)
        with pytest.raises(ValueError, match="512 维"):
            store.search(_vector(1.0)[:-1])
        with pytest.raises(ValueError, match="512 维"):
            store.upsert(_fixture_chunks()[:1], [_vector(1.0)[:-1]])
        assert client.get_collection_stats(COLLECTION)["row_count"] == 0

        wrong = "integration_wrong_dimension"
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        for name, max_bytes in FIELD_BYTES.items():
            schema.add_field(name, DataType.VARCHAR, max_length=max_bytes, is_primary=name == "kb_chunk_id")
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=256)
        index = client.prepare_index_params()
        index.add_index("embedding", index_type="FLAT", metric_type="COSINE")
        client.create_collection(wrong, schema=schema, index_params=index)
        with pytest.raises(ValueError, match=r"embedding.*dim=512"):
            MilvusStore(db_path, wrong)

        # 让真实后端拒绝一次搜索，验证 Store 保留 SDK 异常。
        client.release_collection(COLLECTION)
        with pytest.raises(MilvusException):
            store.search(_vector(1.0))
    print("errors: missing path/collection, wrong vector/schema dimensions, SDK search failure")


def _run_process(phase: str, db_path: Path) -> None:
    script = Path(__file__).resolve()
    # 每个进程都用正式解释器。run 会等待进程退出，避免 Lite 的目录锁还在时重开。
    result = subprocess.run(
        [sys.executable, "-B", str(script), phase, str(db_path)],
        cwd=script.parents[1],
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, (
        f"{phase} 子进程退出码={result.returncode}\n{result.stdout}\n{result.stderr}"
    )
    print(result.stdout, end="")


def test_milvus_store_persists_after_writer_process_exits(tmp_path: Path) -> None:
    """先写入，再由独立解释器按同一路径重开和检索。"""
    db_path = tmp_path / "legal-store.db"
    _run_process("write", db_path)
    _run_process("reopen", db_path)


def test_milvus_store_rejects_incompatible_collection_and_propagates_sdk_error(tmp_path: Path) -> None:
    """在单独的临时数据库验证错误边界。"""
    _run_process("errors", tmp_path / "legal-errors.db")


if __name__ == "__main__":
    {"write": _write, "reopen": _reopen, "errors": _check_errors}[sys.argv[1]](Path(sys.argv[2]))
