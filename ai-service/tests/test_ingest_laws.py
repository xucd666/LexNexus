import json
import sys
from hashlib import sha256
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from app.schemas.law_chunk import LawChunk
from scripts import ingest_laws

FIXTURE = Path(__file__).with_name("fixtures") / "civil_code_2020_articles_577_584"


@pytest.fixture
def options(tmp_path):
    model_path = tmp_path / "snapshot"
    model_path.mkdir()
    return {
        "source": FIXTURE.with_suffix(".txt"),
        "metadata": FIXTURE.with_suffix(".json"),
        "model_path": model_path,
        "db_path": tmp_path / "legal.db",
        "collection": "legal_fixture",
        "manifest": tmp_path / "manifest.json",
    }


@pytest.fixture
def backend(monkeypatch):
    vectors = [[float(i == j) for j in range(512)] for i in range(8)]
    encoder = Mock(spec_set=["encode_documents"])
    encoder.encode_documents.return_value = vectors
    load_model = Mock(return_value=encoder)
    store = Mock(spec_set=["upsert", "close"])
    store.upsert.side_effect = lambda chunks, embeddings: len(chunks)
    open_store = Mock(return_value=store)
    reader = Mock(spec_set=["get", "close"])

    def written_rows(**_):
        chunks, _ = store.upsert.call_args.args
        # 故意倒序返回，验证读回核对按主键建立对应关系。
        return [chunk.model_dump(mode="json") for chunk in reversed(chunks)]

    reader.get.side_effect = written_rows
    sdk = ModuleType("pymilvus")
    sdk.MilvusClient = Mock(return_value=reader)
    monkeypatch.setattr(ingest_laws, "BgeEmbedding", load_model)
    monkeypatch.setattr(ingest_laws, "MilvusStore", open_store)
    monkeypatch.setitem(sys.modules, "pymilvus", sdk)
    return SimpleNamespace(
        vectors=vectors,
        encoder=encoder,
        load_model=load_model,
        store=store,
        open_store=open_store,
        reader=reader,
        open_reader=sdk.MilvusClient,
    )


def test_ingest_splits_eight_articles_and_keeps_vectors_aligned(options, backend, monkeypatch):
    split = Mock(wraps=ingest_laws.split_law_articles)
    monkeypatch.setattr(ingest_laws, "split_law_articles", split)
    raw = options["source"].read_bytes()
    source_text = raw.decode("utf-8")
    metadata = json.loads(options["metadata"].read_text(encoding="utf-8"))

    result = ingest_laws.ingest_laws(**options)

    split.assert_called_once_with(
        source_text,
        document_id=metadata["document_id"],
        law_name=metadata["law_name"],
        source_url=metadata["source_url"],
    )
    backend.load_model.assert_called_once_with(options["model_path"])
    backend.open_store.assert_called_once_with(options["db_path"], options["collection"], create=True)
    backend.store.upsert.assert_called_once()
    chunks, vectors = backend.store.upsert.call_args.args
    assert len(chunks) == len(vectors) == 8
    assert vectors is backend.vectors
    assert [chunk.article_no for chunk in chunks] == [
        "第五百七十七条",
        "第五百七十八条",
        "第五百七十九条",
        "第五百八十条",
        "第五百八十一条",
        "第五百八十二条",
        "第五百八十三条",
        "第五百八十四条",
    ]
    backend.encoder.encode_documents.assert_called_once_with([chunk.text for chunk in chunks])
    for chunk in chunks:
        assert isinstance(chunk, LawChunk)
        assert chunk.text in source_text
        assert chunk.document_id == metadata["document_id"]
        assert chunk.law_name == metadata["law_name"]
        assert str(chunk.source_url) == metadata["source_url"]
    assert "(三)债权人在合理期限内未请求履行。" in chunks[3].text
    backend.open_reader.assert_called_once_with(uri=str(options["db_path"]))
    backend.reader.get.assert_called_once_with(
        collection_name=options["collection"],
        ids=[chunk.kb_chunk_id for chunk in chunks],
        output_fields=list(LawChunk.model_fields),
    )
    backend.store.close.assert_called_once_with()
    backend.reader.close.assert_called_once_with()
    assert options["db_path"].is_dir()
    assert result == {
        "model_id": "BAAI/bge-small-zh-v1.5",
        "collection": options["collection"],
        "embedding_dimension": 512,
        "document_id": metadata["document_id"],
        "law_name": metadata["law_name"],
        "version_basis": metadata["version_basis"],
        "source_url": metadata["source_url"],
        "source_sha256": sha256(raw).hexdigest(),
        "chunk_count": 8,
    }
    assert json.loads(options["manifest"].read_text(encoding="utf-8")) == result


def test_ingest_keeps_original_crlf_and_hashes_raw_bytes(tmp_path, options, backend):
    source = tmp_path / "source-crlf.txt"
    text = options["source"].read_text(encoding="utf-8").replace("\n", "\r\n")
    source.write_bytes(text.encode("utf-8"))
    options["source"] = source

    manifest = ingest_laws.ingest_laws(**options)

    chunks, _ = backend.store.upsert.call_args.args
    assert "\r\n" in chunks[3].text
    assert chunks[3].text in text
    assert manifest["source_sha256"] == sha256(source.read_bytes()).hexdigest()


@pytest.mark.parametrize("field", ["source", "metadata"])
def test_missing_input_fails_before_loading_model(tmp_path, options, backend, field):
    options[field] = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        ingest_laws.ingest_laws(**options)
    backend.load_model.assert_not_called()
    assert not options["db_path"].exists()
    assert not options["manifest"].exists()


def test_missing_model_snapshot_fails_before_creating_db(tmp_path, options, backend, monkeypatch):
    from app.tools.bge_embedding import BgeEmbedding

    options["model_path"] = tmp_path / "missing-snapshot"
    monkeypatch.setattr(ingest_laws, "BgeEmbedding", BgeEmbedding)
    with pytest.raises(FileNotFoundError, match="snapshot does not exist"):
        ingest_laws.ingest_laws(**options)
    backend.open_store.assert_not_called()
    assert not options["db_path"].exists()
    assert not options["manifest"].exists()


def test_metadata_requires_version_basis(tmp_path, options, backend):
    metadata = json.loads(options["metadata"].read_text(encoding="utf-8"))
    del metadata["version_basis"]
    options["metadata"] = tmp_path / "incomplete.json"
    options["metadata"].write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="version_basis"):
        ingest_laws.ingest_laws(**options)
    backend.load_model.assert_not_called()
    assert not options["db_path"].exists()


@pytest.mark.parametrize("is_dir", [True, False])
def test_existing_database_target_is_preserved(options, backend, is_dir):
    target = options["db_path"]
    if is_dir:
        target.mkdir()
        target = target / "original.txt"
    target.write_text("original", encoding="utf-8")

    with pytest.raises(FileExistsError, match="目标已存在"):
        ingest_laws.ingest_laws(**options)
    backend.load_model.assert_not_called()
    backend.open_store.assert_not_called()
    assert target.read_text(encoding="utf-8") == "original"


def test_existing_manifest_is_preserved(options, backend):
    options["manifest"].write_text("original manifest", encoding="utf-8")
    with pytest.raises(FileExistsError, match="目标已存在"):
        ingest_laws.ingest_laws(**options)
    backend.load_model.assert_not_called()
    assert not options["db_path"].exists()
    assert options["manifest"].read_text(encoding="utf-8") == "original manifest"


def test_database_created_during_encoding_is_left_alone(options, backend):
    claimed = options["db_path"] / "original.txt"

    def encode_after_target_is_claimed(_):
        claimed.parent.mkdir()
        claimed.write_text("claimed by another process", encoding="utf-8")
        return backend.vectors

    backend.encoder.encode_documents.side_effect = encode_after_target_is_claimed
    with pytest.raises(FileExistsError):
        ingest_laws.ingest_laws(**options)
    backend.open_store.assert_not_called()
    assert claimed.read_text(encoding="utf-8") == "claimed by another process"
    assert not options["manifest"].exists()


def test_model_failure_does_not_create_database_or_manifest(options, backend):
    cause = RuntimeError("model failed")
    backend.encoder.encode_documents.side_effect = cause
    with pytest.raises(RuntimeError) as failure:
        ingest_laws.ingest_laws(**options)
    assert failure.value is cause
    backend.open_store.assert_not_called()
    assert not options["db_path"].exists()
    assert not options["manifest"].exists()


@pytest.mark.parametrize("count", [0, 7, 9])
def test_mismatched_vector_count_fails_before_creating_database(options, backend, count):
    backend.encoder.encode_documents.return_value = (backend.vectors + backend.vectors[:1])[:count]
    with pytest.raises(ValueError, match=rf"片段数量 8 与向量数量 {count} 不一致"):
        ingest_laws.ingest_laws(**options)
    backend.open_store.assert_not_called()
    assert not options["db_path"].exists()
    assert not options["manifest"].exists()


def test_upsert_failure_closes_store_and_leaves_no_manifest(options, backend):
    cause = RuntimeError("write failed")
    backend.store.upsert.side_effect = cause
    with pytest.raises(RuntimeError) as failure:
        ingest_laws.ingest_laws(**options)
    assert failure.value is cause
    backend.store.close.assert_called_once_with()
    backend.open_reader.assert_not_called()
    assert not options["manifest"].exists()


def test_short_write_prevents_readback_and_manifest(options, backend):
    backend.store.upsert.side_effect = None
    backend.store.upsert.return_value = 7
    with pytest.raises(RuntimeError, match="预期写入 8 条，实际写入 7 条"):
        ingest_laws.ingest_laws(**options)
    backend.store.close.assert_called_once_with()
    backend.open_reader.assert_not_called()
    assert not options["manifest"].exists()


def test_changed_readback_fails_before_manifest_and_closes_clients(options, backend):
    original_rows = backend.reader.get.side_effect

    def corrupted_rows(**kwargs):
        rows = original_rows(**kwargs)
        rows[0]["text"] = rows[0]["text"].strip() + "篡改"
        return rows

    backend.reader.get.side_effect = corrupted_rows
    with pytest.raises(RuntimeError, match="入库读回校验失败"):
        ingest_laws.ingest_laws(**options)
    backend.reader.close.assert_called_once_with()
    backend.store.close.assert_called_once_with()
    assert not options["manifest"].exists()


def test_cli_only_reports_success_for_completed_ingest(options, backend, capsys):
    argv = [item for name, value in options.items() for item in (f"--{name.replace('_', '-')}", str(value))]

    assert ingest_laws.main(argv) == 0
    success = capsys.readouterr()
    assert "写入 8 条" in success.out
    assert "Collection legal_fixture" in success.out
    assert "向量维度 512" in success.out
    assert "文档版本 cn-civil-code-2020-05-28" in success.out
    assert success.err == ""

    assert ingest_laws.main(argv) == 1
    failure = capsys.readouterr()
    assert failure.out == ""
    assert "离线入库失败" in failure.err
    assert "目标已存在" in failure.err
    backend.store.upsert.assert_called_once()
