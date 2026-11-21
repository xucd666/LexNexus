import json
import sys
from enum import IntEnum
from types import ModuleType
from unittest.mock import Mock, call

import pytest
from app.schemas.law_chunk import LawChunk
from app.tools.milvus_store import MilvusStore
from pydantic import ValidationError

COLLECTION = "legal_chunks"
VECTOR = [1.0] + [0.0] * 511
FIELD_LIMITS = {
    "kb_chunk_id": 512,
    "document_id": 512,
    "law_name": 1024,
    "article_no": 256,
    "text": 65535,
    "source_url": 4096,
}


class FakeDataType(IntEnum):
    VARCHAR = 21
    FLOAT_VECTOR = 101


@pytest.fixture
def sdk(monkeypatch):
    client = Mock(
        spec_set=[
            "has_collection",
            "describe_collection",
            "describe_index",
            "load_collection",
            "create_schema",
            "prepare_index_params",
            "create_collection",
            "upsert",
            "search",
            "close",
        ]
    )
    client.has_collection.return_value = True
    client.describe_collection.return_value = {
        "auto_id": False,
        "enable_dynamic_field": False,
        "fields": [
            {
                "name": name,
                "type": FakeDataType.VARCHAR.value,
                "params": {"max_length": max_bytes},
                **({"is_primary": True} if name == "kb_chunk_id" else {}),
            }
            for name, max_bytes in FIELD_LIMITS.items()
        ]
        + [{"name": "embedding", "type": FakeDataType.FLOAT_VECTOR.value, "params": {"dim": 512}}],
    }
    client.describe_index.return_value = {
        "index_name": "embedding",
        "field_name": "embedding",
        "index_type": "FLAT",
        "metric_type": "COSINE",
    }
    client.create_schema.return_value = Mock(spec_set=["add_field"])
    client.prepare_index_params.return_value = Mock(spec_set=["add_index"])
    client.search.return_value = [[]]
    pymilvus = ModuleType("pymilvus")
    pymilvus.DataType = FakeDataType
    pymilvus.MilvusClient = Mock(return_value=client)
    monkeypatch.setitem(sys.modules, "pymilvus", pymilvus)
    return pymilvus.MilvusClient, client


@pytest.fixture
def db_path(tmp_path):
    # Lite 3.x 可用 .db 目录。这里只模拟路径存在，所有存储操作仍经过替身。
    path = tmp_path / "legal.db"
    path.mkdir()
    return path


def make_chunk(**changes):
    return LawChunk.model_validate(
        {
            "document_id": "civil-code-2020",
            "kb_chunk_id": "chunk-580",
            "law_name": "中华人民共和国民法典",
            "article_no": "第五百八十条",
            "text": "  第一款正文。\n  第二款正文。\n",
            "source_url": "https://www.example.invalid/civil-code#article-580",
            **changes,
        }
    )


def make_hit(chunk, score, *, key="kb_chunk_id"):
    entity = chunk.model_dump(mode="json", exclude={"kb_chunk_id"})
    return {key: chunk.kb_chunk_id, "distance": score, "entity": entity}


def test_create_collection_configures_schema_and_flat_index(tmp_path, sdk):
    factory, client = sdk
    client.has_collection.return_value = False
    path = tmp_path / "new.db"

    MilvusStore(path, COLLECTION, create=True)

    factory.assert_called_once_with(uri=str(path.resolve()))
    client.has_collection.assert_called_once_with(collection_name=COLLECTION)
    client.create_schema.assert_called_once_with(auto_id=False, enable_dynamic_field=False)
    schema = client.create_schema.return_value
    assert schema.add_field.call_args_list == [
        call(
            field_name=name, datatype=FakeDataType.VARCHAR, max_length=size, is_primary=name == "kb_chunk_id"
        )
        for name, size in FIELD_LIMITS.items()
    ] + [call(field_name="embedding", datatype=FakeDataType.FLOAT_VECTOR, dim=512)]
    client.prepare_index_params.assert_called_once_with()
    index = client.prepare_index_params.return_value
    index.add_index.assert_called_once_with(field_name="embedding", index_type="FLAT", metric_type="COSINE")
    client.create_collection.assert_called_once_with(
        collection_name=COLLECTION, schema=schema, index_params=index
    )


@pytest.mark.parametrize("index_type", ["FLAT", "BRUTE_FORCE"])
def test_open_checks_existing_collection_without_creating(db_path, sdk, index_type):
    factory, client = sdk
    client.describe_index.return_value["index_type"] = index_type

    MilvusStore(db_path, COLLECTION)

    factory.assert_called_once_with(uri=str(db_path.resolve()))
    assert client.method_calls == [
        call.has_collection(collection_name=COLLECTION),
        call.describe_collection(collection_name=COLLECTION),
        call.describe_index(collection_name=COLLECTION, index_name="embedding"),
        call.load_collection(collection_name=COLLECTION),
    ]


def test_open_refuses_missing_path_without_connecting(tmp_path, sdk):
    factory, _ = sdk
    with pytest.raises(FileNotFoundError, match="数据库不存在"):
        MilvusStore(tmp_path / "missing.db", COLLECTION)
    factory.assert_not_called()


def test_create_requires_explicit_boolean_flag(db_path, sdk):
    factory, _ = sdk
    with pytest.raises(TypeError, match="create"):
        MilvusStore(db_path, COLLECTION, create="false")
    factory.assert_not_called()


def test_open_refuses_missing_collection_and_closes_client(db_path, sdk):
    _, client = sdk
    client.has_collection.return_value = False

    with pytest.raises(LookupError, match="Collection 不存在"):
        MilvusStore(db_path, COLLECTION)

    client.create_collection.assert_not_called()
    client.load_collection.assert_not_called()
    client.close.assert_called_once_with()


def test_create_refuses_existing_collection(db_path, sdk):
    _, client = sdk
    with pytest.raises(ValueError, match="Collection 已存在"):
        MilvusStore(db_path, COLLECTION, create=True)
    client.create_collection.assert_not_called()
    client.close.assert_called_once_with()


@pytest.mark.parametrize("flag", ["auto_id", "enable_dynamic_field"])
def test_open_rejects_incompatible_schema_flags(db_path, sdk, flag):
    _, client = sdk
    client.describe_collection.return_value[flag] = True
    with pytest.raises(ValueError, match="Collection Schema"):
        MilvusStore(db_path, COLLECTION)
    client.load_collection.assert_not_called()
    client.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("field_name", "updated"),
    [
        ("embedding", {"params": {"dim": 256}}),
        ("kb_chunk_id", {"type": 5}),
        ("kb_chunk_id", {"is_primary": False}),
        ("law_name", {"params": {"max_length": 256}}),
        ("text", {"nullable": True}),
        ("source_url", {"name": "unexpected"}),
    ],
)
def test_open_rejects_incompatible_fields(db_path, sdk, field_name, updated):
    _, client = sdk
    fields = client.describe_collection.return_value["fields"]
    field = next(item for item in fields if item["name"] == field_name)
    field.update(updated)
    with pytest.raises(ValueError, match="Collection Schema 字段"):
        MilvusStore(db_path, COLLECTION)
    client.load_collection.assert_not_called()
    client.close.assert_called_once_with()


@pytest.mark.parametrize(
    "updated", [None, {"metric_type": "L2"}, {"index_type": "HNSW"}, {"field_name": "other"}]
)
def test_open_rejects_missing_or_incompatible_index(db_path, sdk, updated):
    _, client = sdk
    if updated is None:
        client.describe_index.return_value = None
    else:
        client.describe_index.return_value.update(updated)
    with pytest.raises(ValueError, match="FLAT/COSINE"):
        MilvusStore(db_path, COLLECTION)
    client.load_collection.assert_not_called()
    client.create_collection.assert_not_called()
    client.close.assert_called_once_with()


def test_upsert_keeps_primary_keys_original_text_and_vector_order(db_path, sdk):
    _, client = sdk
    client.upsert.return_value = {"upsert_count": 2}
    store = MilvusStore(db_path, COLLECTION)
    first = make_chunk()
    second = make_chunk(kb_chunk_id="chunk-581", document_id="civil-code-other")
    other_vector = VECTOR[::-1]

    assert store.upsert([first, second], [VECTOR, other_vector]) == 2

    client.upsert.assert_called_once_with(
        collection_name=COLLECTION,
        data=[
            {**first.model_dump(mode="json"), "embedding": VECTOR},
            {**second.model_dump(mode="json"), "embedding": other_vector},
        ],
    )
    assert client.upsert.call_args.kwargs["data"][0]["text"] == first.text


def test_upsert_refuses_misaligned_or_duplicate_batch_before_writing(db_path, sdk):
    _, client = sdk
    store = MilvusStore(db_path, COLLECTION)
    chunk = make_chunk()
    with pytest.raises(ValueError, match="zip"):
        store.upsert([chunk], [])
    with pytest.raises(ValueError, match="重复 kb_chunk_id"):
        store.upsert([chunk, chunk], [VECTOR, VECTOR])
    assert store.upsert([], []) == 0
    client.upsert.assert_not_called()


def test_upsert_revalidates_mutable_law_chunk(db_path, sdk):
    _, client = sdk
    store = MilvusStore(db_path, COLLECTION)
    chunk = make_chunk()
    chunk.text = " "
    with pytest.raises(ValidationError, match="text"):
        store.upsert([chunk], [VECTOR])
    client.upsert.assert_not_called()


def test_upsert_checks_utf8_byte_limit_without_truncation(db_path, sdk):
    _, client = sdk
    client.upsert.return_value = {"upsert_count": 1}
    store = MilvusStore(db_path, COLLECTION)
    max_text = "法" * 21845
    chunk = make_chunk(text=max_text, law_name="𠀀" * 256)
    assert store.upsert([chunk], [VECTOR]) == 1
    assert client.upsert.call_args.kwargs["data"][0]["text"] == max_text
    client.upsert.reset_mock()

    # 字符数未到 65535，但 UTF-8 字节数已经超限。
    with pytest.raises(ValueError, match=r"text .*65535.*UTF-8"):
        store.upsert([make_chunk(text=max_text + "法")], [VECTOR])
    client.upsert.assert_not_called()


def test_upsert_does_not_report_partial_count_as_success(db_path, sdk):
    _, client = sdk
    client.upsert.return_value = {"upsert_count": 0}
    store = MilvusStore(db_path, COLLECTION)
    with pytest.raises(RuntimeError, match="写入数量"):
        store.upsert([make_chunk()], [VECTOR])


def test_search_keeps_native_order_and_rebuilds_complete_chunks(db_path, sdk):
    _, client = sdk
    earlier_hit = make_chunk(kb_chunk_id="first")
    later_hit = make_chunk(kb_chunk_id="second")
    client.search.return_value = [[make_hit(earlier_hit, 0.2), make_hit(later_hit, 0.8, key="id")]]
    store = MilvusStore(db_path, COLLECTION)

    assert store.search(VECTOR, top_k=2) == [(earlier_hit, 0.2), (later_hit, 0.8)]

    client.search.assert_called_once_with(
        collection_name=COLLECTION,
        data=[VECTOR],
        anns_field="embedding",
        limit=2,
        filter="",
        search_params={"metric_type": "COSINE"},
        output_fields=["document_id", "law_name", "article_no", "text", "source_url"],
    )


def test_search_escapes_document_id_and_preserves_empty_match(db_path, sdk):
    _, client = sdk
    store = MilvusStore(db_path, COLLECTION)
    document_id = '民法典" or document_id == "other\\path'

    assert store.search(VECTOR, document_id=document_id) == []

    filter_expr = client.search.call_args.kwargs["filter"]
    assert filter_expr.startswith("document_id == ")
    assert json.loads(filter_expr.removeprefix("document_id == ")) == document_id


def test_search_uses_lite_lexer_escapes(db_path, sdk):
    _, client = sdk
    document_id = "第一行\n第二行\r\t编号\f1"
    MilvusStore(db_path, COLLECTION).search(VECTOR, document_id=document_id)

    # Lite 词法器支持 \n、\r、\t。换页符作为双引号内的原字符保留。
    assert client.search.call_args.kwargs["filter"] == 'document_id == "第一行\\n第二行\\r\\t编号\f1"'


@pytest.mark.parametrize("top_k", [0, -1, 16_385, True, 1.5])
def test_search_rejects_invalid_top_k_before_sdk_call(db_path, sdk, top_k):
    _, client = sdk
    with pytest.raises(ValueError, match="top_k"):
        MilvusStore(db_path, COLLECTION).search(VECTOR, top_k=top_k)
    client.search.assert_not_called()


@pytest.mark.parametrize(
    ("document_id", "error"), [(" ", ValueError), ("法" * 129, ValueError), (42, TypeError)]
)
def test_search_rejects_invalid_document_id(db_path, sdk, document_id, error):
    _, client = sdk
    with pytest.raises(error, match="document_id"):
        MilvusStore(db_path, COLLECTION).search(VECTOR, document_id=document_id)
    client.search.assert_not_called()


@pytest.mark.parametrize(
    ("vector", "error"),
    [
        (VECTOR[:-1], ValueError),
        ([float("nan"), *VECTOR[1:]], ValueError),
        ([float("inf"), *VECTOR[1:]], ValueError),
        ([-float("inf"), *VECTOR[1:]], ValueError),
        ([1e40, *VECTOR[1:]], ValueError),
        (["1.0", *VECTOR[1:]], TypeError),
        ([True, *VECTOR[1:]], TypeError),
    ],
)
def test_bad_vector_data_is_rejected_before_search_or_upsert(db_path, sdk, vector, error):
    _, client = sdk
    store = MilvusStore(db_path, COLLECTION)
    with pytest.raises(error, match="向量"):
        store.search(vector)
    with pytest.raises(error, match="向量"):
        store.upsert([make_chunk()], [vector])
    client.search.assert_not_called()
    client.upsert.assert_not_called()


@pytest.mark.parametrize("score", [float("nan"), float("inf"), "0.9"])
def test_search_rejects_invalid_hit_score(db_path, sdk, score):
    _, client = sdk
    client.search.return_value = [[make_hit(make_chunk(), score)]]
    with pytest.raises(RuntimeError, match="完整 LawChunk") as failure:
        MilvusStore(db_path, COLLECTION).search(VECTOR)
    assert isinstance(failure.value.__cause__, ValueError)


def test_search_rejects_incomplete_hit(db_path, sdk):
    _, client = sdk
    hit = make_hit(make_chunk(), 0.8)
    del hit["entity"]["text"]
    client.search.return_value = [[hit]]
    with pytest.raises(RuntimeError, match="完整 LawChunk") as failure:
        MilvusStore(db_path, COLLECTION).search(VECTOR)
    assert isinstance(failure.value.__cause__, KeyError)


def test_database_failures_propagate_without_fabricating_results(db_path, sdk):
    _, client = sdk
    store = MilvusStore(db_path, COLLECTION)
    cause = OSError("Milvus connection lost")
    client.search.side_effect = cause
    client.upsert.side_effect = cause
    with pytest.raises(OSError) as search_failure:
        store.search(VECTOR)
    with pytest.raises(OSError) as write_failure:
        store.upsert([make_chunk()], [VECTOR])
    assert search_failure.value is write_failure.value is cause


def test_failed_open_preserves_sdk_failure_even_if_cleanup_fails(db_path, sdk):
    _, client = sdk
    cause = OSError("describe failed")
    client.describe_collection.side_effect = cause
    client.close.side_effect = OSError("close failed")
    with pytest.raises(OSError) as failure:
        MilvusStore(db_path, COLLECTION)
    assert failure.value is cause
    assert any("close failed" in note for note in cause.__notes__)


def test_close_releases_client_once_and_rejects_later_operations(db_path, sdk):
    _, client = sdk
    store = MilvusStore(db_path, COLLECTION)
    store.close()
    store.close()
    client.close.assert_called_once_with()
    with pytest.raises(RuntimeError, match="已关闭"):
        store.search(VECTOR)
    with pytest.raises(RuntimeError, match="已关闭"):
        store.upsert([make_chunk()], [VECTOR])
    client.search.assert_not_called()
    client.upsert.assert_not_called()


def test_missing_sdk_is_reported_at_store_creation(db_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pymilvus", None)
    with pytest.raises(RuntimeError, match="PyMilvus") as failure:
        MilvusStore(db_path, COLLECTION)
    assert isinstance(failure.value.__cause__, ImportError)
