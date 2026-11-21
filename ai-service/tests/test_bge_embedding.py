import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
from app.tools.bge_embedding import DEFAULT_QUERY_INSTRUCTION, BgeEmbedding

UNIT_VECTOR = [1.0, *([0.0] * 511)]


class StubMatrix:
    """提供模块使用的 NumPy 数组属性，单测无需安装模型依赖。"""

    def __init__(self, rows, *, dtype="float32"):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))
        self.dtype = dtype

    def __getitem__(self, index):
        return self.rows[index]


@pytest.fixture
def backend(monkeypatch):
    def tokenize(texts, *, add_special_tokens, padding, truncation):
        special_tokens = 2 if add_special_tokens else 0
        return {"input_ids": [[0] * (len(text) + special_tokens) for text in texts]}

    model = Mock(max_seq_length=512)
    model.tokenizer = Mock(model_max_length=512, side_effect=tokenize)
    model.encode.return_value = StubMatrix([UNIT_VECTOR])
    load = Mock(return_value=model)
    package = ModuleType("sentence_transformers")
    package.SentenceTransformer = load
    monkeypatch.setitem(sys.modules, package.__name__, package)
    return load, model


def test_encodes_single_query_and_loads_model_once(tmp_path, backend):
    load, model = backend
    embedding = BgeEmbedding(tmp_path)
    question = "如何主张违约责任？"

    for _ in range(2):
        assert embedding.encode_query(question) == UNIT_VECTOR

    load.assert_called_once_with(
        str(tmp_path.resolve()), device="cpu", local_files_only=True, trust_remote_code=False
    )
    model.tokenizer.assert_called_with(
        [DEFAULT_QUERY_INSTRUCTION + question], add_special_tokens=True, padding=False, truncation=False
    )
    model.encode.assert_called_with(
        [DEFAULT_QUERY_INSTRUCTION + question],
        device="cpu",
        prompt="",
        convert_to_numpy=True,
        precision="float32",
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    assert model.encode.call_count == 2


def test_encodes_batch_without_prefix_and_preserves_input_order(tmp_path, backend):
    _, model = backend
    embedding = BgeEmbedding(tmp_path, query_instruction="检索：")
    documents = ("  第一条原文\n", "第二条原文")
    matrix = StubMatrix([UNIT_VECTOR, [0.0, *UNIT_VECTOR[:-1]]])
    model.encode.return_value = matrix

    assert embedding.encode_documents(documents) is matrix
    assert matrix.shape == (2, 512)
    assert matrix.dtype == "float32"
    assert model.tokenizer.call_args.args == (list(documents),)
    assert model.encode.call_args.args == (list(documents),)


def test_token_check_counts_prefix_and_special_tokens_before_encoding(tmp_path, backend):
    _, model = backend
    model.max_seq_length = 8
    embedding = BgeEmbedding(tmp_path, query_instruction="检索：")

    with pytest.raises(ValueError, match=r"9 tokens.*8 token limit"):
        embedding.encode_query("合同违约")

    model.tokenizer.assert_called_once_with(
        ["检索：合同违约"], add_special_tokens=True, padding=False, truncation=False
    )
    model.encode.assert_not_called()


def test_uses_stricter_tokenizer_limit_and_accepts_exact_boundary(tmp_path, backend):
    _, model = backend
    model.tokenizer.model_max_length = 6
    embedding = BgeEmbedding(tmp_path, query_instruction="")

    assert embedding.encode_query("合同违约") == UNIT_VECTOR
    assert model.encode.call_args.args == (["合同违约"],)
    model.encode.reset_mock()

    with pytest.raises(ValueError, match=r"input 2 has 7 tokens.*6 token limit"):
        embedding.encode_documents(["短文本", "合同违约金"])
    model.encode.assert_not_called()


@pytest.mark.parametrize("text", ["", " \t\n　"])
def test_rejects_blank_queries_even_with_a_prefix(tmp_path, backend, text):
    _, model = backend
    embedding = BgeEmbedding(tmp_path)

    with pytest.raises(ValueError, match="must not be blank"):
        embedding.encode_query(text)

    model.tokenizer.assert_not_called()
    model.encode.assert_not_called()


def test_rejects_invalid_query_type(tmp_path, backend):
    _, model = backend
    embedding = BgeEmbedding(tmp_path)

    with pytest.raises(TypeError, match="must be a string"):
        embedding.encode_query(None)

    model.tokenizer.assert_not_called()


@pytest.mark.parametrize(
    ("documents", "error", "message"),
    [
        ([], ValueError, "must not be empty"),
        (["有效正文", "　"], ValueError, "must not be blank"),
        ("不是文档批次", TypeError, "sequence of strings"),
        (["有效正文", 123], TypeError, "must be a string"),
    ],
)
def test_rejects_invalid_document_batches(tmp_path, backend, documents, error, message):
    _, model = backend
    embedding = BgeEmbedding(tmp_path)

    with pytest.raises(error, match=message):
        embedding.encode_documents(documents)

    model.tokenizer.assert_not_called()


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (StubMatrix([UNIT_VECTOR[:-1]]), "embedding shape"),
        (StubMatrix([UNIT_VECTOR, UNIT_VECTOR]), "embedding shape"),
        (StubMatrix([UNIT_VECTOR], dtype="float64"), "must be float32"),
        (StubMatrix([[float("nan"), *UNIT_VECTOR[1:]]]), "NaN or Inf"),
        (StubMatrix([[float("inf"), *UNIT_VECTOR[1:]]]), "NaN or Inf"),
        (StubMatrix([[0.0] * 512]), "not L2-normalized"),
    ],
)
def test_rejects_invalid_encoder_output(tmp_path, backend, matrix, message):
    _, model = backend
    model.encode.return_value = matrix

    with pytest.raises(RuntimeError, match=message):
        BgeEmbedding(tmp_path).encode_query("有效问题")


def test_rejects_missing_snapshot_before_model_load(tmp_path, backend):
    load, _ = backend

    with pytest.raises(FileNotFoundError, match="snapshot does not exist"):
        BgeEmbedding(tmp_path / "missing")

    load.assert_not_called()


def test_reports_missing_runtime_dependency_on_construction(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(RuntimeError, match="requires sentence-transformers"):
        BgeEmbedding(tmp_path)


def test_reports_load_failure_without_losing_the_cause(tmp_path, backend):
    load, _ = backend
    load.side_effect = OSError("invalid snapshot")

    with pytest.raises(RuntimeError, match="Failed to load the local BGE model") as error:
        BgeEmbedding(tmp_path)

    assert error.value.__cause__ is load.side_effect


def test_reports_encoding_failure_without_losing_the_cause(tmp_path, backend):
    _, model = backend
    model.encode.side_effect = RuntimeError("device error")

    with pytest.raises(RuntimeError, match="BGE encoding failed") as error:
        BgeEmbedding(tmp_path).encode_query("有效问题")

    assert error.value.__cause__ is model.encode.side_effect
