import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from app.schemas.contracts import Citation
from app.schemas.law_chunk import LawChunk
from app.tools.citation_validator import CitationIssue, CitationReferenceError, CitationValidator
from app.tools.law_splitter import split_law_articles

FIXTURE_DIR = Path(__file__).with_name("fixtures")
SAMPLE_NAME = "civil_code_2020_articles_577_584"


def _read_reference_chunks(directory: Path) -> Iterator[LawChunk]:
    # 从原文和版本元数据独立构建参考数据。先读字节以保留样本的换行序列。
    metadata = json.loads((directory / f"{SAMPLE_NAME}.json").read_text(encoding="utf-8"))
    text = (directory / f"{SAMPLE_NAME}.txt").read_bytes().decode("utf-8")
    yield from split_law_articles(
        text,
        document_id=metadata["document_id"],
        law_name=metadata["law_name"],
        source_url=metadata["source_url"],
    )


@pytest.fixture
def reference_chunks() -> list[LawChunk]:
    return list(_read_reference_chunks(FIXTURE_DIR))


@pytest.fixture
def validator(reference_chunks: list[LawChunk]) -> CitationValidator:
    return CitationValidator(reference_chunks)


@pytest.fixture
def citation(reference_chunks: list[LawChunk]) -> Citation:
    chunk = next(chunk for chunk in reference_chunks if chunk.article_no == "第五百八十条")
    return Citation(
        document_id=chunk.document_id,
        kb_chunk_id=chunk.kb_chunk_id,
        law_name=chunk.law_name,
        article_no=chunk.article_no,
        quoted_text=chunk.text,
    )


def test_accepts_exact_reference_with_optional_score_absent(validator: CitationValidator, citation: Citation):
    assert citation.relevance_score is None
    assert validator.validate(citation) is None


@pytest.mark.parametrize(
    ("field", "value", "issue"),
    [
        ("kb_chunk_id", "forged-chunk", CitationIssue.UNKNOWN_CHUNK),
        ("document_id", "unknown-version", CitationIssue.UNKNOWN_DOCUMENT),
        ("law_name", "中华人民共和国合同法", CitationIssue.LAW_NAME_MISMATCH),
        ("article_no", "第五百七十九条", CitationIssue.ARTICLE_NO_MISMATCH),
        ("quoted_text", "这段文字不是参考法条。", CitationIssue.QUOTED_TEXT_MISMATCH),
    ],
)
def test_rejects_altered_fields_without_editing_citation(
    validator: CitationValidator, citation: Citation, field: str, value: str, issue: CitationIssue
):
    altered = citation.model_copy(update={field: value})

    assert validator.validate(altered) is issue
    assert getattr(altered, field) == value


def test_rejects_chunk_from_a_different_known_document_version(
    reference_chunks: list[LawChunk], citation: Citation
):
    # 增加一个测试专用版本，让两个文档版本都已知，再尝试跨版本引用。
    other_version = reference_chunks[0].model_copy(
        update={"document_id": "test-other-version", "kb_chunk_id": "test-other-version-chunk"}
    )
    validator = CitationValidator([*reference_chunks, other_version])
    cross_version = citation.model_copy(update={"document_id": other_version.document_id})

    assert validator.validate(cross_version) is CitationIssue.DOCUMENT_VERSION_MISMATCH


@pytest.mark.parametrize("field", ["document_id", "kb_chunk_id", "law_name", "article_no", "quoted_text"])
@pytest.mark.parametrize("blank", [None, "", " \n\t"])
def test_rejects_missing_or_blank_fields(
    validator: CitationValidator, citation: Citation, field: str, blank: str | None
):
    fields = citation.model_dump(exclude={field})
    if blank is not None:
        fields[field] = blank
    incomplete = Citation.model_validate(fields)

    assert validator.validate(incomplete) is CitationIssue.MISSING_FIELD


def test_rejects_changes_to_whitespace_and_line_endings(validator: CitationValidator, citation: Citation):
    text = citation.quoted_text
    assert text is not None
    assert "\n" in text

    other_line_endings = text.replace("\r\n", "\n") if "\r\n" in text else text.replace("\n", "\r\n")
    for changed in (" " + text, text + " ", other_line_endings):
        altered = citation.model_copy(update={"quoted_text": changed})
        assert validator.validate(altered) is CitationIssue.QUOTED_TEXT_MISMATCH


def test_uses_reference_snapshot_after_original_chunk_is_edited(
    validator: CitationValidator, citation: Citation, reference_chunks: list[LawChunk]
):
    chunk = next(chunk for chunk in reference_chunks if chunk.kb_chunk_id == citation.kb_chunk_id)
    chunk.text = "修改后的正文"

    assert validator.validate(citation) is None


def test_reports_reference_file_read_failure(tmp_path: Path):
    with pytest.raises(CitationReferenceError, match="无法加载") as error:
        CitationValidator(_read_reference_chunks(tmp_path))

    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_reports_reference_iterating_failure_even_after_reading_a_chunk(reference_chunks: list[LawChunk]):
    def broken_source() -> Iterator[LawChunk]:
        yield reference_chunks[0]
        raise OSError("读取中断")

    with pytest.raises(CitationReferenceError, match="无法加载") as error:
        CitationValidator(broken_source())

    assert isinstance(error.value.__cause__, OSError)


def test_rejects_empty_reference_data():
    with pytest.raises(CitationReferenceError, match="参考数据为空"):
        CitationValidator([])


def test_rejects_duplicate_reference_chunk_ids(reference_chunks: list[LawChunk]):
    with pytest.raises(CitationReferenceError, match="无法加载") as error:
        CitationValidator([reference_chunks[0], reference_chunks[0]])

    assert isinstance(error.value.__cause__, ValueError)
    assert "重复的 kb_chunk_id" in str(error.value.__cause__)


def test_rejects_reference_chunk_with_invalid_body(reference_chunks: list[LawChunk]):
    invalid_chunk = reference_chunks[0].model_copy(update={"text": " \n"})

    with pytest.raises(CitationReferenceError, match="无法加载"):
        CitationValidator([invalid_chunk])


def test_validates_batch_in_input_order(validator: CitationValidator, citation: Citation):
    invalid = citation.model_copy(update={"article_no": "第五百七十七条"})

    assert validator.validate_many([invalid, citation]) == [CitationIssue.ARTICLE_NO_MISMATCH, None]
    assert validator.validate_many([]) == []
