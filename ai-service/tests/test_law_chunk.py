import pytest
from app.schemas.law_chunk import LawChunk
from pydantic import HttpUrl, ValidationError

METADATA_FIELDS = ("document_id", "kb_chunk_id", "law_name", "article_no")
REQUIRED_FIELDS = (*METADATA_FIELDS, "text", "source_url")


@pytest.fixture
def chunk_data():
    return {
        "document_id": "example-law-2026",
        "kb_chunk_id": "example-law-2026-article-1",
        "law_name": "示例法",
        "article_no": "第一条",
        "text": "此处为测试用法律正文。",
        "source_url": "https://example.invalid/laws/2026",
    }


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_accepts_valid_chunk(chunk_data, scheme):
    chunk_data["source_url"] = f"{scheme}://example.invalid/laws/2026"

    chunk = LawChunk.model_validate(chunk_data)

    assert isinstance(chunk.source_url, HttpUrl)
    assert chunk.model_dump(mode="json") == chunk_data


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_rejects_missing_field(chunk_data, field):
    del chunk_data[field]

    with pytest.raises(ValidationError) as error:
        LawChunk.model_validate(chunk_data)

    assert error.value.errors()[0]["loc"] == (field,)
    assert error.value.errors()[0]["type"] == "missing"


@pytest.mark.parametrize("field", METADATA_FIELDS)
@pytest.mark.parametrize("value", ["", " \t\n\u3000"])
def test_rejects_blank_metadata(chunk_data, field, value):
    chunk_data[field] = value

    with pytest.raises(ValidationError) as error:
        LawChunk.model_validate(chunk_data)

    assert error.value.errors()[0]["loc"] == (field,)
    assert error.value.errors()[0]["type"] == "string_too_short"


@pytest.mark.parametrize(
    ("field", "limit"), [("document_id", 128), ("kb_chunk_id", 128), ("law_name", 256), ("article_no", 64)]
)
def test_checks_metadata_length_after_trimming(chunk_data, field, limit):
    at_limit = "法" * limit
    chunk_data[field] = f" \t{at_limit}\n "
    assert getattr(LawChunk.model_validate(chunk_data), field) == at_limit

    chunk_data[field] = f" {at_limit}法 "
    with pytest.raises(ValidationError) as error:
        LawChunk.model_validate(chunk_data)

    assert error.value.errors()[0]["loc"] == (field,)
    assert error.value.errors()[0]["type"] == "string_too_long"


@pytest.mark.parametrize("text", ["", " \t\r\n", "\u3000\xa0"])
def test_rejects_blank_text(chunk_data, text):
    chunk_data["text"] = text

    with pytest.raises(ValidationError) as error:
        LawChunk.model_validate(chunk_data)

    assert error.value.errors()[0]["loc"] == ("text",)


def test_preserves_original_text(chunk_data):
    text = " \t第一段  保留空格\r\n    第二段\n\t第三段  \n " * 300
    chunk_data["text"] = text

    assert LawChunk.model_validate(chunk_data).text == text


@pytest.mark.parametrize("url", ["not-a-url", "https://", "ftp://example.invalid/law", "file:///law"])
def test_rejects_invalid_source_url(chunk_data, url):
    chunk_data["source_url"] = url

    with pytest.raises(ValidationError) as error:
        LawChunk.model_validate(chunk_data)

    assert error.value.errors()[0]["loc"] == ("source_url",)


def test_rejects_extra_field(chunk_data):
    chunk_data["unexpected"] = "value"

    with pytest.raises(ValidationError) as error:
        LawChunk.model_validate(chunk_data)

    assert error.value.errors()[0]["loc"] == ("unexpected",)
    assert error.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize(
    ("field", "value"), [(field, 123) for field in REQUIRED_FIELDS] + [("text", b"some text")]
)
def test_rejects_wrong_field_type(chunk_data, field, value):
    chunk_data[field] = value

    with pytest.raises(ValidationError) as error:
        LawChunk.model_validate(chunk_data)

    assert error.value.errors()[0]["loc"] == (field,)
