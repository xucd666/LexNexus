import json
from hashlib import sha256
from pathlib import Path

import pytest
from app.tools.law_splitter import split_law_articles
from pydantic import ValidationError

FIXTURE = Path(__file__).with_name("fixtures") / "civil_code_2020_articles_577_584"
SOURCE_URL = "https://www.beijing.gov.cn/zhengce/zhengcefagui/202006/t20200602_1913330.html"

# 从官网各条的 HTML 段落独立计算正文摘要，每段使用一个 LF 结束。
OFFICIAL_ARTICLES = (
    ("第五百七十七条", "474f6967d0f020aa8481bdb78fd8643650e7f24fd71b3b59407f1926560d6042"),
    ("第五百七十八条", "6d54b1a74fb3af7865f8974e984079aa5e7df12be799c221202ff557b49c8d14"),
    ("第五百七十九条", "5a749f4d3903295a652249f92ff5c12e4f78e1f5c35ef2ac8925b1fe6a2a168c"),
    ("第五百八十条", "0a5430370b67100cc9ff5e420cea89bb67bcff749a1cc9820b00c35c94d08110"),
    ("第五百八十一条", "56404387f5fb8f00c49c1f09f576fa49ff38871cffb3830256621f9c0a65ee76"),
    ("第五百八十二条", "e70a9d5c03e2915ac9c8f3bca6415b35c9ccb470b660ff2d15a77b45a3930f0e"),
    ("第五百八十三条", "4d4e2d6f331641971aceed5d07d36700cfdd4da35113edcb5ca77208bc7bc35c"),
    ("第五百八十四条", "16ba1ccfc308f06452f1d9a775b18b75b5ca27522dfb32e0c1a45dec10d3c4f5"),
)


@pytest.fixture
def sample():
    metadata = json.loads(FIXTURE.with_suffix(".json").read_text(encoding="utf-8"))
    text = FIXTURE.with_suffix(".txt").read_text(encoding="utf-8")
    return text, {key: metadata[key] for key in ("document_id", "law_name", "source_url")}


def test_splits_official_sample_without_losing_text(sample):
    text, metadata = sample
    chunks = split_law_articles(text, **metadata)

    assert len(chunks) == 8
    expected_numbers, expected_hashes = zip(*OFFICIAL_ARTICLES, strict=True)
    assert tuple(chunk.article_no for chunk in chunks) == expected_numbers
    assert {chunk.document_id for chunk in chunks} == {"cn-civil-code-2020-05-28"}
    assert {chunk.law_name for chunk in chunks} == {"中华人民共和国民法典"}
    assert {str(chunk.source_url) for chunk in chunks} == {SOURCE_URL}
    assert tuple(sha256(chunk.text.encode("utf-8")).hexdigest() for chunk in chunks) == expected_hashes


def test_keeps_all_paragraphs_and_items_in_article_580(sample):
    text, metadata = sample
    chunks = split_law_articles(text, **metadata)
    article = next(chunk for chunk in chunks if chunk.article_no == "第五百八十条")

    first, *items, final = article.text.splitlines()
    assert first.endswith("有下列情形之一的除外：")
    assert [line.lstrip()[:3] for line in items] == ["(一)", "(二)", "(三)"]
    assert final.startswith("\u3000\u3000有前款规定")
    assert final.endswith("不影响违约责任的承担。")


def test_generates_stable_ids_per_document_version_and_article(sample):
    text, metadata = sample
    chunks = split_law_articles(text, **metadata)

    assert split_law_articles(text, **metadata) == chunks
    assert len({chunk.kb_chunk_id for chunk in chunks}) == 8

    different_version = split_law_articles(text, **(metadata | {"document_id": "other-version"}))
    assert {chunk.kb_chunk_id for chunk in chunks}.isdisjoint(
        chunk.kb_chunk_id for chunk in different_version
    )


def test_preserves_body_whitespace_and_inline_article_references(sample):
    _, metadata = sample
    text = "　第一条　 首款  \r\n第二条的规定适用于本款。\r\n\r\n\t第二条\r\n  次款。\r\n"

    first, second = split_law_articles(text, **metadata)

    assert first.text == " 首款  \r\n第二条的规定适用于本款。\r\n\r\n"
    assert second.text == "\r\n  次款。\r\n"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "法律文本为空"),
        (" \n　", "法律文本为空"),
        ("按第一条处理。", "未找到行首法律条号"),
        ("未解析的导言\n第一条　正文。", "首个条号前存在未解析的文本"),
        ("第一条　正文。\n第一条　重复正文。", "重复条号：第一条"),
        ("第一条\n　\n第二条　正文。", "第一条正文为空"),
    ],
)
def test_rejects_invalid_article_text(sample, text, message):
    _, metadata = sample

    with pytest.raises(ValueError, match=message):
        split_law_articles(text, **metadata)


def test_validates_chunks_with_existing_schema(sample):
    text, metadata = sample

    with pytest.raises(ValidationError, match="source_url"):
        split_law_articles(text, **(metadata | {"source_url": "ftp://example.invalid/law"}))
