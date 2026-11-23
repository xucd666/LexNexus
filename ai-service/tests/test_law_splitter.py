import json
from hashlib import sha256
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest
from app.tools.law_splitter import split_law_articles
from pydantic import ValidationError

FIXTURE = Path(__file__).with_name("fixtures") / "civil_code_2020_articles_577_584"
EXPANDED_FIXTURE = FIXTURE.with_name("civil_code_2020_articles_577_594")
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

# 沿用 B1-1 的官网段落提取规则。从 SOURCE_URL 的 HTML 段落独立核对正文，
# 去除条号及其分隔符后，每段保留一个 LF。固定摘要不从被测 fixture 生成。
OFFICIAL_ADDED_ARTICLES = (
    ("第五百八十五条", "f62f4227d7411a011d47d47f323e3188b8e5ca8e56301246faed6147626fbe14"),
    ("第五百八十六条", "5f2fc9154d1426b67189d34d090c015d43600b21c28ba7a23e852fa945b27760"),
    ("第五百八十七条", "aec2f7cad85520e55bcfec875988d4f2ff2b76dfbce70c46071b1e8c72a39a70"),
    ("第五百八十八条", "cbe6d561400738d8127aa73e6955da9d31108d412cba14b1be34793f1dd1154a"),
    ("第五百八十九条", "ba9693e4f1ae1bc99a10c2c29871574faf6d17eaa15451d9180d6829b0c85f29"),
    ("第五百九十条", "89fd69d8f69b673f2b2ba9cf2de93408d9de28b2be7d0b4af5d0c8ef116151ad"),
    ("第五百九十一条", "b195d8df5add225c26f5e840bb3f0ff0eb3215432aa3151876c29de431ededb5"),
    ("第五百九十二条", "9dc9d2786cfc9ebdb865384c37ee31b17fe7765e67f2f12d26b32ca49cd0d73c"),
    ("第五百九十三条", "5d1299b33a60f3955bff284e36f6d65196ed874837b754deef51cb524d39116e"),
    ("第五百九十四条", "2c3a2c73ad80cbe98485099775de9ecdb5cc2c996b06db7a3b772eda651b1bf9"),
)


def _read_sample(path: Path):
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    # 按字节读取，避免 read_text 把 CRLF 自动转换成 LF 后掩盖正文变化。
    text = path.with_suffix(".txt").read_bytes().decode("utf-8")
    return text, {key: metadata[key] for key in ("document_id", "law_name", "source_url")}


@pytest.fixture
def sample():
    return _read_sample(FIXTURE)


@pytest.fixture
def expanded_sample():
    return _read_sample(EXPANDED_FIXTURE)


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


def test_expanded_sample_contains_exact_article_range_and_stable_ids(expanded_sample):
    text, metadata = expanded_sample
    chunks = split_law_articles(text, **metadata)
    expected_numbers = [number for number, _ in (*OFFICIAL_ARTICLES, *OFFICIAL_ADDED_ARTICLES)]
    document_id = "cn-civil-code-2020-05-28"

    assert len(chunks) == 18
    assert [chunk.article_no for chunk in chunks] == expected_numbers
    assert "第五百九十五条" not in text
    assert all(chunk.text.strip() for chunk in chunks)
    assert {chunk.document_id for chunk in chunks} == {document_id}
    assert len({chunk.kb_chunk_id for chunk in chunks}) == 18
    assert split_law_articles(text, **metadata) == chunks

    # 两个样本共用切分器。独立按现有 ID 规则比较，防止两组 ID 同时改变后仍通过。
    expected_ids = [
        uuid5(NAMESPACE_URL, f"urn:lexnexus:law-chunk:{document_id}:{number}").hex
        for number in expected_numbers
    ]
    assert [chunk.kb_chunk_id for chunk in chunks] == expected_ids


def test_expanded_sample_preserves_original_chunks(sample, expanded_sample):
    old_text, old_metadata = sample
    new_text, new_metadata = expanded_sample
    old_chunks = split_law_articles(old_text, **old_metadata)
    new_chunks = split_law_articles(new_text, **new_metadata)
    assert len(old_chunks) == 8

    # 比较 LawChunk 的全部六个字段，旧条文正文逐字符一致。
    for old_chunk, new_chunk in zip(old_chunks, new_chunks[:8], strict=True):
        assert new_chunk.model_dump() == old_chunk.model_dump(), old_chunk.article_no

    article_584 = new_chunks[7]
    assert article_584.article_no == "第五百八十四条"
    assert article_584.text.endswith("。\n")


def test_expanded_sample_preserves_version_metadata():
    original = json.loads(FIXTURE.with_suffix(".json").read_text(encoding="utf-8"))
    expanded = json.loads(EXPANDED_FIXTURE.with_suffix(".json").read_text(encoding="utf-8"))

    assert expanded == original
    assert expanded["version_basis"] == (
        "2020年5月28日由第十三届全国人民代表大会第三次会议通过，"
        "中华人民共和国主席令第四十五号公布，自2021年1月1日起施行。"
    )


def test_expanded_sample_matches_official_new_article_bodies(expanded_sample):
    text, metadata = expanded_sample
    chunks = split_law_articles(text, **metadata)
    added = {chunk.article_no: chunk.text for chunk in chunks[8:]}

    for article_no, expected_hash in OFFICIAL_ADDED_ARTICLES:
        assert sha256(added[article_no].encode("utf-8")).hexdigest() == expected_hash, article_no

    assert len(added["第五百八十五条"].splitlines()) == 3
    assert len(added["第五百九十条"].splitlines()) == 2
    assert chunks[-1].article_no == "第五百九十四条"
    assert chunks[-1].text.endswith("。\n")
