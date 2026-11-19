import re
from uuid import NAMESPACE_URL, uuid5

from app.schemas.law_chunk import LawChunk

_ARTICLE_HEADER = re.compile(
    r"^[ \t\u00a0\u3000]*(第[〇零一二三四五六七八九十百千万亿两]+条)(?:[ \t\u00a0\u3000]|(?=\r?$))",
    re.MULTILINE,
)


def split_law_articles(text: str, *, document_id: str, law_name: str, source_url: str) -> list[LawChunk]:
    """按行首条号切分法条。条号后接空白分隔符或单独占一行。"""
    if not text.strip():
        raise ValueError("法律文本为空")

    headers = list(_ARTICLE_HEADER.finditer(text))
    if not headers:
        raise ValueError("未找到行首法律条号")
    if text[: headers[0].start()].strip():
        raise ValueError("首个条号前存在未解析的文本")

    document_id = document_id.strip()
    chunks = []
    seen = set()
    for index, header in enumerate(headers):
        article_no = header[1]
        if article_no in seen:
            raise ValueError(f"重复条号：{article_no}")
        seen.add(article_no)

        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        body = text[header.end() : end]
        if not body.strip():
            raise ValueError(f"{article_no}正文为空")

        chunks.append(
            LawChunk(
                document_id=document_id,
                kb_chunk_id=uuid5(NAMESPACE_URL, f"urn:lexnexus:law-chunk:{document_id}:{article_no}").hex,
                law_name=law_name,
                article_no=article_no,
                text=body,
                source_url=source_url,
            )
        )

    return chunks
