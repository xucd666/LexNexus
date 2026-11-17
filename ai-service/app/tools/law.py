from typing import Protocol

from app.schemas.contracts import Citation


class LawSearch(Protocol):
    async def search(self, query: str, *, request_id: str, trace_id: str | None) -> list[Citation]: ...


class FakeLawTool:
    """按关键词返回固定测试材料。"""

    async def search(self, query: str, *, request_id: str, trace_id: str | None) -> list[Citation]:
        if "违约金" not in query:
            return []
        return [
            Citation(
                document_id="fake-law-1",
                kb_chunk_id="fake-chunk-1",
                law_name="LexNexus 离线测试材料",
                article_no="测试条目 1",
                quoted_text="这是用于验证违约金问题检索和引用传递的测试文本。",
                relevance_score=1.0,
            )
        ]
