import json
from collections.abc import Iterable, Iterator
from enum import StrEnum
from pathlib import Path
from typing import Self

from app.schemas.contracts import Citation
from app.schemas.law_chunk import LawChunk
from app.tools.law_splitter import split_law_articles

# 将返回的 Citation 与独立可信原文进行核对，确保引用的文档版本、片段标识、元数据及正文与参考法条一致。


class CitationReferenceError(RuntimeError):
    """可信参考数据缺失、读取失败或包含无效记录。"""


class CitationIssue(StrEnum):
    """单条引用未通过校验的原因。"""

    MISSING_FIELD = "missing_field"
    UNKNOWN_DOCUMENT = "unknown_document"
    UNKNOWN_CHUNK = "unknown_chunk"
    DOCUMENT_VERSION_MISMATCH = "document_version_mismatch"
    LAW_NAME_MISMATCH = "law_name_mismatch"
    ARTICLE_NO_MISMATCH = "article_no_mismatch"
    QUOTED_TEXT_MISMATCH = "quoted_text_mismatch"


class CitationValidator:
    """按独立参考法条校验引用的文档版本、片段标识、元数据及正文。

    调用方应注入已核验的参考片段，例如从独立法条快照切分所得的 LawChunk。
    校验结论限于与该快照的一致性。片段由调用方保证来源可信。
    """

    def __init__(self, reference_chunks: Iterable[LawChunk]) -> None:
        """一次性读取并索引参考片段。生成器的读取异常也会转为 CitationReferenceError。"""
        chunks: dict[str, LawChunk] = {}
        document_ids: set[str] = set()
        try:
            for chunk in reference_chunks:
                if not isinstance(chunk, LawChunk):
                    raise TypeError("参考记录须为 LawChunk")
                # 重新校验并保存独立副本，避免调用方后续修改片段影响参考内容。
                reference = LawChunk.model_validate(chunk.model_dump(warnings=False))
                if reference.kb_chunk_id in chunks:
                    raise ValueError("参考数据包含重复的 kb_chunk_id")
                chunks[reference.kb_chunk_id] = reference
                document_ids.add(reference.document_id)
        except Exception as exc:
            raise CitationReferenceError("无法加载可信法条参考数据") from exc

        if not chunks:
            raise CitationReferenceError("可信法条参考数据为空")

        self._chunks = chunks
        self._document_ids = document_ids

    @classmethod
    def from_snapshot(cls, source_path: Path, metadata_path: Path) -> Self:
        """从已核验的本地 UTF-8 原文和 JSON 版本元数据装配校验器。"""

        def read_chunks() -> Iterator[LawChunk]:
            metadata = json.loads(metadata_path.expanduser().read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("法律快照元数据须为 JSON 对象")
            for field in ("document_id", "law_name", "source_url", "version_basis"):
                value = metadata.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"法律快照元数据字段 {field} 须为非空字符串")

            # 与离线入库使用相同的原文读取方式和切分规则，保留快照中的换行。
            text = source_path.expanduser().read_bytes().decode("utf-8")
            yield from split_law_articles(
                text,
                document_id=metadata["document_id"],
                law_name=metadata["law_name"],
                source_url=metadata["source_url"],
            )

        # 通过生成器将读取和解析纳入构造函数的参考数据异常边界。
        return cls(read_chunks())

    def validate(self, citation: Citation) -> CitationIssue | None:
        """返回首个不一致原因，通过时返回 None。引用本身保持原样。"""
        fields = (
            citation.document_id,
            citation.kb_chunk_id,
            citation.law_name,
            citation.article_no,
            citation.quoted_text,
        )
        if any(not isinstance(value, str) or not value.strip() for value in fields):
            return CitationIssue.MISSING_FIELD

        if citation.document_id not in self._document_ids:
            return CitationIssue.UNKNOWN_DOCUMENT
        reference = self._chunks.get(citation.kb_chunk_id)
        if reference is None:
            return CitationIssue.UNKNOWN_CHUNK
        # document_id 标识文档版本，已知文档和已知片段也必须属于同一版本。
        if citation.document_id != reference.document_id:
            return CitationIssue.DOCUMENT_VERSION_MISMATCH
        if citation.law_name != reference.law_name:
            return CitationIssue.LAW_NAME_MISMATCH
        if citation.article_no != reference.article_no:
            return CitationIssue.ARTICLE_NO_MISMATCH
        # 文本逐字符匹配，保留缩进、首尾空格及换行差异。
        if citation.quoted_text != reference.text:
            return CitationIssue.QUOTED_TEXT_MISMATCH
        return None

    def validate_many(self, citations: Iterable[Citation]) -> list[CitationIssue | None]:
        """按输入顺序返回校验结果。空引用批次返回空列表。"""
        return [self.validate(citation) for citation in citations]
