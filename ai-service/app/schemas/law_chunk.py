from typing import Annotated

from pydantic import BaseModel, ConfigDict, HttpUrl, StringConstraints

_ChunkIdentifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]


class LawChunk(BaseModel):
    """描述特定版本的法律文档切分后生成的一条知识片段。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    document_id: _ChunkIdentifier  # 标识特定版本的法律文档
    kb_chunk_id: _ChunkIdentifier  # 标识具体的知识片段
    law_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
    article_no: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
    # 要求正文包含非空白字符，保留输入原文。
    text: Annotated[str, StringConstraints(pattern=r"\S")]
    source_url: HttpUrl
