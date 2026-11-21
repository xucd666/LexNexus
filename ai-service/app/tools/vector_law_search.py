"""将同步 BGE 编码与 Milvus 查询接到 LawSearch 的异步调用契约。"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from app.schemas.contracts import Citation

if TYPE_CHECKING:
    from app.tools.bge_embedding import BgeEmbedding
    from app.tools.milvus_store import MilvusStore


class VectorLawSearch:
    """在同一事件循环内复用已初始化的模型和存储，每个实例串行执行检索。

    依赖由调用方持有并关闭。共享同一对依赖的请求应共用一个适配器，
    调用方需等在途同步操作结束后再关闭存储。
    """

    def __init__(self, embedding: BgeEmbedding, store: MilvusStore, *, top_k: int = 3) -> None:
        """注入现有模型和存储。top_k 采用 MilvusStore 的有效范围。"""
        if type(top_k) is not int or not 1 <= top_k <= 16_384:
            raise ValueError("top_k 必须是 1 至 16384 的整数")
        self._embedding = embedding
        self._store = store
        self._top_k = top_k
        self._search_lock = asyncio.Lock()

    async def search(self, query: str, *, request_id: str, trace_id: str | None) -> list[Citation]:
        """按存储返回顺序生成引用，成功且无命中时返回空列表。

        query 须为非空白字符串，具体 Token 上限由编码器检查。
        request_id 和 trace_id 遵循 LawSearch 契约，当前底层接口不接收这两个字段。
        编码、存储或引用构造的异常直接交给调用方。
        """
        if not isinstance(query, str):
            raise TypeError("query 必须是字符串")
        if not query.strip():
            raise ValueError("query 不能为空或纯空白")

        # 在提交前异步等待，使每个实例最多占用一个工作线程。None 表示使用事件循环的默认线程池。
        await self._search_lock.acquire()
        try:
            worker = asyncio.get_running_loop().run_in_executor(None, self._search_sync, query)
        except Exception:
            self._search_lock.release()
            raise
        # 请求取消只结束等待。同步工作无法随协程中断，锁交由完成回调释放。
        worker.add_done_callback(self._on_search_done)
        return await asyncio.shield(worker)

    def _on_search_done(self, worker: asyncio.Future[list[Citation]]) -> None:
        self._search_lock.release()
        if not worker.cancelled():
            # 请求提前退出时也须读取异常，避免事件循环报告未处理异常。正常 await 仍会收到同一异常。
            worker.exception()

    def _search_sync(self, query: str) -> list[Citation]:
        embedding = self._embedding.encode_query(query)
        hits = self._store.search(embedding, top_k=self._top_k)
        # 命中已由存储还原为 LawChunk。正文直接进入引用，分数和排序均沿用存储结果。
        return [
            Citation(
                document_id=chunk.document_id,
                kb_chunk_id=chunk.kb_chunk_id,
                law_name=chunk.law_name,
                article_no=chunk.article_no,
                quoted_text=chunk.text,
                relevance_score=score,
            )
            for chunk, score in hits
        ]
