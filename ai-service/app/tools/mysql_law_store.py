from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from typing import TYPE_CHECKING

import pymysql
from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from pymysql.constants import ER
from pymysql.cursors import DictCursor

from app.schemas.law_chunk import LawChunk

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pymysql.connections import Connection

_SELECT_VERSION = """
    SELECT document_id, law_id, law_name, version_basis, source_url,
           promulgation_date, effective_date, repeal_date
    FROM law_versions
    WHERE document_id = %s
"""


class LawVersion(BaseModel):
    """登记官方文本版本所需的元数据。法律日期由调用方核实后显式提供。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    document_id: str = Field(max_length=128, pattern=r"\S")
    law_id: str = Field(max_length=128, pattern=r"\S")
    law_name: str = Field(max_length=256, pattern=r"\S")
    version_basis: str = Field(pattern=r"\S")
    source_url: HttpUrl
    promulgation_date: date
    effective_date: date
    repeal_date: date | None = None


@dataclass(frozen=True, slots=True)
class LawWriteResult:
    """记录一次已提交批次的写入和跳过数量。"""

    version_inserted: bool
    articles_inserted: int
    articles_skipped: int


class LawStoreConflictError(RuntimeError):
    """目标标识已存在，但映射或内容不一致，当前批次不能提交。"""


class MySQLLawStore:
    """使用单条专用连接读写权威法律数据，由调用方在使用结束后关闭。"""

    def __init__(self, connection: Connection) -> None:
        """接收专用连接，要求其游标返回字典且调用方未开启外部事务。"""
        self._connection = connection

    @classmethod
    def connect(cls, *, database: str, user: str, host: str = "127.0.0.1", port: int = 3306) -> MySQLLawStore:
        """用最小权限账号建立连接。密码仅在连接时从环境变量读取。"""
        if not user.strip() or user.strip().casefold() == "root":
            raise ValueError("请使用专用 MySQL 应用账号")
        password = os.environ.get("AI_SERVICE_MYSQL_PASSWORD")
        if not password:
            raise ValueError("请设置 AI_SERVICE_MYSQL_PASSWORD")
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset="utf8mb4",
            collation="utf8mb4_0900_bin",
            sql_mode="TRADITIONAL",
            init_command="SET SESSION time_zone = '+00:00'",
            cursorclass=DictCursor,
            # 单次读取结束后即释放当前读视图。批量写入显式开启事务。
            autocommit=True,
        )
        return cls(connection)

    def close(self) -> None:
        """释放本实例持有的连接，允许重复调用。"""
        if self._connection.open:
            self._connection.close()

    def get_version(self, document_id: str) -> LawVersion | None:
        """按版本主键查询元数据。不存在时返回 None。"""
        with self._connection.cursor() as cursor:
            cursor.execute(_SELECT_VERSION, (document_id,))
            row = cursor.fetchone()
        return None if row is None else LawVersion.model_validate(row)

    def get_article(self, kb_chunk_id: str) -> LawChunk | None:
        """按片段主键读取条文，并从版本表取得 LawChunk 所需的法律名称。"""
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT a.kb_chunk_id, a.document_id, v.law_name, a.article_no,
                       a.`text`, a.source_url
                FROM law_articles AS a
                JOIN law_versions AS v ON v.document_id = a.document_id
                WHERE a.kb_chunk_id = %s
                """,
                (kb_chunk_id,),
            )
            row = cursor.fetchone()
        return None if row is None else LawChunk.model_validate(row)

    def write_batch(self, version: LawVersion, chunks: Iterable[LawChunk]) -> LawWriteResult:
        """写入一个版本及其条文。内容相同则跳过，任何冲突或写入错误均回滚批次。"""
        connection = self._connection
        inserted = skipped = 0
        connection.begin()
        try:
            with connection.cursor() as cursor:
                version_inserted = self._write_version(cursor, version)
                for supplied_chunk in chunks:
                    # 以回读格式重新校验，确保模型被修改后仍满足正文和标识约束。
                    chunk = LawChunk.model_validate(supplied_chunk.model_dump(mode="json"))
                    if chunk.document_id != version.document_id or chunk.law_name != version.law_name:
                        raise LawStoreConflictError("待写入条文的文档版本或法律名称与批次元数据不一致")
                    if self._write_article(cursor, chunk):
                        inserted += 1
                    else:
                        skipped += 1
            connection.commit()
        except BaseException as exc:
            try:
                connection.rollback()
            except pymysql.MySQLError as rollback_error:
                exc.add_note(f"MySQL 回滚失败，关闭连接以停止使用该会话（{type(rollback_error).__name__}）")
                self.close()
            # 唯一键可能在查询后由其他会话占用。失败时交给调用方确认最新数据并重新运行。
            if isinstance(exc, pymysql.IntegrityError) and exc.args[:1] == (ER.DUP_ENTRY,):
                raise LawStoreConflictError("写入时 MySQL 拒绝了重复唯一键，当前批次未被接受") from exc
            raise
        return LawWriteResult(version_inserted, inserted, skipped)

    @staticmethod
    def _write_version(cursor: DictCursor, version: LawVersion) -> bool:
        # 保留 DATE 参数的 Python date 类型，同时检查序列化后的 URL 可在回读时重新校验。
        expected = version.model_dump()
        expected["source_url"] = str(HttpUrl(str(version.source_url)))
        cursor.execute(_SELECT_VERSION, (version.document_id,))
        existing = cursor.fetchone()
        if existing is not None:
            if existing != expected:
                raise LawStoreConflictError(f"法律版本元数据冲突（{version.document_id}）")
            return False
        cursor.execute(
            """
            INSERT INTO law_versions (
                document_id, law_id, law_name, version_basis, source_url,
                promulgation_date, effective_date, repeal_date
            ) VALUES (
                %(document_id)s, %(law_id)s, %(law_name)s, %(version_basis)s, %(source_url)s,
                %(promulgation_date)s, %(effective_date)s, %(repeal_date)s
            )
            """,
            expected,
        )
        return True

    @staticmethod
    def _write_article(cursor: DictCursor, chunk: LawChunk) -> bool:
        expected = chunk.model_dump(mode="json", exclude={"law_name"})
        expected["text_sha256"] = sha256(chunk.text.encode("utf-8")).digest()
        # 一条记录可以同时命中两个键。如果返回两条，则两个键分别指向了不同记录。
        cursor.execute(
            """
            SELECT kb_chunk_id, document_id, article_no, `text`, text_sha256, source_url
            FROM law_articles
            WHERE kb_chunk_id = %s OR (document_id = %s AND article_no = %s)
            """,
            (chunk.kb_chunk_id, chunk.document_id, chunk.article_no),
        )
        existing = cursor.fetchall()
        if existing:
            if len(existing) != 1 or any(
                existing[0][key] != expected[key] for key in ("kb_chunk_id", "document_id", "article_no")
            ):
                raise LawStoreConflictError(f"条文主键与版本、条号的映射冲突（{chunk.kb_chunk_id}）")
            if existing[0] != expected:
                raise LawStoreConflictError(f"条文正文、摘要或来源冲突（{chunk.kb_chunk_id}）")
            return False

        # 传入摘要的十六进制表示，由 MySQL 解码成原始字节，避免连接字符集解释摘要字节。
        cursor.execute(
            """
            INSERT INTO law_articles (kb_chunk_id, document_id, article_no, `text`, text_sha256, source_url)
            VALUES (%(kb_chunk_id)s, %(document_id)s, %(article_no)s, %(text)s,
                    UNHEX(%(text_sha256_hex)s), %(source_url)s)
            """,
            {**expected, "text_sha256_hex": expected["text_sha256"].hex()},
        )
        return True
