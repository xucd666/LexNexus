from datetime import date
from hashlib import sha256
from unittest.mock import MagicMock, call
from uuid import uuid4

import pytest

pymysql = pytest.importorskip("pymysql", reason="MySQL 存储测试需要 mysql extra")

from app.schemas.law_chunk import LawChunk  # noqa: E402
from app.tools.mysql_law_store import (  # noqa: E402
    LawStoreConflictError,
    LawVersion,
    LawWriteResult,
    MySQLLawStore,
)


@pytest.fixture
def version():
    return LawVersion(
        document_id="fixture-law-v1",
        law_id="fixture-law",
        law_name="测试条例",
        version_basis="测试用版本材料",
        source_url="https://example.test/law",
        promulgation_date=date(2020, 5, 28),
        effective_date=date(2021, 1, 1),
    )


@pytest.fixture
def chunk(version):
    return LawChunk(
        kb_chunk_id="fixture-chunk-1",
        document_id=version.document_id,
        law_name=version.law_name,
        article_no="第一条",
        text="  第一款：保留两端空格。\r\n\t第二款：保留 ' 引号、50% 和换行。\n ",
        source_url="https://example.test/law#article-1",
    )


@pytest.fixture
def db():
    connection = MagicMock(name="connection")
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    return connection, cursor


def _version_row(version):
    return version.model_dump() | {"source_url": str(version.source_url)}


def _article_row(chunk):
    return chunk.model_dump(mode="json", exclude={"law_name"}) | {
        "text_sha256": sha256(chunk.text.encode("utf-8")).digest(),
    }


def test_connect_configures_a_dedicated_session_and_close_is_idempotent(monkeypatch):
    token = uuid4().hex
    monkeypatch.setenv("AI_SERVICE_MYSQL_PASSWORD", token)
    connection = MagicMock()
    connect = MagicMock(return_value=connection)
    monkeypatch.setattr(pymysql, "connect", connect)
    connection.open = True

    def disconnect():
        connection.open = False

    connection.close.side_effect = disconnect
    store = MySQLLawStore.connect(database="test_law_db", user="test_law_app")
    kwargs = connect.call_args.kwargs
    assert kwargs["password"] == token
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 3306
    assert kwargs["database"] == "test_law_db"
    assert kwargs["user"] == "test_law_app"
    assert kwargs["charset"] == "utf8mb4"
    assert kwargs["collation"] == "utf8mb4_0900_bin"
    assert kwargs["sql_mode"] == "TRADITIONAL"
    assert kwargs["cursorclass"] is pymysql.cursors.DictCursor
    assert kwargs["autocommit"] is True
    assert kwargs["init_command"] == "SET SESSION time_zone = '+00:00'"
    store.close()
    store.close()
    connection.close.assert_called_once_with()


def test_connect_requires_a_dedicated_account_and_an_environment_secret(monkeypatch):
    monkeypatch.delenv("AI_SERVICE_MYSQL_PASSWORD", raising=False)
    connect = MagicMock()
    monkeypatch.setattr(pymysql, "connect", connect)
    with pytest.raises(ValueError, match="应用账号"):
        MySQLLawStore.connect(database="test_law_db", user=" Root ")
    with pytest.raises(ValueError, match="AI_SERVICE_MYSQL_PASSWORD"):
        MySQLLawStore.connect(database="test_law_db", user="test_law_app")
    connect.assert_not_called()


def test_first_batch_preserves_original_text_and_binary_digest(db, version, chunk):
    connection, cursor = db
    result = MySQLLawStore(connection).write_batch(version, [chunk])
    assert result == LawWriteResult(version_inserted=True, articles_inserted=1, articles_skipped=0)
    assert connection.mock_calls[0] == call.begin()
    assert connection.mock_calls[-1] == call.commit()
    connection.rollback.assert_not_called()

    find_version, insert_version, find_article, insert_article = cursor.execute.call_args_list
    assert find_version.args[1] == (version.document_id,)
    assert "INSERT INTO law_versions" in insert_version.args[0]
    assert insert_version.args[1] == _version_row(version)
    assert "WHERE kb_chunk_id = %s OR (document_id = %s AND article_no = %s)" in find_article.args[0]
    assert find_article.args[1] == (chunk.kb_chunk_id, chunk.document_id, chunk.article_no)
    sql, params = insert_article.args
    assert "INSERT INTO law_articles" in sql
    assert "UNHEX(%(text_sha256_hex)s)" in sql
    assert chunk.text not in sql
    assert params["text"] == chunk.text
    assert bytes.fromhex(params["text_sha256_hex"]) == sha256(chunk.text.encode("utf-8")).digest()
    assert params["source_url"] == str(chunk.source_url)


def test_repeat_batch_skips_matching_version_and_article(db, version, chunk):
    connection, cursor = db
    store = MySQLLawStore(connection)
    store.write_batch(version, [chunk])
    cursor.execute.reset_mock()

    cursor.fetchone.return_value = _version_row(version)
    cursor.fetchall.return_value = [_article_row(chunk)]
    assert store.write_batch(version, [chunk]) == LawWriteResult(
        version_inserted=False, articles_inserted=0, articles_skipped=1
    )
    assert len(cursor.execute.call_args_list) == 2
    assert all(stmt.args[0].lstrip().startswith("SELECT") for stmt in cursor.execute.call_args_list)
    assert connection.commit.call_count == 2
    connection.rollback.assert_not_called()


@pytest.mark.parametrize(
    ("column", "different_value"),
    [
        ("law_id", "another-law"),
        ("law_name", "另一名称"),
        ("version_basis", "另一份版本依据"),
        ("source_url", "https://example.test/other-source"),
        ("promulgation_date", date(2020, 5, 29)),
        ("effective_date", date(2021, 1, 2)),
        ("repeal_date", date(2030, 1, 1)),
    ],
)
def test_existing_version_metadata_must_match(db, version, chunk, column, different_value):
    connection, cursor = db
    cursor.fetchone.return_value = _version_row(version) | {column: different_value}
    with pytest.raises(LawStoreConflictError, match="版本元数据冲突"):
        MySQLLawStore(connection).write_batch(version, [chunk])
    assert cursor.execute.call_count == 1
    connection.rollback.assert_called_once_with()
    connection.commit.assert_not_called()


@pytest.mark.parametrize(
    ("column", "different_value"),
    [
        ("text_sha256", bytes(32)),
        ("text", "不同正文（即便保留原摘要也应被拒绝）"),
        ("source_url", "https://example.test/wrong-source"),
    ],
)
def test_existing_article_content_must_match(db, version, chunk, column, different_value):
    connection, cursor = db
    cursor.fetchone.return_value = _version_row(version)
    cursor.fetchall.return_value = [_article_row(chunk) | {column: different_value}]

    with pytest.raises(LawStoreConflictError, match="正文、摘要或来源冲突"):
        MySQLLawStore(connection).write_batch(version, [chunk])
    assert cursor.execute.call_count == 2
    connection.rollback.assert_called_once_with()
    connection.commit.assert_not_called()


@pytest.mark.parametrize(
    "row_overrides",
    [
        [{"kb_chunk_id": "other-key"}],
        [{"document_id": "other-version"}],
        [{"article_no": "第二条"}],
        [{"article_no": "第二条"}, {"kb_chunk_id": "other-key"}],
    ],
)
def test_checks_primary_key_and_business_key_mappings(db, version, chunk, row_overrides):
    connection, cursor = db
    cursor.fetchone.return_value = _version_row(version)
    cursor.fetchall.return_value = [_article_row(chunk) | override for override in row_overrides]

    with pytest.raises(LawStoreConflictError, match="映射冲突"):
        MySQLLawStore(connection).write_batch(version, [chunk])
    connection.rollback.assert_called_once_with()
    connection.commit.assert_not_called()


def test_late_conflict_rolls_back_earlier_inserts(db, version, chunk):
    connection, cursor = db
    next_chunk = chunk.model_copy(update={"kb_chunk_id": "fixture-chunk-2", "article_no": "第二条"})
    cursor.fetchall.side_effect = [[], [_article_row(next_chunk) | {"text_sha256": bytes(32)}]]

    with pytest.raises(LawStoreConflictError, match="摘要"):
        MySQLLawStore(connection).write_batch(version, [chunk, next_chunk])
    statements = [stmt.args[0] for stmt in cursor.execute.call_args_list]
    assert sum("INSERT INTO" in statement for statement in statements) == 2
    assert connection.mock_calls[-1] == call.rollback()
    connection.commit.assert_not_called()


def test_rejects_a_chunk_from_another_version(db, version, chunk):
    connection, cursor = db
    foreign_chunk = chunk.model_copy(update={"document_id": "other-version"})

    with pytest.raises(LawStoreConflictError, match="与批次元数据不一致"):
        MySQLLawStore(connection).write_batch(version, [foreign_chunk])
    assert cursor.execute.call_count == 2
    connection.rollback.assert_called_once_with()
    connection.commit.assert_not_called()


def test_mysql_duplicate_key_is_a_conflict_after_rollback(db, version, chunk):
    connection, cursor = db
    duplicate = pymysql.IntegrityError(1062, "duplicate")
    cursor.execute.side_effect = [None, None, None, duplicate]

    with pytest.raises(LawStoreConflictError, match="重复唯一键") as error:
        MySQLLawStore(connection).write_batch(version, [chunk])
    assert error.value.__cause__ is duplicate
    connection.rollback.assert_called_once_with()
    connection.commit.assert_not_called()


def test_other_database_error_is_preserved(db, version, chunk):
    connection, cursor = db
    failure = pymysql.OperationalError(1205, "lock wait timeout")
    cursor.execute.side_effect = [None, None, None, failure]

    with pytest.raises(pymysql.OperationalError) as error:
        MySQLLawStore(connection).write_batch(version, [chunk])
    assert error.value is failure
    connection.rollback.assert_called_once_with()
    connection.commit.assert_not_called()


def test_rollback_failure_closes_connection_and_preserves_error(db, version, chunk):
    connection, cursor = db
    failure = pymysql.OperationalError(2013, "connection lost")
    cursor.execute.side_effect = failure
    connection.rollback.side_effect = pymysql.OperationalError(2006, "server gone")

    with pytest.raises(pymysql.OperationalError) as error:
        MySQLLawStore(connection).write_batch(version, [chunk])
    assert error.value is failure
    assert any("回滚失败" in note for note in error.value.__notes__)
    connection.close.assert_called_once_with()


def test_read_reconstructs_version_and_chunk_and_returns_none_when_missing(db, version, chunk):
    connection, cursor = db
    cursor.fetchone.side_effect = [_version_row(version), chunk.model_dump(mode="json"), None, None]
    store = MySQLLawStore(connection)

    assert store.get_version(version.document_id) == version
    assert store.get_article(chunk.kb_chunk_id) == chunk
    assert store.get_version("missing") is None
    assert store.get_article("missing") is None
    get_article = cursor.execute.call_args_list[1].args
    assert "JOIN law_versions AS v ON v.document_id = a.document_id" in get_article[0]
    assert get_article[1] == (chunk.kb_chunk_id,)
    connection.begin.assert_not_called()
