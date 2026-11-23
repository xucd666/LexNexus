import json
from datetime import date
from hashlib import sha256
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("pymysql", reason="MySQL 导入测试需要 mysql extra")

from app.tools import mysql_law_store
from app.tools.mysql_law_store import LawStoreConflictError, LawWriteResult, MySQLLawStore
from scripts import import_laws_mysql

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_STEM = "civil_code_2020_articles_577_"


def _arguments(*, last_article=594, source=None, metadata=None):
    return [
        "--source",
        str(source if source is not None else _FIXTURES / f"{_STEM}{last_article}.txt"),
        "--metadata",
        str(metadata if metadata is not None else _FIXTURES / f"{_STEM}{last_article}.json"),
        "--law-id",
        "cn-civil-code",
        "--promulgation-date",
        "2020-05-28",
        "--effective-date",
        "2021-01-01",
    ]


@pytest.fixture
def connect(monkeypatch):
    connect = MagicMock(side_effect=AssertionError("此测试不应连接数据库"))
    monkeypatch.setattr(MySQLLawStore, "connect", connect)
    return connect


def test_check_mode_uses_exact_source_bytes_and_never_connects(connect, monkeypatch, capsys):
    splitter = MagicMock(wraps=import_laws_mysql.split_law_articles)
    monkeypatch.setattr(import_laws_mysql, "split_law_articles", splitter)
    source = (_FIXTURES / f"{_STEM}594.txt").read_bytes()
    metadata = json.loads((_FIXTURES / f"{_STEM}594.json").read_bytes())

    assert import_laws_mysql.main(_arguments()) == 0

    splitter.assert_called_once_with(
        source.decode("utf-8"),
        document_id=metadata["document_id"],
        law_name=metadata["law_name"],
        source_url=metadata["source_url"],
    )
    out, err = capsys.readouterr()
    assert "cn-civil-code-2020-05-28，待处理 18 条" in out
    assert sha256(source).hexdigest() in out
    assert "未连接 MySQL" in out
    assert err == ""
    connect.assert_not_called()


@pytest.mark.parametrize("bad_date", ["2020-02-30", "20200528", "2020-W22-4"])
def test_invalid_or_noncanonical_dates_are_rejected(connect, capsys, bad_date):
    args = _arguments()
    args[args.index("--promulgation-date") + 1] = bad_date

    with pytest.raises(SystemExit) as error:
        import_laws_mysql.main([*args, "--apply"])

    assert error.value.code == 2
    assert "YYYY-MM-DD" in capsys.readouterr().err
    connect.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [("version_basis", None), ("document_id", " "), ("source_url", "ftp://example.test/law")],
)
def test_invalid_metadata_is_rejected_before_connecting(connect, tmp_path, capsys, field, value):
    content = json.loads((_FIXTURES / f"{_STEM}594.json").read_bytes())
    if value is None:
        del content[field]
    else:
        content[field] = value
    metadata = tmp_path / "invalid.json"
    metadata.write_text(json.dumps(content), encoding="utf-8")

    assert import_laws_mysql.main([*_arguments(metadata=metadata), "--apply"]) == 1

    assert f"字段 {field}" in capsys.readouterr().err
    connect.assert_not_called()


def test_metadata_must_be_an_object(connect, tmp_path, capsys):
    metadata = tmp_path / "invalid.json"
    metadata.write_text("[]", encoding="utf-8")

    assert import_laws_mysql.main(_arguments(metadata=metadata)) == 1

    assert "JSON 对象" in capsys.readouterr().err
    connect.assert_not_called()


def test_missing_source_is_rejected(connect, tmp_path, capsys):
    assert import_laws_mysql.main([*_arguments(source=tmp_path / "missing.txt"), "--apply"]) == 1
    assert "无法加载" in capsys.readouterr().err
    connect.assert_not_called()


def test_duplicate_chunk_ids_are_rejected_before_connecting(connect, monkeypatch, capsys):
    splitter = import_laws_mysql.split_law_articles

    def split_with_duplicate_id(*args, **kwargs):
        chunks = splitter(*args, **kwargs)
        chunks[1].kb_chunk_id = chunks[0].kb_chunk_id
        return chunks

    monkeypatch.setattr(import_laws_mysql, "split_law_articles", split_with_duplicate_id)
    assert import_laws_mysql.main([*_arguments(), "--apply"]) == 1
    assert "重复的 kb_chunk_id" in capsys.readouterr().err
    connect.assert_not_called()


def test_apply_must_be_spelled_out_and_unknown_password_argument_is_not_echoed(connect, capsys):
    secret = uuid4().hex
    with pytest.raises(SystemExit) as error:
        import_laws_mysql.main([*_arguments(), "--app", "--password", secret])
    assert error.value.code == 2
    assert secret not in capsys.readouterr().err
    connect.assert_not_called()


def test_apply_passes_explicit_version_metadata_to_store(connect, capsys):
    store = MagicMock()
    connect.side_effect = None
    connect.return_value = store
    store.write_batch.return_value = LawWriteResult(True, 18, 0)
    args = [*_arguments(), "--port", "3310", "--repeal-date", "2030-01-01", "--apply"]

    assert import_laws_mysql.main(args) == 0

    connect.assert_called_once_with(
        database="lexnexus_law_dev", user="lexnexus_app", host="127.0.0.1", port=3310
    )
    version, chunks = store.write_batch.call_args.args
    assert version.law_id == "cn-civil-code"
    assert version.document_id == "cn-civil-code-2020-05-28"
    assert version.law_name == "中华人民共和国民法典"
    assert version.version_basis == json.loads((_FIXTURES / f"{_STEM}594.json").read_bytes())["version_basis"]
    assert version.promulgation_date == date(2020, 5, 28)
    assert version.effective_date == date(2021, 1, 1)
    assert version.repeal_date == date(2030, 1, 1)
    assert len(chunks) == 18
    assert chunks[7].text.endswith("\n")
    assert chunks[0].article_no == "第五百七十七条"
    assert chunks[-1].article_no == "第五百九十四条"
    assert all(chunk.document_id == version.document_id for chunk in chunks)
    assert all(chunk.source_url == version.source_url for chunk in chunks)
    assert "条文新增 18 条，跳过 0 条" in capsys.readouterr().out
    store.close.assert_called_once_with()


class _MemoryDatabase:
    """模拟 Store 的字典游标和事务，保留提交的状态供下次 CLI 调用读取。"""

    def __init__(self):
        self.versions = {}
        self.articles = {}
        self.connections = []

    def connect(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        self.connections.append(connection)
        pending_versions = self.versions.copy()
        pending_articles = self.articles.copy()

        def execute(sql, params):
            statement = " ".join(sql.split())
            if statement.startswith("SELECT document_id"):
                cursor.fetchone.return_value = pending_versions.get(params[0])
            elif statement.startswith("INSERT INTO law_versions"):
                pending_versions[params["document_id"]] = dict(params)
            elif statement.startswith("SELECT kb_chunk_id"):
                chunk_id, document_id, article_no = params
                cursor.fetchall.return_value = [
                    row
                    for row in pending_articles.values()
                    if row["kb_chunk_id"] == chunk_id
                    or (row["document_id"], row["article_no"]) == (document_id, article_no)
                ]
            elif statement.startswith("INSERT INTO law_articles"):
                bound = dict(params)
                bound["text_sha256"] = bytes.fromhex(bound.pop("text_sha256_hex"))
                pending_articles[params["kb_chunk_id"]] = bound
            else:
                pytest.fail("Store 执行了替身不支持的 SQL")

        def commit():
            self.versions = pending_versions
            self.articles = pending_articles

        cursor.execute.side_effect = execute
        connection.commit.side_effect = commit
        return MySQLLawStore(connection)


def test_real_store_against_fake_mysql_imports_eight_then_adds_ten_then_skips_all(connect, capsys):
    db = _MemoryDatabase()
    connect.side_effect = lambda **_: db.connect()
    batches = [
        (584, "版本新增，条文新增 8 条，跳过 0 条", 8),
        (594, "版本已存在，跳过，条文新增 10 条，跳过 8 条", 18),
        (594, "版本已存在，跳过，条文新增 0 条，跳过 18 条", 18),
    ]
    old_articles = {}
    for last_article, expected_summary, total in batches:
        assert import_laws_mysql.main([*_arguments(last_article=last_article), "--apply"]) == 0
        assert expected_summary in capsys.readouterr().out
        assert len(db.versions) == 1
        assert len(db.articles) == total
        assert all(db.articles[key] == row for key, row in old_articles.items())
        old_articles = db.articles.copy()
        assert next(iter(db.versions.values()))["repeal_date"] is None

    for connection in db.connections:
        connection.commit.assert_called_once_with()
        connection.rollback.assert_not_called()
        connection.close.assert_called_once_with()


def test_store_conflict_rolls_back_and_cli_returns_failure(connect, tmp_path, capsys):
    db = _MemoryDatabase()
    connect.side_effect = lambda **_: db.connect()
    assert import_laws_mysql.main([*_arguments(last_article=584), "--apply"]) == 0
    capsys.readouterr()
    original_rows = db.articles.copy()
    # 向临时副本追加正文，条号和片段 ID 保持一致，触发现有 Store 的正文冲突。
    source = tmp_path / "changed.txt"
    source.write_bytes((_FIXTURES / f"{_STEM}584.txt").read_bytes() + "附加文字\n".encode())

    assert import_laws_mysql.main([*_arguments(source=source), "--apply"]) == 1

    out, err = capsys.readouterr()
    assert "导入冲突，批次未提交" in err
    assert "导入完成" not in out
    db.connections[-1].rollback.assert_called_once_with()
    db.connections[-1].commit.assert_not_called()
    db.connections[-1].close.assert_called_once_with()
    assert db.articles == original_rows


@pytest.mark.parametrize("stage", ["connect", "write", "conflict"])
def test_driver_and_conflict_errors_do_not_expose_connection_details(connect, monkeypatch, capsys, stage):
    secret = uuid4().hex
    monkeypatch.setenv("AI_SERVICE_MYSQL_PASSWORD", secret)
    sensitive_message = f"mysql://user:{secret}@127.0.0.1"
    if stage == "connect":
        connect.side_effect = RuntimeError(sensitive_message)
    else:
        error = LawStoreConflictError if stage == "conflict" else RuntimeError
        connect.side_effect = None
        connect.return_value.write_batch.side_effect = error(sensitive_message)

    assert import_laws_mysql.main([*_arguments(), "--apply"]) == 1

    out, err = capsys.readouterr()
    assert secret not in out + err
    assert "mysql://" not in out + err
    assert err
    if stage != "connect":
        connect.return_value.close.assert_called_once_with()


def test_missing_password_fails_before_driver_opens_connection(monkeypatch, capsys):
    monkeypatch.delenv("AI_SERVICE_MYSQL_PASSWORD", raising=False)
    driver_connect = MagicMock(side_effect=AssertionError("不应连接真实 MySQL"))
    monkeypatch.setattr(mysql_law_store.pymysql, "connect", driver_connect)

    assert import_laws_mysql.main([*_arguments(), "--apply"]) == 1

    assert "AI_SERVICE_MYSQL_PASSWORD" in capsys.readouterr().err
    driver_connect.assert_not_called()
