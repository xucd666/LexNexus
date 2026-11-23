"""离线检查一份法律文档，并在显式授权后写入本机 MySQL 开发库。"""

import argparse
import json
import sys
from contextlib import closing
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from app.tools.law_splitter import split_law_articles
from pydantic import ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

_METADATA_FIELDS = ("document_id", "law_name", "source_url", "version_basis")
_DATABASE = "lexnexus_law_dev"
_USER = "lexnexus_app"
_HOST = "127.0.0.1"


def _parse_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError("日期须为有效的 YYYY-MM-DD") from None
    # fromisoformat 也接受紧凑日期和 ISO 周日期，这里要求使用明确的日历日期格式。
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("日期须为有效的 YYYY-MM-DD")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    """先完成输入检查，按需提交一个版本的条文批次。失败时返回非零退出码。"""
    parser = argparse.ArgumentParser(description="检查法律文档或导入本机 MySQL 开发库", allow_abbrev=False)
    parser.add_argument("--source", type=Path, required=True, help="UTF-8 法律原文 TXT")
    parser.add_argument("--metadata", type=Path, required=True, help="法律版本与来源的 JSON 文件")
    parser.add_argument("--law-id", required=True, help="同一部法律各版本共享的标识")
    parser.add_argument("--promulgation-date", type=_parse_date, required=True, help="公布日期 YYYY-MM-DD")
    parser.add_argument("--effective-date", type=_parse_date, required=True, help="施行日期 YYYY-MM-DD")
    parser.add_argument("--repeal-date", type=_parse_date, help="失效日期 YYYY-MM-DD，省略时不登记")
    parser.add_argument("--port", type=int, default=3306, help="本机 MySQL 端口，默认 3306")
    parser.add_argument("--apply", action="store_true", help="连接 MySQL 并提交，省略时只检查本地输入")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        parser.error("存在未识别的参数，请使用 --help 查看支持的选项")
    if not 1 <= args.port <= 65535:
        parser.error("MySQL 端口须在 1 至 65535 之间")

    try:
        from app.tools.mysql_law_store import LawStoreConflictError, LawVersion, MySQLLawStore
    except ModuleNotFoundError as exc:
        if exc.name != "pymysql":
            raise
        print("缺少 PyMySQL，请先同步项目的 mysql 可选依赖。", file=sys.stderr)
        return 1

    try:
        raw_metadata = json.loads(args.metadata.expanduser().read_text(encoding="utf-8"))
        # 从同一份字节读取计算摘要和切分，保留正文中的 LF、CRLF 及原始空格。
        source_bytes = args.source.expanduser().read_bytes()
        source_text = source_bytes.decode("utf-8")
    except OSError, UnicodeError, json.JSONDecodeError:
        print("无法加载法律原文或元数据，请检查文件路径、UTF-8 编码和 JSON 格式。", file=sys.stderr)
        return 1

    if not isinstance(raw_metadata, dict):
        print("法律元数据须为 JSON 对象。", file=sys.stderr)
        return 1

    metadata = {field: raw_metadata.get(field) for field in _METADATA_FIELDS}
    # 版本记录和切分结果需要同一套文档标识、法律名称。与 LawChunk 一样清理两端空白。
    for field in ("document_id", "law_name"):
        if isinstance(metadata[field], str):
            metadata[field] = metadata[field].strip()

    try:
        version = LawVersion.model_validate(
            {
                **metadata,
                "law_id": args.law_id,
                "promulgation_date": args.promulgation_date,
                "effective_date": args.effective_date,
                "repeal_date": args.repeal_date,
            }
        )
        chunks = split_law_articles(
            source_text,
            document_id=version.document_id,
            law_name=version.law_name,
            source_url=str(version.source_url),
        )
    except ValidationError as exc:
        # Pydantic 异常的默认文本包含输入值。CLI 只报告失败字段。
        fields = "、".join(sorted({error["loc"][0] for error in exc.errors(include_input=False)}))
        print(f"法律数据校验失败，字段 {fields}。", file=sys.stderr)
        return 1
    except ValueError:
        print("法条切分失败，请核对行首条号、条号唯一性和正文。", file=sys.stderr)
        return 1

    if not chunks:
        print("原文未切分出法律片段。", file=sys.stderr)
        return 1
    if len({chunk.kb_chunk_id for chunk in chunks}) != len(chunks):
        print("待导入片段包含重复的 kb_chunk_id。", file=sys.stderr)
        return 1

    print(f"文档版本 {version.document_id}，待处理 {len(chunks)} 条。")
    print(f"原文文件 SHA-256 {sha256(source_bytes).hexdigest()}")
    if not args.apply:
        print("检查完成，未连接 MySQL。")
        return 0

    # 固定本机开发库和应用账号。Store 从环境变量取密码，CLI 不接受密码参数。
    try:
        store = MySQLLawStore.connect(database=_DATABASE, user=_USER, host=_HOST, port=args.port)
    except Exception:
        print(
            "MySQL 连接失败，请检查本机开发库、应用账号和 AI_SERVICE_MYSQL_PASSWORD 配置。",
            file=sys.stderr,
        )
        return 1

    try:
        with closing(store):
            result = store.write_batch(version, chunks)
    except LawStoreConflictError:
        print("导入冲突，批次未提交。请核对已有版本元数据、条文标识和正文。", file=sys.stderr)
        return 1
    except Exception:
        # 驱动异常可能含连接信息或 SQL 值。提交期间断连时，需核对实际落库状态。
        print("MySQL 导入或连接关闭失败，请核对数据库状态后再重试。", file=sys.stderr)
        return 1

    status = "新增" if result.version_inserted else "已存在，跳过"
    print(
        f"导入完成，版本{status}，条文新增 {result.articles_inserted} 条，跳过 {result.articles_skipped} 条。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
