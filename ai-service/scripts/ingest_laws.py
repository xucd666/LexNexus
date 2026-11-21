"""离线为一份法律文档建库并核查。产物留待验收，在线检索可继续使用已有索引。"""

import argparse
import json
import sys
from collections.abc import Sequence
from contextlib import closing
from hashlib import sha256
from pathlib import Path

from app.schemas.law_chunk import LawChunk
from app.tools.bge_embedding import BgeEmbedding
from app.tools.law_splitter import split_law_articles
from app.tools.milvus_store import MilvusStore

_MODEL_ID = "BAAI/bge-small-zh-v1.5"
_METADATA_FIELDS = ("document_id", "law_name", "source_url", "version_basis")


def ingest_laws(
    *,
    source: Path,
    metadata: Path,
    model_path: Path,
    db_path: Path,
    collection: str,
    manifest: Path,
    embedding: BgeEmbedding | None = None,
) -> dict[str, str | int]:
    """读取本地 UTF-8 原文与 JSON 元数据，返回写入后已核查的清单内容。

    数据库和清单必须是不同的新路径，父目录须已存在。模型路径须指向本地 BGE 快照。
    调用方可传入已加载的同一快照实例，在同进程的离线查询中复用。
    输入、编码、存储和核查异常交给调用方处理。入库产物供后续独立验收后使用。
    """
    db_path = _new_target(db_path)
    manifest = _new_target(manifest)
    if db_path.suffix != ".db":
        raise ValueError("数据库路径须以 .db 结尾")
    if db_path == manifest:
        raise ValueError("数据库和清单须使用不同路径")
    if not collection.strip():
        raise ValueError("Collection 名称不能为空")

    # 同一份原文同时用于计算文件摘要和切分。按字节读取可保留 CRLF 等原始换行。
    source_bytes = source.expanduser().read_bytes()
    source_metadata = _read_metadata(metadata.expanduser())
    chunks = split_law_articles(
        source_bytes.decode("utf-8"),
        document_id=source_metadata["document_id"],
        law_name=source_metadata["law_name"],
        source_url=source_metadata["source_url"],
    )
    if not chunks:
        raise ValueError("原文未切分出法律片段")

    # 先完成全部编码，避免模型失败时留下空索引。片段和向量按同一顺序传给 Store。
    encoder = embedding if embedding is not None else BgeEmbedding(model_path.expanduser())
    embeddings = encoder.encode_documents([chunk.text for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise ValueError(f"片段数量 {len(chunks)} 与向量数量 {len(embeddings)} 不一致")

    manifest_data = {
        "model_id": _MODEL_ID,
        "collection": collection,
        "embedding_dimension": len(embeddings[0]),
        "document_id": chunks[0].document_id,
        "law_name": chunks[0].law_name,
        "version_basis": source_metadata["version_basis"],
        "source_url": str(chunks[0].source_url),
        "source_sha256": sha256(source_bytes).hexdigest(),
        "chunk_count": len(chunks),
    }

    # mkdir 独占申请目标路径，覆盖编码期间其他进程抢先创建目录的情况。
    db_path.mkdir()
    with closing(MilvusStore(db_path, collection, create=True)) as store:
        count = store.upsert(chunks, embeddings)
        if count != len(chunks):
            raise RuntimeError(f"预期写入 {len(chunks)} 条，实际写入 {count} 条")
        _verify_written_chunks(db_path, collection, chunks)

    # 清单在写入、核查及客户端关闭成功后生成。x 模式拒绝覆盖已经存在的文件。
    with manifest.open("x", encoding="utf-8") as output:
        output.write(json.dumps(manifest_data, ensure_ascii=False, indent=2) + "\n")
    return manifest_data


def _new_target(path: Path) -> Path:
    path = path.expanduser()
    if path.exists() or path.is_symlink() or path.is_junction():
        raise FileExistsError(f"目标已存在：{path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"目标父目录不存在：{path.parent}")
    return path.parent.resolve() / path.name


def _read_metadata(path: Path) -> dict[str, str]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("法律元数据须为 JSON 对象")
    values = {}
    for field in _METADATA_FIELDS:
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"法律元数据字段 {field} 须为非空字符串")
        values[field] = value
    return values


def _verify_written_chunks(db_path: Path, collection: str, chunks: list[LawChunk]) -> None:
    # Store 的在线检索按向量排序。离线完整性检查用 SDK 的 get 按主键取回指定记录。
    from pymilvus import MilvusClient

    fields = list(LawChunk.model_fields)
    expected = {chunk.kb_chunk_id: chunk.model_dump(mode="json") for chunk in chunks}
    with closing(MilvusClient(uri=str(db_path))) as client:
        rows = client.get(collection_name=collection, ids=list(expected), output_fields=fields)
    # 按主键比较，不依赖 SDK 返回记录的顺序，也不重新规范化读回的原文。
    actual = {row["kb_chunk_id"]: {field: row[field] for field in fields} for row in rows}
    if len(rows) != len(chunks) or actual != expected:
        raise RuntimeError("入库读回校验失败，主键、原文或元数据不一致")


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行参数，成功返回 0，入库失败返回 1。"""
    parser = argparse.ArgumentParser(description="将一份法律文档离线写入新的 Milvus Lite 数据库")
    parser.add_argument("--source", type=Path, required=True, help="UTF-8 法律原文文件")
    parser.add_argument("--metadata", type=Path, required=True, help="法律版本与来源的 JSON 文件")
    parser.add_argument("--model-path", type=Path, required=True, help=f"本地 {_MODEL_ID} 快照目录")
    parser.add_argument("--db-path", type=Path, required=True, help="待创建的 .db 目录")
    parser.add_argument("--collection", required=True, help="待创建的 Collection 名称")
    parser.add_argument("--manifest", type=Path, required=True, help="待生成的 JSON 索引清单路径")
    args = parser.parse_args(argv)

    try:
        result = ingest_laws(
            source=args.source,
            metadata=args.metadata,
            model_path=args.model_path,
            db_path=args.db_path,
            collection=args.collection,
            manifest=args.manifest,
        )
    except Exception as exc:
        print(f"离线入库失败（{type(exc).__name__}）：{exc}", file=sys.stderr)
        return 1

    print(
        f"离线入库完成，写入 {result['chunk_count']} 条，Collection {result['collection']}，"
        f"向量维度 {result['embedding_dimension']}，文档版本 {result['document_id']}。"
    )
    print(f"数据库：{args.db_path}\n索引清单：{args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
