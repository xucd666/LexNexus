from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING

from app.schemas.law_chunk import LawChunk

if TYPE_CHECKING:
    from pymilvus import DataType

_PRIMARY_KEY = "kb_chunk_id"
_VECTOR_FIELD = "embedding"
_DIMENSION = 512
_METRIC = "COSINE"
_INDEX_TYPE = "FLAT"
_MAX_TOP_K = 16_384
# FLOAT_VECTOR 用 float32 存储。有限的 Python float 仍可能超出该范围。
_FLOAT32_MAX = 3.4028234663852886e38
_DOCUMENT_ID_MAX_CHARS = 128
_FILTER_ESCAPES = str.maketrans({"\\": "\\\\", '"': r"\"", "\n": r"\n", "\r": r"\r", "\t": r"\t"})

# Milvus 按 UTF-8 字节计长。按每字符最多 4 字节预留 LawChunk 的标识和名称空间。
_VARCHAR_LIMITS = {
    _PRIMARY_KEY: 128 * 4,
    "document_id": _DOCUMENT_ID_MAX_CHARS * 4,
    "law_name": 256 * 4,
    "article_no": 64 * 4,
    "text": 65_535,
    "source_url": 4_096,
}
_OUTPUT_FIELDS = tuple(name for name in _VARCHAR_LIMITS if name != _PRIMARY_KEY)


class MilvusStore:
    """管理一个本地 Milvus Lite Collection，存取 LawChunk 原文及其向量。

    离线入库使用 create=True 显式创建，在线检索使用默认方式打开已有 Collection。
    实例持有一个同步客户端，调用方负责关闭。同一数据库路径由一个进程持有，
    对同一 Collection 的写入由调用方串行执行。
    """

    def __init__(self, db_path: str | Path, collection_name: str, *, create: bool = False) -> None:
        """绑定数据库路径和 Collection 名称，创建或检查索引后即可查询。

        db_path 须以 .db 结尾且父目录已存在。默认要求数据库和 Collection 均存在。
        路径不存在时抛出 FileNotFoundError，Collection 不存在时抛出 LookupError。
        参数或 Schema 不兼容时抛出 ValueError。显式创建遇到同名 Collection 时拒绝覆盖。
        """
        if not isinstance(create, bool):
            raise TypeError("create 须为布尔值")
        path = Path(db_path).expanduser().resolve()
        if path.suffix != ".db":
            raise ValueError("Milvus Lite 数据库路径须以 .db 结尾")
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("Collection 名称须为非空字符串")
        if not path.parent.is_dir():
            raise FileNotFoundError(f"数据库父目录不存在：{path.parent}")
        if not create and not path.exists():
            raise FileNotFoundError(f"数据库不存在：{path}")

        # 只有创建 Store 实例才需要 SDK，默认 Fake 流程可以独立导入。
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as exc:
            raise RuntimeError("MilvusStore 需要可用的 PyMilvus SDK 与 Milvus Lite 后端") from exc

        self._client = MilvusClient(uri=str(path))
        self._collection_name = collection_name
        self._closed = False
        try:
            exists = self._client.has_collection(collection_name=collection_name)
            if create:
                if exists:
                    raise ValueError(f"Collection 已存在：{collection_name}")
                self._create_collection(DataType)
            else:
                if not exists:
                    raise LookupError(f"Collection 不存在：{collection_name}")
                self._check_collection(DataType)
                self._client.load_collection(collection_name=collection_name)
        except Exception as exc:
            # 初始化失败也须关闭客户端。若清理失败，将详情附在原始异常上。
            try:
                self.close()
            except Exception as close_error:
                exc.add_note(f"关闭 Milvus 客户端失败：{close_error}")
            raise

    def upsert(self, chunks: Iterable[LawChunk], embeddings: Iterable[Iterable[Real]]) -> int:
        """按位置配对一批法律片段与向量，返回 SDK 确认写入的记录数。

        已有主键会更新整条记录。空批次返回 0。数量不匹配、主键重复或数据不合法时
        整批在调用 SDK 前被拒绝，抛出 TypeError 或 ValueError。SDK 写入失败时原样传播异常。
        """
        self._ensure_open()
        records = []
        seen = set()
        # strict=True 禁止两组输入长度不同时被 zip 截短，避免片段与向量漏配。
        for chunk, vector in zip(chunks, embeddings, strict=True):
            if not isinstance(chunk, LawChunk):
                raise TypeError("chunks 的每一项都须为 LawChunk")
            # LawChunk 可被修改。重新校验字段，再以 JSON 模式将 HttpUrl 转成字符串。
            validated = LawChunk.model_validate(chunk.model_dump(warnings=False))
            record = validated.model_dump(mode="json")
            for name, max_bytes in _VARCHAR_LIMITS.items():
                _check_utf8_length(name, record[name], max_bytes)
            if validated.kb_chunk_id in seen:
                raise ValueError(f"同一批次含重复 kb_chunk_id：{validated.kb_chunk_id}")
            seen.add(validated.kb_chunk_id)
            record[_VECTOR_FIELD] = _validate_vector(vector)
            records.append(record)

        if not records:
            return 0
        outcome = self._client.upsert(collection_name=self._collection_name, data=records)
        count = outcome.get("upsert_count")
        if type(count) is not int or count != len(records):
            raise RuntimeError(f"Milvus 报告的写入数量与预期 {len(records)} 条不符")
        return count

    def search(
        self, embedding: Iterable[Real], *, top_k: int = 3, document_id: str | None = None
    ) -> list[tuple[LawChunk, float]]:
        """检索一个 512 维向量，可限定特定文档版本。

        top_k 为 1 至 16384 的整数。返回原排序的 (LawChunk, COSINE 分数) 列表。
        输入不合法时抛出 TypeError 或 ValueError，返回结构异常时抛出 RuntimeError。
        SDK 查询失败时原样传播异常。只有成功且无命中时返回空列表。
        """
        self._ensure_open()
        if type(top_k) is not int or not 1 <= top_k <= _MAX_TOP_K:
            raise ValueError(f"top_k 须为 1 至 {_MAX_TOP_K} 的整数")
        query = _validate_vector(embedding)
        expression = ""
        if document_id is not None:
            if not isinstance(document_id, str):
                raise TypeError("document_id 须为字符串")
            document_id = document_id.strip()
            if not 1 <= len(document_id) <= _DOCUMENT_ID_MAX_CHARS:
                raise ValueError(f"document_id 须包含 1 至 {_DOCUMENT_ID_MAX_CHARS} 个字符")
            _check_utf8_length("document_id", document_id, _VARCHAR_LIMITS["document_id"])
            # Lite 3.2.2 的 Search 解码只读取 DSL 表达式。固定字段和比较符，
            # 按 Lite 词法器的规则一次性转义值，防止引号或反斜杠改变表达式结构。
            literal = document_id.translate(_FILTER_ESCAPES)
            expression = f'document_id == "{literal}"'

        batches = self._client.search(
            collection_name=self._collection_name,
            data=[query],
            anns_field=_VECTOR_FIELD,
            limit=top_k,
            filter=expression,
            search_params={"metric_type": _METRIC},
            output_fields=list(_OUTPUT_FIELDS),
        )
        # SDK 按查询向量分组。一组内的命中已经排好序，此处保留原顺序。
        if (
            not isinstance(batches, list)
            or len(batches) > 1
            or (batches and not isinstance(batches[0], list))
        ):
            raise RuntimeError("Milvus 返回了非预期的单向量查询结果")
        return [_read_hit(hit) for hit in batches[0]] if batches else []

    def close(self) -> None:
        """释放本实例的 SDK 客户端，可重复调用。SDK 关闭异常交由调用方处理。

        Lite 3.2.2 的嵌入服务由 SDK 按路径共享，其进程级生命周期独立于客户端。
        """
        if not self._closed:
            self._client.close()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MilvusStore 已关闭")

    def _create_collection(self, data_type: type[DataType]) -> None:
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        for name, max_bytes in _VARCHAR_LIMITS.items():
            schema.add_field(
                field_name=name,
                datatype=data_type.VARCHAR,
                max_length=max_bytes,
                is_primary=name == _PRIMARY_KEY,
            )
        schema.add_field(field_name=_VECTOR_FIELD, datatype=data_type.FLOAT_VECTOR, dim=_DIMENSION)
        # FLAT 对小规模语料做精确比较。创建时同时配置向量字段的 COSINE 索引。
        index = self._client.prepare_index_params()
        index.add_index(field_name=_VECTOR_FIELD, index_type=_INDEX_TYPE, metric_type=_METRIC)
        self._client.create_collection(
            collection_name=self._collection_name, schema=schema, index_params=index
        )

    def _check_collection(self, data_type: type[DataType]) -> None:
        schema = self._client.describe_collection(collection_name=self._collection_name)
        if schema.get("auto_id") is not False or schema.get("enable_dynamic_field") is not False:
            raise ValueError("Collection Schema 须关闭自动 ID 和动态字段")
        fields = {field["name"]: field for field in schema.get("fields", [])}
        if fields.keys() != (_VARCHAR_LIMITS.keys() | {_VECTOR_FIELD}):
            raise ValueError("Collection Schema 字段与 MilvusStore 不兼容")
        for name, field in fields.items():
            dtype, param, size = (
                (data_type.FLOAT_VECTOR, "dim", _DIMENSION)
                if name == _VECTOR_FIELD
                else (data_type.VARCHAR, "max_length", _VARCHAR_LIMITS[name])
            )
            if (
                field["type"] != dtype
                or field.get("params", {}).get(param) != size
                or field.get("is_primary", False) != (name == _PRIMARY_KEY)
                or field.get("nullable", False)
            ):
                raise ValueError(f"Collection Schema 字段不兼容：{name}，要求 {dtype.name}，{param}={size}")

        # Lite 的 DescribeIndex 使用字段名作为索引名。检查通过后才允许加载已有索引。
        index = self._client.describe_index(collection_name=self._collection_name, index_name=_VECTOR_FIELD)
        if (
            not index
            or index.get("field_name") != _VECTOR_FIELD
            or index.get("metric_type") != _METRIC
            or index.get("index_type") not in {_INDEX_TYPE, "BRUTE_FORCE"}
        ):
            raise ValueError("Collection 缺少兼容的 embedding FLAT/COSINE 索引")


def _check_utf8_length(name: str, value: str, limit: int) -> None:
    if len(value.encode("utf-8")) > limit:
        raise ValueError(f"{name} 超过 {limit} 个 UTF-8 字节")


def _validate_vector(vector: Iterable[Real]) -> list[float]:
    # 一维 NumPy 数组和普通列表都可迭代。转换为 Python float 方便 SDK 序列化。
    values = list(vector)
    if len(values) != _DIMENSION:
        raise ValueError(f"向量必须为 {_DIMENSION} 维，实际为 {len(values)} 维")
    normalized = []
    for value in values:
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError("向量分量须为实数")
        if not -_FLOAT32_MAX <= value <= _FLOAT32_MAX:
            raise ValueError("向量分量须有限且可表示为 float32")
        normalized.append(float(value))
    return normalized


def _read_hit(hit: dict) -> tuple[LawChunk, float]:
    try:
        # PyMilvus 在外层返回主键和 distance，entity 只包含请求的元数据。
        # 较早的响应若未提供主键字段名，SDK 使用 id。
        metadata = {name: hit["entity"][name] for name in _OUTPUT_FIELDS}
        metadata[_PRIMARY_KEY] = hit[_PRIMARY_KEY] if _PRIMARY_KEY in hit else hit["id"]
        chunk = LawChunk.model_validate(metadata)
        score = hit["distance"]
        if not isinstance(score, Real) or isinstance(score, bool) or not math.isfinite(score):
            raise ValueError("命中的 COSINE 分数须为有限数值")
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Milvus 命中无法转换为完整 LawChunk 和分数") from exc
    return chunk, float(score)
