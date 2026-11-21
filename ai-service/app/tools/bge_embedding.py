# 将注解保留为字符串，运行时无需解析仅供静态检查的 NumPy 类型。
from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

# TYPE_CHECKING 在运行时为 False，导入本模块时无需加载 NumPy。
if TYPE_CHECKING:
    from numpy import float32
    from numpy.typing import NDArray

DEFAULT_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
_EMBEDDING_DIMENSION = 512
_NORM_TOLERANCE = 1e-5


class BgeEmbedding:
    """用本地 BAAI/bge-small-zh-v1.5 快照在 CPU 上离线生成文本向量。

    查询可以带检索指令，文档直接使用原文。输入是单条非空白字符串或一批此类字符串，
    输出是 512 维查询向量或按输入顺序排列的文档矩阵，均为 L2 归一化的 float32 NumPy 数组。
    超出 Token 上限的输入会在模型编码前被拒绝。
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ) -> None:
        """从本地快照目录加载模型，后续编码复用此实例中的模型。

        query_instruction 是仅限关键字传入的查询前缀，空字符串关闭前缀。
        """
        # 先验证路径并解析为绝对目录，便于在加载模型前定位快照配置错误。
        if isinstance(model_path, str) and not model_path.strip():
            raise ValueError("BGE model path must not be blank")
        if not isinstance(query_instruction, str):
            raise TypeError("BGE query instruction must be a string")
        snapshot = Path(model_path).expanduser().resolve()
        if not snapshot.exists():
            raise FileNotFoundError(f"BGE model snapshot does not exist: {snapshot}")
        if not snapshot.is_dir():
            raise NotADirectoryError(f"BGE model snapshot must be a directory: {snapshot}")

        # 只在创建实例时导入重型依赖，默认 Fake 流程仍可在未安装 RAG 依赖时运行。
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("BGE requires sentence-transformers and its runtime dependencies") from exc

        # 固定使用 CPU、只读取本地文件，并禁用模型仓库提供的自定义代码。
        try:
            self._model = SentenceTransformer(
                str(snapshot), device="cpu", local_files_only=True, trust_remote_code=False
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load the local BGE model: {snapshot}") from exc

        self._query_instruction = query_instruction
        # 两处上限都须是正整数，严格检查类型可排除布尔值。取较小值以满足双方限制。
        model_limit = self._model.max_seq_length
        tokenizer_limit = self._model.tokenizer.model_max_length
        if any(type(limit) is not int or limit <= 0 for limit in (model_limit, tokenizer_limit)):
            raise RuntimeError("BGE model and tokenizer must report positive token limits")
        self._token_limit = min(model_limit, tokenizer_limit)

    def encode_query(self, text: str) -> NDArray[float32]:
        """返回形状为 (512,) 的 float32 查询向量。

        查询指令让模型按检索任务编码问题，便于匹配相关文档。
        """
        self._validate_text(text)
        # 用单元素列表复用批量编码，再取第 0 行，将 (1, 512) 矩阵变为 (512,) 向量。
        return self._encode([self._query_instruction + text])[0]

    def encode_documents(self, texts: Sequence[str]) -> NDArray[float32]:
        """接收列表、元组等非空字符串序列，返回 (N, 512) 的 float32 矩阵。

        每行对应同位置的文档原文，文档不加查询指令。
        """
        # str、bytes 和 bytearray 本身也是序列，排除它们以避免逐字符或逐字节处理单个值。
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes, bytearray)):
            raise TypeError("BGE documents must be a sequence of strings")
        # 统一为有序列表，供批量校验和编码共用。
        documents = list(texts)
        if not documents:
            raise ValueError("BGE document batch must not be empty")
        for text in documents:
            self._validate_text(text)
        return self._encode(documents)

    @staticmethod
    def _validate_text(text: str) -> None:
        """校验单条文本，无需访问实例状态，因此使用静态方法。

        非字符串抛出 TypeError。空字符串或纯空白字符串类型正确，但值无效，抛出 ValueError。
        """
        if not isinstance(text, str):
            raise TypeError("BGE input text must be a string")
        if not text.strip():
            raise ValueError("BGE input text must not be blank")

    def _encode(self, texts: list[str]) -> NDArray[float32]:
        """对已校验的文本列表进行批量编码。

        raise ... from exc 将底层异常设为显式原因，便于从阶段错误追溯到具体故障。
        """
        # 分词与长度校验。add_special_tokens 计入特殊标记，padding=False 保留各条原长，
        # truncation=False 保证超限输入仍按完整内容计数。
        try:
            token_ids = self._model.tokenizer(
                texts, add_special_tokens=True, padding=False, truncation=False
            )["input_ids"]
        except Exception as exc:
            raise RuntimeError("BGE tokenization failed") from exc
        if len(token_ids) != len(texts):
            raise RuntimeError("BGE tokenizer returned an unexpected batch size")
        # 在推理前拒绝超限输入，避免模型自动截断并丢失文本内容。
        for index, tokens in enumerate(token_ids, start=1):
            if len(tokens) > self._token_limit:
                raise ValueError(
                    f"BGE input {index} has {len(tokens)} tokens, exceeding the {self._token_limit} token limit"
                )

        # 模型推理。空 prompt 覆盖模型默认提示，确保推理收到的文本与分词计数时一致。
        # convert_to_numpy 指定输出容器，precision 保持 float32，normalize_embeddings 使每行范数为 1。
        try:
            vectors = self._model.encode(
                texts,
                device="cpu",
                prompt="",
                convert_to_numpy=True,
                precision="float32",
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise RuntimeError("BGE encoding failed") from exc

        # 输出校验。类型注解不会校验返回值，需确认每条输入都有约定维度的 float32 向量。
        expected_shape = (len(texts), _EMBEDDING_DIMENSION)
        shape = getattr(vectors, "shape", None)
        if shape != expected_shape:
            raise RuntimeError(f"BGE embedding shape must be {expected_shape}, got {shape}")
        if str(vectors.dtype) != "float32":
            raise RuntimeError(f"BGE embeddings must be float32, got {vectors.dtype}")
        for index, vector in enumerate(vectors, start=1):
            # *vector 展开所有分量，hypot 计算欧氏长度。拒绝 NaN 和 Inf，避免相似度结果无效。
            norm = math.hypot(*vector)
            if not math.isfinite(norm):
                raise RuntimeError(f"BGE embedding {index} contains NaN or Inf")
            # float32 舍入会产生微小偏差，按绝对容差判断 L2 范数是否接近 1。
            if not math.isclose(norm, 1.0, rel_tol=0, abs_tol=_NORM_TOLERANCE):
                raise RuntimeError(f"BGE embedding {index} is not L2-normalized")
        return vectors
