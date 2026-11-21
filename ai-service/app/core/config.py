"""LexNexus 服务配置。"""

from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="AI_SERVICE_", extra="ignore"
    )

    mode: Literal["fake", "live"] = "fake"
    # 检索后端与 LLM 模式独立，向量参数仅在服务自行装配 vector 后端时必填。
    law_search_backend: Literal["fake", "vector"] = "fake"
    bge_model_path: str = ""
    milvus_db_path: str = ""
    milvus_collection_name: str = ""
    law_search_top_k: int = Field(default=3, ge=1, le=16_384)
    # 向量检索使用独立验收的原文和版本元数据作为引用校验基准。
    citation_reference_source_path: str = ""
    citation_reference_metadata_path: str = ""
    request_timeout_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    deepseek_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "AI_SERVICE_DEEPSEEK_API_KEY"),
    )
    deepseek_base_url: str = Field(
        default="", validation_alias=AliasChoices("DEEPSEEK_BASE_URL", "AI_SERVICE_DEEPSEEK_BASE_URL")
    )
    deepseek_model: str = Field(
        default="", validation_alias=AliasChoices("DEEPSEEK_MODEL", "AI_SERVICE_DEEPSEEK_MODEL")
    )
