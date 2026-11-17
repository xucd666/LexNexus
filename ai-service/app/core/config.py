"""LexNexus 服务配置。"""

from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="AI_SERVICE_", extra="ignore"
    )

    mode: Literal["fake", "live"] = "fake"
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
