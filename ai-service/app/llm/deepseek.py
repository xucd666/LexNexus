"""DeepSeek 的 LangChain OpenAI-compatible 封装。"""

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from app.core.config import Settings

logger = logging.getLogger(__name__)


class DeepSeekConfigurationError(RuntimeError):
    """真实模型配置不完整。"""


class DeepSeekInvocationError(RuntimeError):
    """模型请求失败或响应空白。"""


def create_chat_model(settings: Settings) -> BaseChatModel:
    """按需创建 ChatOpenAI。"""
    api_key = settings.deepseek_api_key.get_secret_value().strip()
    base_url = settings.deepseek_base_url.strip()
    model = settings.deepseek_model.strip()
    if not api_key or not base_url or not model:
        raise DeepSeekConfigurationError("DeepSeek 需要配置 API key、base URL 和模型名")

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=settings.request_timeout_seconds,
        max_retries=1,
    )


async def invoke_chat(model: BaseChatModel, payload: list[BaseMessage]) -> AIMessage:
    """统一转换厂商异常，日志只记录异常类型。"""
    try:
        return await model.ainvoke(payload)
    except Exception as exc:
        logger.warning("DeepSeek invocation failed: %s", type(exc).__name__)
        raise DeepSeekInvocationError("DeepSeek 服务暂不可用") from exc
