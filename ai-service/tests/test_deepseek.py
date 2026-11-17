import unittest
from unittest.mock import patch

from app.core.config import Settings
from app.llm.deepseek import DeepSeekConfigurationError, create_chat_model


class DeepSeekProviderTest(unittest.TestCase):
    def test_requires_complete_configuration(self) -> None:
        settings = Settings(
            _env_file=None, DEEPSEEK_API_KEY="test-key", DEEPSEEK_BASE_URL=" ", DEEPSEEK_MODEL="test-model"
        )
        with self.assertRaises(DeepSeekConfigurationError):
            create_chat_model(settings)

    def test_constructs_langchain_model_without_network_request(self) -> None:
        settings = Settings(
            _env_file=None,
            DEEPSEEK_API_KEY="test-key",
            DEEPSEEK_BASE_URL="https://model.example.invalid",
            DEEPSEEK_MODEL="test-model",
        )
        with patch("langchain_openai.ChatOpenAI") as chat_model:
            create_chat_model(settings)

        chat_model.assert_called_once_with(
            api_key="test-key",
            base_url="https://model.example.invalid",
            model="test-model",
            timeout=settings.request_timeout_seconds,
            max_retries=1,
        )
