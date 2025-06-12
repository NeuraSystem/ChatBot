import unittest
import os
from unittest.mock import patch, MagicMock, call # Import call for checking call arguments

# Import classes to be mocked
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI # Temporarily removed

# Import items to be tested
from src.services import ServiceContainer
from src.config import GlobalConfig

# Helper to temporarily set environment variables
class TempEnvVars:
    def __init__(self, new_vars):
        self.new_vars = new_vars
        self.old_vars = {}

    def __enter__(self):
        for key, value in self.new_vars.items():
            self.old_vars[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)  # Use pop for proper unsetting
            else:
                os.environ[key] = value
        return self

    def __exit__(self, type, value, traceback):
        for key, old_value in self.old_vars.items(): # Iterate over old_vars to restore correctly
            if old_value is None:
                os.environ.pop(key, None) # If original was None (not set), ensure it's unset
            else:
                os.environ[key] = old_value # Restore original value


class TestLLMConfiguration(unittest.TestCase):
    """
    Test suite for flexible LLM configuration based on environment variables.
    """

    @patch('langchain_anthropic.ChatAnthropic')
    def test_anthropic_provider(self, mock_chat_anthropic_class):
        """Test configuration for Anthropic provider."""
        env_vars = {
            "LLM_PROVIDER": "anthropic",
            "LLM_MODEL_NAME": "claude-test",
            "ANTHROPIC_API_KEY": "fake_anthropic_key",
            "RAG_ENABLED": "false" # Simplify by disabling RAG for these tests
        }
        with TempEnvVars(env_vars):
            mock_chat_anthropic_instance = mock_chat_anthropic_class.return_value
            # Mock methods that might be called during LangGraphService setup if not fully mocked out
            mock_chat_anthropic_instance.get_num_tokens = MagicMock(return_value=10)

            sc = ServiceContainer()
            self.assertIsInstance(sc.langgraph_service.llm, MagicMock) # It should be the instance of the mocked class
            mock_chat_anthropic_class.assert_called_once()
            # Check specific args. model_name for community, model for new langchain_anthropic
            called_args = mock_chat_anthropic_class.call_args
            self.assertEqual(called_args.kwargs['model'], "claude-test")
            self.assertEqual(str(called_args.kwargs['anthropic_api_key']), "fake_anthropic_key")

    @patch('langchain_openai.ChatOpenAI')
    def test_openai_provider(self, mock_chat_openai_class):
        """Test configuration for OpenAI provider."""
        env_vars = {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL_NAME": "gpt-test",
            "OPENAI_API_KEY": "fake_openai_key",
            "RAG_ENABLED": "false"
        }
        with TempEnvVars(env_vars):
            mock_chat_openai_instance = mock_chat_openai_class.return_value
            mock_chat_openai_instance.get_num_tokens = MagicMock(return_value=10)

            sc = ServiceContainer()
            self.assertIsInstance(sc.langgraph_service.llm, MagicMock)
            # For ChatOpenAI, model_name is a common param, but it might just be 'model'
            mock_chat_openai_class.assert_called_once_with(
                model_name="gpt-test", # or model="gpt-test"
                openai_api_key="fake_openai_key",
                max_tokens=512 # from LangGraphService default
            )

    # @patch('langchain_google_genai.ChatGoogleGenerativeAI') # Skipping Gemini test due to dependency conflict
    # def test_gemini_provider(self, mock_chat_google_genai_class):
    #     """Test configuration for Gemini provider."""
    #     env_vars = {
    #         "LLM_PROVIDER": "gemini",
    #         "LLM_MODEL_NAME": "gemini-test",
    #         "GEMINI_API_KEY": "fake_gemini_key",
    #         "RAG_ENABLED": "false"
    #     }
    #     with TempEnvVars(env_vars):
    #         mock_chat_google_genai_instance = mock_chat_google_genai_class.return_value
    #         # Gemini might not have get_num_tokens, or MessageTrimmer might use a different way for it.
    #         # If it's needed and not on the model, MessageTrimmer might fail.
    #         # For now, assume it's okay or not strictly checked by MessageTrimmer for all models.

    #         sc = ServiceContainer()
    #         self.assertIsInstance(sc.langgraph_service.llm, MagicMock)
    #         mock_chat_google_genai_class.assert_called_once_with(
    #             model="gemini-test",
    #             google_api_key="fake_gemini_key"
    #             # max_output_tokens=512 # Check if LangGraphService sets this
    #         )

    @patch('langchain_openai.ChatOpenAI') # DeepSeek uses ChatOpenAI
    def test_deepseek_provider(self, mock_chat_openai_class_for_deepseek):
        """Test configuration for DeepSeek provider."""
        env_vars = {
            "LLM_PROVIDER": "deepseek",
            "LLM_MODEL_NAME": "deepseek-test",
            "DEEPSEEK_API_KEY": "fake_deepseek_key",
            "RAG_ENABLED": "false"
        }
        with TempEnvVars(env_vars):
            mock_chat_openai_instance = mock_chat_openai_class_for_deepseek.return_value
            mock_chat_openai_instance.get_num_tokens = MagicMock(return_value=10)

            sc = ServiceContainer()
            self.assertIsInstance(sc.langgraph_service.llm, MagicMock)
            mock_chat_openai_class_for_deepseek.assert_called_once_with(
                model="deepseek-test",
                openai_api_key="fake_deepseek_key",
                base_url="https://api.deepseek.com/v1",
                max_tokens=512 # from LangGraphService default
            )

    def test_invalid_provider(self):
        """Test for an unsupported LLM provider."""
        env_vars = {
            "LLM_PROVIDER": "unknown_provider",
            "RAG_ENABLED": "false"
        }
        with TempEnvVars(env_vars):
            with self.assertRaisesRegex(ValueError, "Unsupported LLM_PROVIDER: unknown_provider"):
                ServiceContainer()

    def test_missing_api_key_anthropic(self):
        """Test missing Anthropic API key."""
        env_vars = {
            "LLM_PROVIDER": "anthropic",
            "LLM_MODEL_NAME": "claude-test",
            "ANTHROPIC_API_KEY": None, # Ensure it's unset
            "RAG_ENABLED": "false"
        }
        with TempEnvVars(env_vars):
            config_instance = GlobalConfig()
            # Ensure the field is None on the instance if env var was meant to be unset
            if env_vars.get("ANTHROPIC_API_KEY") is None:
                config_instance.ANTHROPIC_API_KEY = None
            with self.assertRaisesRegex(ValueError, "Missing ANTHROPIC_API_KEY for LLM_PROVIDER='anthropic'"):
                config_instance.validate_instance()

    def test_missing_api_key_openai(self):
        """Test missing OpenAI API key."""
        env_vars = {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL_NAME": "gpt-test",
            "OPENAI_API_KEY": None,
            "RAG_ENABLED": "false"
        }
        with TempEnvVars(env_vars):
            config_instance = GlobalConfig()
            if env_vars.get("OPENAI_API_KEY") is None:
                config_instance.OPENAI_API_KEY = None
            with self.assertRaisesRegex(ValueError, "Missing OPENAI_API_KEY for LLM_PROVIDER='openai'"):
                config_instance.validate_instance()

    # Similar tests for Gemini and DeepSeek missing keys can be added.

if __name__ == '__main__':
    unittest.main()
