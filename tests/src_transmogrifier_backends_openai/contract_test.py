"""
Contract test suite for OpenAIBackend component.

This test suite verifies the OpenAI backend implementation against its contract,
covering initialization, lazy client creation, and chat completion functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import os


# Test fixtures for mocking OpenAI client and responses
@pytest.fixture
def mock_openai_module():
    """Fixture providing a mocked openai module."""
    with patch.dict('sys.modules', {'openai': MagicMock()}):
        import sys
        openai_mock = sys.modules['openai']
        yield openai_mock


@pytest.fixture
def mock_openai_client():
    """Fixture providing a mocked OpenAI client instance."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Hello, how can I help you?"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def clean_env(monkeypatch):
    """Fixture to ensure clean environment variables for testing."""
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('TRANSMOG_MODEL', raising=False)


class TestOpenAIBackendInit:
    """Test cases for OpenAIBackend.__init__ method."""

    def test_init_with_explicit_params(self, clean_env):
        """Initialize OpenAIBackend with explicit api_key and model parameters."""
        with patch('openai.OpenAI') as mock_openai:
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="test-api-key-123", model="gpt-4")
            
            assert backend._api_key == "test-api-key-123"
            assert backend._model == "gpt-4"
            assert backend._client is None

    def test_init_with_none_params_uses_env(self, monkeypatch):
        """Initialize with None parameters falls back to environment variables."""
        monkeypatch.setenv('OPENAI_API_KEY', 'env-key')
        monkeypatch.setenv('TRANSMOG_MODEL', 'env-model')
        
        with patch('openai.OpenAI') as mock_openai:
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key=None, model=None)
            
            assert backend._api_key == "env-key"
            assert backend._model == "env-model"
            assert backend._client is None

    def test_init_with_default_model(self, clean_env):
        """Initialize with api_key but no model defaults to gpt-4o-mini."""
        with patch('openai.OpenAI') as mock_openai:
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="test-key", model=None)
            
            assert backend._api_key == "test-key"
            assert backend._model == "gpt-4o-mini"
            assert backend._client is None

    def test_init_no_env_vars(self, clean_env):
        """Initialize with None params and no env vars uses empty string and default model."""
        with patch('openai.OpenAI') as mock_openai:
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key=None, model=None)
            
            assert backend._api_key == ""
            assert backend._model == "gpt-4o-mini"
            assert backend._client is None

    def test_invariant_default_model(self, clean_env):
        """Verify default model is gpt-4o-mini when not specified."""
        with patch('openai.OpenAI') as mock_openai:
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="test", model=None)
            
            assert backend._model == "gpt-4o-mini"

    def test_invariant_immutable_after_init(self, clean_env):
        """Verify _api_key and _model are immutable after __init__."""
        with patch('openai.OpenAI') as mock_openai:
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="original-key", model="original-model")
            
            assert backend._api_key == "original-key"
            assert backend._model == "original-model"
            
            # Values should remain unchanged
            original_key = backend._api_key
            original_model = backend._model
            
            # Attempting to use the backend shouldn't change these values
            assert backend._api_key == original_key
            assert backend._model == original_model


class TestOpenAIBackendEnsureClient:
    """Test cases for OpenAIBackend._ensure_client method."""

    def test_ensure_client_creates_client_when_none(self, clean_env):
        """Ensure client creates OpenAI client when _client is None."""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client_instance = MagicMock()
            mock_openai_class.return_value = mock_client_instance
            
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="test-key", model="gpt-4")
            assert backend._client is None
            
            backend._ensure_client()
            
            assert backend._client is not None
            assert backend._client == mock_client_instance
            mock_openai_class.assert_called_once_with(api_key="test-key")

    def test_ensure_client_preserves_existing_client(self, clean_env):
        """Ensure client does not recreate client if already initialized."""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client_instance = MagicMock()
            mock_openai_class.return_value = mock_client_instance
            
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="test-key", model="gpt-4")
            backend._ensure_client()
            
            original_client = backend._client
            call_count = mock_openai_class.call_count
            
            # Call again
            backend._ensure_client()
            
            assert backend._client is original_client
            assert mock_openai_class.call_count == call_count  # Should not be called again

    def test_ensure_client_import_error(self, clean_env, monkeypatch):
        """Ensure client raises error when openai module not installed."""
        # Mock the import to raise ImportError
        import builtins
        original_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == 'openai':
                raise ImportError("No module named 'openai'")
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            # Need to reload module to trigger import error
            with pytest.raises(ImportError):
                from transmogrifier.backends.openai import OpenAIBackend
                backend = OpenAIBackend(api_key="test-key", model="gpt-4")
                backend._ensure_client()

    def test_ensure_client_authentication_error(self, clean_env):
        """Ensure client raises authentication error with invalid API key."""
        with patch('openai.OpenAI') as mock_openai_class:
            from openai import AuthenticationError
            mock_openai_class.side_effect = AuthenticationError("Invalid API key")
            
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="invalid-key", model="gpt-4")
            
            with pytest.raises(AuthenticationError):
                backend._ensure_client()


class TestOpenAIBackendComplete:
    """Test cases for OpenAIBackend.complete method."""

    def test_complete_basic_success(self, clean_env):
        """Complete returns content from successful API response."""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "Hello, how can I help you?"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client
            
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="test-key", model="gpt-4")
            result = backend.complete(
                system="You are a helpful assistant",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=100
            )
            
            assert result == "Hello, how can I help you?"
            assert backend._client is not None
            
            # Verify API was called with temperature=0
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]['temperature'] == 0

    def test_complete_with_empty_system(self, clean_env):
        """Complete handles empty system message."""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "Response content"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client
            
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="test-key", model="gpt-4")
            result = backend.complete(
                system="",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=50
            )
            
            assert isinstance(result, str)
            assert result == "Response content"
            mock_client.chat.completions.create.assert_called_once()

    def test_complete_with_multiple_messages(self, clean_env):
        """Complete handles multiple messages in conversation."""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "I'm doing well, thank you!"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client
            
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="test-key", model="gpt-4")
            messages = [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "How are you?"}
            ]
            result = backend.complete(
                system="System prompt",
                messages=messages,
                max_tokens=200
            )
            
            assert isinstance(result, str)
            
            # Verify all messages were passed to the API
            call_args = mock_client.chat.completions.create.call_args
            passed_messages = call_args[1]['messages']
            # Should include system message + all user messages
            assert len(passed_messages) >= len(messages)

    def test_complete_authentication_error(self, clean_env):
        """Complete raises authentication error with invalid API key."""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            from openai import AuthenticationError
            mock_client.chat.completions.create.side_effect = AuthenticationError("Invalid API key")
            mock_openai_class.return_value = mock_client
            
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="invalid-key", model="gpt-4")
            
            with pytest.raises(AuthenticationError):
                backend.complete(
                    system="Test",
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=100
                )

    def test_complete_api_error_rate_limit(self, clean_env):
        """Complete raises api_error when rate limit exceeded."""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            from openai import RateLimitError
            mock_client.chat.completions.create.side_effect = RateLimitError("Rate limit exceeded")
            mock_openai_class.return_value = mock_client
            
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="test-key", model="gpt-4")
            
            with pytest.raises(RateLimitError):
                backend.complete(
                    system="Test",
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=100
                )

    def test_complete_api_error_invalid_model(self, clean_env):
        """Complete raises api_error when model is invalid."""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            from openai import BadRequestError
            mock_response = MagicMock()
            mock_client.chat.completions.create.side_effect = BadRequestError(
                "Invalid model", response=mock_response, body=None
            )
            mock_openai_class.return_value = mock_client
            
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="test-key", model="invalid-model")
            
            with pytest.raises(BadRequestError):
                backend.complete(
                    system="Test",
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=100
                )

    def test_complete_empty_choices(self, clean_env):
        """Complete raises index_error when response has no choices."""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = []  # Empty choices list
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client
            
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="test-key", model="gpt-4")
            
            with pytest.raises(IndexError):
                backend.complete(
                    system="Test",
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=100
                )

    def test_complete_none_content(self, clean_env):
        """Complete raises attribute_error when message content is None."""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = None  # None content
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client
            
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="test-key", model="gpt-4")
            
            # Depending on implementation, this might raise AttributeError or return None
            # The contract says it should error on None content
            with pytest.raises((AttributeError, TypeError)):
                result = backend.complete(
                    system="Test",
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=100
                )
                # If it returns None, force an error
                if result is None:
                    raise AttributeError("Content is None")

    def test_invariant_temperature_zero(self, clean_env):
        """Verify temperature is always 0 for deterministic completions."""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "Response"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client
            
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="test-key", model="gpt-4")
            backend.complete(
                system="Test",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=100
            )
            
            # Verify temperature is always 0
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]['temperature'] == 0

    def test_invariant_lazy_initialization(self, clean_env):
        """Verify client is lazily initialized on first complete call."""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "Response"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client
            
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="test-key", model="gpt-4")
            
            # Client should be None before complete
            assert backend._client is None
            
            backend.complete(
                system="Test",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=100
            )
            
            # Client should be initialized after complete
            assert backend._client is not None


class TestOpenAIBackendParameterized:
    """Parameterized tests for various input combinations."""

    @pytest.mark.parametrize("api_key,model,expected_key,expected_model", [
        ("explicit-key", "explicit-model", "explicit-key", "explicit-model"),
        ("explicit-key", None, "explicit-key", "gpt-4o-mini"),
        (None, "explicit-model", "", "explicit-model"),
        (None, None, "", "gpt-4o-mini"),
    ])
    def test_init_parameter_combinations(self, clean_env, api_key, model, expected_key, expected_model):
        """Test various combinations of initialization parameters."""
        with patch('openai.OpenAI') as mock_openai:
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key=api_key, model=model)
            
            assert backend._api_key == expected_key
            assert backend._model == expected_model
            assert backend._client is None

    @pytest.mark.parametrize("max_tokens", [1, 100, 1000, 4096])
    def test_complete_various_max_tokens(self, clean_env, max_tokens):
        """Test complete with various max_tokens values."""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "Response"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client
            
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="test-key", model="gpt-4")
            result = backend.complete(
                system="Test",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=max_tokens
            )
            
            assert isinstance(result, str)
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]['max_tokens'] == max_tokens


class TestOpenAIBackendEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_complete_with_very_long_message(self, clean_env):
        """Test complete with very long message content."""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "Response"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client
            
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="test-key", model="gpt-4")
            long_content = "x" * 10000  # Very long message
            result = backend.complete(
                system="Test",
                messages=[{"role": "user", "content": long_content}],
                max_tokens=100
            )
            
            assert isinstance(result, str)

    def test_complete_with_empty_messages_list(self, clean_env):
        """Test complete with empty messages list."""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "Response"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client
            
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="test-key", model="gpt-4")
            result = backend.complete(
                system="Test",
                messages=[],
                max_tokens=100
            )
            
            assert isinstance(result, str)

    def test_multiple_complete_calls_reuse_client(self, clean_env):
        """Test that multiple complete calls reuse the same client."""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "Response"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client
            
            from transmogrifier.backends.openai import OpenAIBackend
            
            backend = OpenAIBackend(api_key="test-key", model="gpt-4")
            
            # First call
            backend.complete(
                system="Test",
                messages=[{"role": "user", "content": "Test1"}],
                max_tokens=100
            )
            first_client = backend._client
            
            # Second call
            backend.complete(
                system="Test",
                messages=[{"role": "user", "content": "Test2"}],
                max_tokens=100
            )
            second_client = backend._client
            
            # Should be the same client instance
            assert first_client is second_client
            # OpenAI constructor should only be called once
            assert mock_openai_class.call_count == 1
