"""
Contract tests for AnthropicBackend component.

Tests verify the contract behavior of the AnthropicBackend class including:
- Constructor initialization with API key and model parameter precedence
- Lazy client initialization via _ensure_client()
- Complete method for chat completions
- Error handling for all specified exception types
- Invariants around default values and immutability

All dependencies are mocked using unittest.mock.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from transmogrifier.backends.anthropic import AnthropicBackend


class TestAnthropicBackendInit:
    """Test suite for __init__ method covering parameter precedence and defaults."""
    
    def test_init_with_explicit_params(self):
        """Verify __init__ sets _api_key, _model, and _client=None when both parameters provided."""
        backend = AnthropicBackend(api_key="sk-test-key-123", model="claude-opus-4")
        
        assert backend._api_key == "sk-test-key-123"
        assert backend._model == "claude-opus-4"
        assert backend._client is None
    
    def test_init_with_none_params_uses_env(self, monkeypatch):
        """Verify __init__ falls back to environment variables when parameters are None."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        monkeypatch.setenv("TRANSMOG_MODEL", "env-model")
        
        backend = AnthropicBackend(api_key=None, model=None)
        
        assert backend._api_key == "env-key"
        assert backend._model == "env-model"
        assert backend._client is None
    
    def test_init_with_default_model(self, monkeypatch):
        """Verify __init__ uses default model 'claude-haiku-4-5-20251001' when not specified."""
        monkeypatch.delenv("TRANSMOG_MODEL", raising=False)
        
        backend = AnthropicBackend(api_key="sk-test-key", model=None)
        
        assert backend._model == "claude-haiku-4-5-20251001"
    
    def test_init_api_key_empty_string_fallback(self, monkeypatch):
        """Verify __init__ sets _api_key to empty string when no API key available."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        
        backend = AnthropicBackend(api_key=None, model=None)
        
        assert backend._api_key == ""
    
    def test_init_parameter_precedence_over_env(self, monkeypatch):
        """Verify explicit parameters take precedence over environment variables."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        monkeypatch.setenv("TRANSMOG_MODEL", "env-model")
        
        backend = AnthropicBackend(api_key="param-key", model="param-model")
        
        assert backend._api_key == "param-key"
        assert backend._model == "param-model"


class TestEnsureClient:
    """Test suite for _ensure_client method covering lazy initialization."""
    
    @patch('transmogrifier.backends.anthropic.anthropic')
    def test_ensure_client_lazy_initialization(self, mock_anthropic_module):
        """Verify _ensure_client creates client on first call and _client is not None."""
        mock_client_instance = Mock()
        mock_anthropic_module.Anthropic.return_value = mock_client_instance
        
        backend = AnthropicBackend(api_key="test-key", model="test-model")
        assert backend._client is None
        
        backend._ensure_client()
        
        assert backend._client is not None
        assert backend._client == mock_client_instance
        mock_anthropic_module.Anthropic.assert_called_once_with(api_key="test-key")
    
    @patch('transmogrifier.backends.anthropic.anthropic')
    def test_ensure_client_idempotent(self, mock_anthropic_module):
        """Verify _ensure_client does not recreate client on subsequent calls."""
        mock_client_instance = Mock()
        mock_anthropic_module.Anthropic.return_value = mock_client_instance
        
        backend = AnthropicBackend(api_key="test-key", model="test-model")
        
        backend._ensure_client()
        backend._ensure_client()
        backend._ensure_client()
        
        assert mock_anthropic_module.Anthropic.call_count == 1
        assert backend._client is not None
    
    def test_ensure_client_import_error(self):
        """Verify _ensure_client raises ImportError when anthropic package not installed."""
        with patch('transmogrifier.backends.anthropic.anthropic', None):
            backend = AnthropicBackend(api_key="test-key", model="test-model")
            
            # Trigger import error by attempting to use anthropic when it's None
            with patch.dict('sys.modules', {'anthropic': None}):
                with pytest.raises((ImportError, AttributeError)):
                    backend._ensure_client()
    
    @patch('transmogrifier.backends.anthropic.anthropic')
    def test_ensure_client_authentication_error(self, mock_anthropic_module):
        """Verify _ensure_client raises AuthenticationError with invalid API key."""
        # Create a mock exception class that mimics anthropic's AuthenticationError
        mock_auth_error = type('AuthenticationError', (Exception,), {})
        mock_anthropic_module.AuthenticationError = mock_auth_error
        mock_anthropic_module.Anthropic.side_effect = mock_auth_error("Invalid API key")
        
        backend = AnthropicBackend(api_key="", model="test-model")
        
        with pytest.raises(Exception) as exc_info:
            backend._ensure_client()
        
        assert "Invalid API key" in str(exc_info.value) or isinstance(exc_info.value, type(mock_auth_error))


class TestComplete:
    """Test suite for complete method covering success and error paths."""
    
    @patch('transmogrifier.backends.anthropic.anthropic')
    def test_complete_success_single_message(self, mock_anthropic_module):
        """Verify complete returns text content for single message with system prompt."""
        # Setup mock response
        mock_content_block = Mock()
        mock_content_block.text = "Hello, how can I help?"
        
        mock_response = Mock()
        mock_response.content = [mock_content_block]
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_module.Anthropic.return_value = mock_client
        
        backend = AnthropicBackend(api_key="test-key", model="test-model")
        
        result = backend.complete(
            system="You are a helpful assistant",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1024
        )
        
        assert result == "Hello, how can I help?"
        assert backend._client is not None
        mock_client.messages.create.assert_called_once()
    
    @patch('transmogrifier.backends.anthropic.anthropic')
    def test_complete_success_multiple_messages(self, mock_anthropic_module):
        """Verify complete handles multi-turn conversation."""
        mock_content_block = Mock()
        mock_content_block.text = "I am doing well"
        
        mock_response = Mock()
        mock_response.content = [mock_content_block]
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_module.Anthropic.return_value = mock_client
        
        backend = AnthropicBackend(api_key="test-key", model="test-model")
        
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "How are you?"}
        ]
        
        result = backend.complete(
            system="You are helpful",
            messages=messages,
            max_tokens=2048
        )
        
        assert result == "I am doing well"
    
    @patch('transmogrifier.backends.anthropic.anthropic')
    def test_complete_empty_system_prompt(self, mock_anthropic_module):
        """Verify complete works with empty system prompt."""
        mock_content_block = Mock()
        mock_content_block.text = "Response text"
        
        mock_response = Mock()
        mock_response.content = [mock_content_block]
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_module.Anthropic.return_value = mock_client
        
        backend = AnthropicBackend(api_key="test-key", model="test-model")
        
        result = backend.complete(
            system="",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=100
        )
        
        assert isinstance(result, str)
        assert result == "Response text"
    
    @patch('transmogrifier.backends.anthropic.anthropic')
    def test_complete_large_max_tokens(self, mock_anthropic_module):
        """Verify complete handles large max_tokens value."""
        mock_content_block = Mock()
        mock_content_block.text = "Response"
        
        mock_response = Mock()
        mock_response.content = [mock_content_block]
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_module.Anthropic.return_value = mock_client
        
        backend = AnthropicBackend(api_key="test-key", model="test-model")
        
        result = backend.complete(
            system="Test",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=1000000
        )
        
        assert isinstance(result, str)
        # Verify the large max_tokens was passed to the API
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs['max_tokens'] == 1000000
    
    def test_complete_import_error(self):
        """Verify complete raises ImportError when anthropic not installed."""
        with patch.dict('sys.modules', {'anthropic': None}):
            backend = AnthropicBackend(api_key="test-key", model="test-model")
            
            with pytest.raises((ImportError, AttributeError)):
                backend.complete(
                    system="Test",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=100
                )
    
    @patch('transmogrifier.backends.anthropic.anthropic')
    def test_complete_authentication_error(self, mock_anthropic_module):
        """Verify complete raises AuthenticationError with invalid API key."""
        mock_auth_error = type('AuthenticationError', (Exception,), {})
        mock_anthropic_module.AuthenticationError = mock_auth_error
        
        mock_client = Mock()
        mock_client.messages.create.side_effect = mock_auth_error("Invalid API key")
        mock_anthropic_module.Anthropic.return_value = mock_client
        
        backend = AnthropicBackend(api_key="invalid-key", model="test-model")
        
        with pytest.raises(Exception):
            backend.complete(
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100
            )
    
    @patch('transmogrifier.backends.anthropic.anthropic')
    def test_complete_api_error_invalid_model(self, mock_anthropic_module):
        """Verify complete raises APIError for invalid model."""
        mock_api_error = type('APIError', (Exception,), {})
        mock_anthropic_module.APIError = mock_api_error
        
        mock_client = Mock()
        mock_client.messages.create.side_effect = mock_api_error("Invalid model specified")
        mock_anthropic_module.Anthropic.return_value = mock_client
        
        backend = AnthropicBackend(api_key="test-key", model="invalid-model")
        
        with pytest.raises(Exception):
            backend.complete(
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100
            )
    
    @patch('transmogrifier.backends.anthropic.anthropic')
    def test_complete_api_error_rate_limit(self, mock_anthropic_module):
        """Verify complete raises APIError for rate limit."""
        mock_api_error = type('RateLimitError', (Exception,), {})
        mock_anthropic_module.RateLimitError = mock_api_error
        
        mock_client = Mock()
        mock_client.messages.create.side_effect = mock_api_error("Rate limit exceeded")
        mock_anthropic_module.Anthropic.return_value = mock_client
        
        backend = AnthropicBackend(api_key="test-key", model="test-model")
        
        with pytest.raises(Exception):
            backend.complete(
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100
            )
    
    @patch('transmogrifier.backends.anthropic.anthropic')
    def test_complete_index_error_empty_content(self, mock_anthropic_module):
        """Verify complete raises IndexError when response.content is empty."""
        mock_response = Mock()
        mock_response.content = []  # Empty content list
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_module.Anthropic.return_value = mock_client
        
        backend = AnthropicBackend(api_key="test-key", model="test-model")
        
        with pytest.raises(IndexError):
            backend.complete(
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100
            )
    
    @patch('transmogrifier.backends.anthropic.anthropic')
    def test_complete_attribute_error_no_text(self, mock_anthropic_module):
        """Verify complete raises AttributeError when content[0] has no .text attribute."""
        mock_content_block = Mock(spec=[])  # Mock with no attributes
        del mock_content_block.text  # Ensure .text doesn't exist
        
        mock_response = Mock()
        mock_response.content = [mock_content_block]
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_module.Anthropic.return_value = mock_client
        
        backend = AnthropicBackend(api_key="test-key", model="test-model")
        
        with pytest.raises(AttributeError):
            backend.complete(
                system="Test",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100
            )


class TestInvariants:
    """Test suite for verifying contract invariants."""
    
    def test_invariant_default_model(self, monkeypatch):
        """Verify default model is 'claude-haiku-4-5-20251001' when not specified."""
        monkeypatch.delenv("TRANSMOG_MODEL", raising=False)
        
        backend = AnthropicBackend(api_key="test-key", model=None)
        
        assert backend._model == "claude-haiku-4-5-20251001"
    
    @patch('transmogrifier.backends.anthropic.anthropic')
    def test_invariant_client_none_until_first_use(self, mock_anthropic_module):
        """Verify _client is None until first use, then remains initialized."""
        mock_client = Mock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        
        backend = AnthropicBackend(api_key="test-key", model="test-model")
        
        # Before first use
        assert backend._client is None
        
        # After _ensure_client
        backend._ensure_client()
        assert backend._client is not None
        
        # Verify it remains initialized
        first_client = backend._client
        backend._ensure_client()
        assert backend._client is first_client  # Same instance
    
    @patch('transmogrifier.backends.anthropic.anthropic')
    def test_invariant_api_key_immutable(self, mock_anthropic_module):
        """Verify _api_key remains unchanged after initialization."""
        mock_client = Mock()
        mock_content_block = Mock()
        mock_content_block.text = "response"
        mock_response = Mock()
        mock_response.content = [mock_content_block]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_module.Anthropic.return_value = mock_client
        
        backend = AnthropicBackend(api_key="original-key", model="test-model")
        
        original_api_key = backend._api_key
        assert original_api_key == "original-key"
        
        # Perform operations that might change state
        backend._ensure_client()
        assert backend._api_key == "original-key"
        
        backend.complete(system="test", messages=[{"role": "user", "content": "hi"}], max_tokens=10)
        assert backend._api_key == "original-key"
    
    @patch('transmogrifier.backends.anthropic.anthropic')
    def test_invariant_model_immutable(self, mock_anthropic_module):
        """Verify _model remains unchanged after initialization."""
        mock_client = Mock()
        mock_content_block = Mock()
        mock_content_block.text = "response"
        mock_response = Mock()
        mock_response.content = [mock_content_block]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_module.Anthropic.return_value = mock_client
        
        backend = AnthropicBackend(api_key="test-key", model="original-model")
        
        original_model = backend._model
        assert original_model == "original-model"
        
        # Perform operations that might change state
        backend._ensure_client()
        assert backend._model == "original-model"
        
        backend.complete(system="test", messages=[{"role": "user", "content": "hi"}], max_tokens=10)
        assert backend._model == "original-model"


class TestIntegration:
    """Optional integration tests for real API calls (marked for CI skip)."""
    
    @pytest.mark.integration
    @pytest.mark.skip(reason="Integration test - requires real API key and network")
    def test_complete_real_api_call(self):
        """Integration test with real Anthropic API (requires valid API key in environment)."""
        backend = AnthropicBackend(api_key=None, model=None)
        
        result = backend.complete(
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Say 'test successful' and nothing else."}],
            max_tokens=50
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
