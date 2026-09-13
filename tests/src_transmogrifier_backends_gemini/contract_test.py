"""
Contract tests for GeminiBackend component.
Tests verify behavior against the contract specification with mocked dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import os


# Mock the google.generativeai module before importing the component
import sys
mock_genai = MagicMock()
sys.modules['google.generativeai'] = mock_genai


from transmogrifier.backends.gemini import GeminiBackend


class TestGeminiBackendInit:
    """Test cases for __init__ method."""
    
    def test_init_with_explicit_parameters(self):
        """Initialize GeminiBackend with explicit API key and model parameters."""
        backend = GeminiBackend(api_key="test-api-key-123", model="gemini-pro")
        
        assert backend._api_key == "test-api-key-123"
        assert backend._model == "gemini-pro"
        assert backend._configured == False
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'env-api-key', 'TRANSMOG_MODEL': 'env-model'})
    def test_init_with_none_parameters_uses_env_vars(self):
        """Initialize GeminiBackend with None parameters falls back to environment variables."""
        backend = GeminiBackend(api_key=None, model=None)
        
        assert backend._api_key == 'env-api-key'
        assert backend._model == 'env-model'
        assert backend._configured == False
    
    @patch.dict(os.environ, {}, clear=True)
    def test_init_with_missing_env_vars_uses_defaults(self):
        """Initialize GeminiBackend when environment variables are not set uses empty string for api_key and default model."""
        backend = GeminiBackend(api_key=None, model=None)
        
        assert backend._api_key == ''
        assert backend._model == 'gemini-2.5-flash'
        assert backend._configured == False
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'env-key'}, clear=True)
    def test_init_with_partial_env_vars(self):
        """Initialize GeminiBackend with only API key in env var, model defaults."""
        backend = GeminiBackend(api_key=None, model=None)
        
        assert backend._api_key == 'env-key'
        assert backend._model == 'gemini-2.5-flash'
        assert backend._configured == False
    
    @patch.dict(os.environ, {}, clear=True)
    def test_invariant_default_model(self):
        """Verify default model is gemini-2.5-flash when not overridden."""
        backend = GeminiBackend(api_key="test-key", model=None)
        
        assert backend._model == 'gemini-2.5-flash'


class TestGeminiBackendEnsureConfigured:
    """Test cases for _ensure_configured method."""
    
    @patch('google.generativeai.configure')
    def test_ensure_configured_first_call(self, mock_configure):
        """First call to _ensure_configured sets configured to True and configures genai."""
        backend = GeminiBackend(api_key="test-api-key", model="gemini-pro")
        
        assert backend._configured == False
        backend._ensure_configured()
        
        assert backend._configured == True
        mock_configure.assert_called_once_with(api_key="test-api-key")
    
    @patch('google.generativeai.configure')
    def test_ensure_configured_subsequent_calls(self, mock_configure):
        """Subsequent calls to _ensure_configured are no-ops."""
        backend = GeminiBackend(api_key="test-api-key", model="gemini-pro")
        
        backend._ensure_configured()
        mock_configure.reset_mock()
        
        backend._ensure_configured()
        backend._ensure_configured()
        
        mock_configure.assert_not_called()
    
    def test_ensure_configured_import_error(self):
        """Raises import error when google.generativeai is not available."""
        # Temporarily replace the module with one that raises ImportError
        original_module = sys.modules.get('google.generativeai')
        
        # Create a module that raises ImportError on attribute access
        class FailingModule:
            def __getattr__(self, name):
                raise ImportError("google.generativeai not installed")
        
        sys.modules['google.generativeai'] = FailingModule()
        
        try:
            # Reimport to get the failing module
            import importlib
            import transmogrifier.backends.gemini
            importlib.reload(transmogrifier.backends.gemini)
            
            backend = transmogrifier.backends.gemini.GeminiBackend(api_key="test-key", model="test-model")
            
            with pytest.raises(ImportError):
                backend._ensure_configured()
        finally:
            # Restore the original mock module
            sys.modules['google.generativeai'] = original_module
            importlib.reload(transmogrifier.backends.gemini)
    
    @patch('google.generativeai.configure')
    def test_ensure_configured_empty_api_key(self, mock_configure):
        """Raises invalid_api_key error when API key is empty string."""
        mock_configure.side_effect = ValueError("Invalid API key")
        
        backend = GeminiBackend(api_key="", model="gemini-pro")
        
        with pytest.raises(ValueError, match="Invalid API key"):
            backend._ensure_configured()
    
    @patch('google.generativeai.configure')
    def test_invariant_configured_never_reverts(self, mock_configure):
        """Verify _configured never transitions from True back to False."""
        backend = GeminiBackend(api_key="test-key", model="gemini-pro")
        
        assert backend._configured == False
        
        backend._ensure_configured()
        assert backend._configured == True
        
        backend._ensure_configured()
        backend._ensure_configured()
        assert backend._configured == True


class TestGeminiBackendComplete:
    """Test cases for complete method."""
    
    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_complete_happy_path(self, mock_configure, mock_model_class):
        """Successfully generate completion with valid messages."""
        # Setup mocks
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Generated response"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        backend = GeminiBackend(api_key="test-key", model="gemini-pro")
        messages = [{'role': 'user', 'content': 'Hello'}]
        
        result = backend.complete(system="You are a helpful assistant", messages=messages, max_tokens=100)
        
        assert isinstance(result, str)
        assert result == "Generated response"
        assert backend._configured == True
        mock_model_class.assert_called_once_with("gemini-pro")
        mock_model.generate_content.assert_called_once()
        
        # Verify generation config has temperature=0
        call_args = mock_model.generate_content.call_args
        assert 'generation_config' in call_args.kwargs
        assert call_args.kwargs['generation_config']['temperature'] == 0
    
    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_complete_multiple_user_messages(self, mock_configure, mock_model_class):
        """Complete with multiple user messages joins them with double newlines."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Response"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        backend = GeminiBackend(api_key="test-key", model="gemini-pro")
        messages = [
            {'role': 'user', 'content': 'First'},
            {'role': 'user', 'content': 'Second'}
        ]
        
        result = backend.complete(system="System prompt", messages=messages, max_tokens=100)
        
        # Check that the prompt contains joined messages
        call_args = mock_model.generate_content.call_args
        prompt = call_args[0][0]
        assert 'First' in prompt
        assert 'Second' in prompt
        assert '\n\n' in prompt
        assert isinstance(result, str)
    
    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_complete_filters_non_user_messages(self, mock_configure, mock_model_class):
        """Complete filters out non-user role messages."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Response"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        backend = GeminiBackend(api_key="test-key", model="gemini-pro")
        messages = [
            {'role': 'user', 'content': 'User msg'},
            {'role': 'assistant', 'content': 'Assistant msg'}
        ]
        
        result = backend.complete(system="System", messages=messages, max_tokens=100)
        
        call_args = mock_model.generate_content.call_args
        prompt = call_args[0][0]
        assert 'User msg' in prompt
        assert 'Assistant msg' not in prompt
    
    def test_complete_empty_messages_error(self):
        """Raises error when messages list is empty."""
        backend = GeminiBackend(api_key="test-key", model="gemini-pro")
        messages = []
        
        with pytest.raises((ValueError, RuntimeError, KeyError)):
            backend.complete(system="System", messages=messages, max_tokens=100)
    
    def test_complete_no_user_messages_error(self):
        """Raises error when no user role messages found."""
        backend = GeminiBackend(api_key="test-key", model="gemini-pro")
        messages = [{'role': 'assistant', 'content': 'Only assistant'}]
        
        with pytest.raises((ValueError, RuntimeError, KeyError)):
            backend.complete(system="System", messages=messages, max_tokens=100)
    
    def test_complete_missing_content_key(self):
        """Raises error when message dict missing content key."""
        backend = GeminiBackend(api_key="test-key", model="gemini-pro")
        messages = [{'role': 'user'}]
        
        with pytest.raises(KeyError):
            backend.complete(system="System", messages=messages, max_tokens=100)
    
    def test_complete_missing_role_key(self):
        """Raises error when message dict missing role key."""
        backend = GeminiBackend(api_key="test-key", model="gemini-pro")
        messages = [{'content': 'Hello'}]
        
        with pytest.raises(KeyError):
            backend.complete(system="System", messages=messages, max_tokens=100)
    
    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_complete_api_error(self, mock_configure, mock_model_class):
        """Raises api_error when Gemini API call fails."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API call failed")
        mock_model_class.return_value = mock_model
        
        backend = GeminiBackend(api_key="test-key", model="gemini-pro")
        messages = [{'role': 'user', 'content': 'Hello'}]
        
        with pytest.raises(Exception, match="API call failed"):
            backend.complete(system="System", messages=messages, max_tokens=100)
    
    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_complete_response_blocked(self, mock_configure, mock_model_class):
        """Raises response_blocked when response.text is unavailable due to safety filters."""
        mock_model = Mock()
        mock_response = Mock()
        # Simulate response.text raising AttributeError
        type(mock_response).text = property(lambda self: (_ for _ in ()).throw(AttributeError("Response blocked")))
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        backend = GeminiBackend(api_key="test-key", model="gemini-pro")
        messages = [{'role': 'user', 'content': 'Hello'}]
        
        with pytest.raises(AttributeError):
            backend.complete(system="System", messages=messages, max_tokens=100)
    
    def test_complete_import_error(self):
        """Raises import_error when google.generativeai not available."""
        original_module = sys.modules.get('google.generativeai')
        
        class FailingModule:
            def __getattr__(self, name):
                raise ImportError("google.generativeai not installed")
        
        sys.modules['google.generativeai'] = FailingModule()
        
        try:
            import importlib
            import transmogrifier.backends.gemini
            importlib.reload(transmogrifier.backends.gemini)
            
            backend = transmogrifier.backends.gemini.GeminiBackend(api_key="test-key", model="test-model")
            messages = [{'role': 'user', 'content': 'Hello'}]
            
            with pytest.raises(ImportError):
                backend.complete(system="System", messages=messages, max_tokens=100)
        finally:
            sys.modules['google.generativeai'] = original_module
            importlib.reload(transmogrifier.backends.gemini)
    
    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_complete_configures_backend_if_not_configured(self, mock_configure, mock_model_class):
        """Complete call ensures backend is configured."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Response"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        backend = GeminiBackend(api_key="test-key", model="gemini-pro")
        messages = [{'role': 'user', 'content': 'Hello'}]
        
        assert backend._configured == False
        backend.complete(system="System", messages=messages, max_tokens=100)
        assert backend._configured == True
    
    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_invariant_temperature_zero(self, mock_configure, mock_model_class):
        """Verify temperature is always 0 for deterministic output."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Response"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        backend = GeminiBackend(api_key="test-key", model="gemini-pro")
        messages = [{'role': 'user', 'content': 'Test'}]
        
        backend.complete(system="System", messages=messages, max_tokens=100)
        
        call_args = mock_model.generate_content.call_args
        generation_config = call_args.kwargs['generation_config']
        assert generation_config['temperature'] == 0
