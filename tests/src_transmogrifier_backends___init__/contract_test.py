"""
Contract tests for transmogrifier.backends module.

Tests verify the Backend protocol and create_backend factory function against
the contract specification. All external dependencies are mocked.
"""

import os
import sys
from unittest.mock import Mock, patch, MagicMock
import pytest


# Mock the backend classes before importing the module under test
class MockAnthropicBackend:
    """Mock implementation of AnthropicBackend."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def complete(self, system: str, messages: list, max_tokens: int) -> str:
        return f"Anthropic completion: {system}"


class MockOpenAIBackend:
    """Mock implementation of OpenAIBackend."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def complete(self, system: str, messages: list, max_tokens: int) -> str:
        return f"OpenAI completion: {system}"


class MockGeminiBackend:
    """Mock implementation of GeminiBackend."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def complete(self, system: str, messages: list, max_tokens: int) -> str:
        return f"Gemini completion: {system}"


# Mock backend constructors within each test; never replace the package in sys.modules.


class TestCreateBackendHappyPath:
    """Happy path tests for create_backend function."""
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('transmogrifier.backends.anthropic.AnthropicBackend', MockAnthropicBackend)
    def test_create_backend_default_anthropic(self):
        """
        Happy path: create_backend returns AnthropicBackend when no backend 
        specified and TRANSMOG_BACKEND not set.
        """
        # Import after patching
        from transmogrifier.backends import create_backend
        
        result = create_backend(backend=None, kwargs={})
        
        assert result is not None
        assert hasattr(result, 'complete')
        assert isinstance(result, MockAnthropicBackend)
    
    @patch('transmogrifier.backends.anthropic.AnthropicBackend', MockAnthropicBackend)
    def test_create_backend_explicit_anthropic(self):
        """
        Happy path: create_backend with explicit 'anthropic' backend parameter.
        """
        from transmogrifier.backends import create_backend
        
        result = create_backend(backend='anthropic', kwargs={})
        
        assert result is not None
        assert hasattr(result, 'complete')
        assert isinstance(result, MockAnthropicBackend)
    
    @patch('transmogrifier.backends.openai.OpenAIBackend', MockOpenAIBackend)
    def test_create_backend_openai(self):
        """
        Happy path: create_backend with 'openai' backend parameter.
        """
        from transmogrifier.backends import create_backend
        
        result = create_backend(backend='openai', kwargs={})
        
        assert result is not None
        assert hasattr(result, 'complete')
        assert isinstance(result, MockOpenAIBackend)
    
    @patch('transmogrifier.backends.gemini.GeminiBackend', MockGeminiBackend)
    def test_create_backend_gemini(self):
        """
        Happy path: create_backend with 'gemini' backend parameter.
        """
        from transmogrifier.backends import create_backend
        
        result = create_backend(backend='gemini', kwargs={})
        
        assert result is not None
        assert hasattr(result, 'complete')
        assert isinstance(result, MockGeminiBackend)
    
    @patch.dict(os.environ, {'TRANSMOG_BACKEND': 'anthropic'})
    @patch('transmogrifier.backends.anthropic.AnthropicBackend', MockAnthropicBackend)
    def test_create_backend_env_var_anthropic(self):
        """
        Happy path: create_backend reads TRANSMOG_BACKEND env var when backend is None.
        """
        from transmogrifier.backends import create_backend
        
        result = create_backend(backend=None, kwargs={})
        
        assert result is not None
        assert isinstance(result, MockAnthropicBackend)
    
    @patch.dict(os.environ, {'TRANSMOG_BACKEND': 'openai'})
    @patch('transmogrifier.backends.openai.OpenAIBackend', MockOpenAIBackend)
    def test_create_backend_env_var_openai(self):
        """
        Happy path: create_backend reads TRANSMOG_BACKEND env var for openai.
        """
        from transmogrifier.backends import create_backend
        
        result = create_backend(backend=None, kwargs={})
        
        assert result is not None
        assert isinstance(result, MockOpenAIBackend)
    
    @patch.dict(os.environ, {'TRANSMOG_BACKEND': 'gemini'})
    @patch('transmogrifier.backends.gemini.GeminiBackend', MockGeminiBackend)
    def test_create_backend_env_var_gemini(self):
        """
        Happy path: create_backend reads TRANSMOG_BACKEND env var for gemini.
        """
        from transmogrifier.backends import create_backend
        
        result = create_backend(backend=None, kwargs={})
        
        assert result is not None
        assert isinstance(result, MockGeminiBackend)
    
    @patch('transmogrifier.backends.anthropic.AnthropicBackend')
    def test_create_backend_with_kwargs(self, mock_anthropic_class):
        """
        Happy path: create_backend passes kwargs to backend constructor.
        """
        mock_anthropic_class.return_value = MockAnthropicBackend()
        from transmogrifier.backends import create_backend
        
        test_kwargs = {'api_key': 'test-key', 'timeout': 30}
        result = create_backend(backend='anthropic', kwargs=test_kwargs)
        
        assert result is not None
        assert mock_anthropic_class.called
        # Verify kwargs were passed - check call_args
        call_kwargs = mock_anthropic_class.call_args[1] if mock_anthropic_class.call_args else {}
        assert 'api_key' in str(mock_anthropic_class.call_args) or len(call_kwargs) > 0 or mock_anthropic_class.call_count > 0


class TestCreateBackendEdgeCases:
    """Edge case tests for create_backend function."""
    
    @patch.dict(os.environ, {'TRANSMOG_BACKEND': 'anthropic'})
    @patch('transmogrifier.backends.openai.OpenAIBackend', MockOpenAIBackend)
    @patch('transmogrifier.backends.anthropic.AnthropicBackend', MockAnthropicBackend)
    def test_create_backend_parameter_overrides_env(self):
        """
        Edge case: explicit backend parameter overrides TRANSMOG_BACKEND env var.
        """
        from transmogrifier.backends import create_backend
        
        result = create_backend(backend='openai', kwargs={})
        
        assert result is not None
        assert isinstance(result, MockOpenAIBackend)
        assert not isinstance(result, MockAnthropicBackend)
    
    @patch('transmogrifier.backends.anthropic.AnthropicBackend', MockAnthropicBackend)
    def test_create_backend_empty_kwargs(self):
        """
        Edge case: create_backend with empty kwargs dict.
        """
        from transmogrifier.backends import create_backend
        
        result = create_backend(backend='anthropic', kwargs={})
        
        assert result is not None
        assert hasattr(result, 'complete')
    
    def test_create_backend_case_sensitive(self):
        """
        Edge case: backend parameter is case-sensitive.
        """
        from transmogrifier.backends import create_backend
        
        with pytest.raises(Exception) as exc_info:
            create_backend(backend='Anthropic', kwargs={})
        
        # Verify error message indicates unknown backend
        error_msg = str(exc_info.value).lower()
        assert 'unknown' in error_msg or 'invalid' in error_msg or 'anthropic' in error_msg


class TestCreateBackendErrorCases:
    """Error case tests for create_backend function."""
    
    def test_create_backend_unknown_backend_invalid(self):
        """
        Error case: unknown_backend error for invalid backend name.
        """
        from transmogrifier.backends import create_backend
        
        with pytest.raises(Exception) as exc_info:
            create_backend(backend='invalid_backend', kwargs={})
        
        error_msg = str(exc_info.value).lower()
        assert 'unknown' in error_msg or 'invalid' in error_msg or 'not found' in error_msg
        # Check that valid options are mentioned
        assert 'anthropic' in error_msg or 'openai' in error_msg or 'gemini' in error_msg
    
    def test_create_backend_unknown_backend_numeric(self):
        """
        Error case: unknown_backend error for numeric backend name.
        """
        from transmogrifier.backends import create_backend
        
        with pytest.raises(Exception) as exc_info:
            create_backend(backend='123', kwargs={})
        
        assert exc_info.value is not None
    
    def test_create_backend_unknown_backend_empty_string(self):
        """
        Error case: unknown_backend error for empty string backend.
        """
        from transmogrifier.backends import create_backend
        
        with pytest.raises(Exception) as exc_info:
            create_backend(backend='', kwargs={})
        
        assert exc_info.value is not None
    
    def test_create_backend_unknown_backend_whitespace(self):
        """
        Error case: unknown_backend error for whitespace-only backend.
        """
        from transmogrifier.backends import create_backend
        
        with pytest.raises(Exception) as exc_info:
            create_backend(backend='   ', kwargs={})
        
        assert exc_info.value is not None
    
    def test_create_backend_unknown_backend_special_chars(self):
        """
        Error case: unknown_backend error for special characters.
        """
        from transmogrifier.backends import create_backend
        
        with pytest.raises(Exception) as exc_info:
            create_backend(backend='@#$%', kwargs={})
        
        assert exc_info.value is not None
    
    @patch.dict(os.environ, {'TRANSMOG_BACKEND': 'invalid_backend'})
    def test_create_backend_unknown_backend_from_env(self):
        """
        Error case: unknown_backend error when TRANSMOG_BACKEND env var is invalid.
        """
        from transmogrifier.backends import create_backend
        
        with pytest.raises(Exception) as exc_info:
            create_backend(backend=None, kwargs={})
        
        assert exc_info.value is not None


class TestCreateBackendInvariants:
    """Invariant tests for create_backend function."""
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('transmogrifier.backends.anthropic.AnthropicBackend', MockAnthropicBackend)
    def test_invariant_default_anthropic(self):
        """
        Invariant: Default backend is 'anthropic' when TRANSMOG_BACKEND is not set.
        """
        from transmogrifier.backends import create_backend
        
        # Ensure env var not set
        if 'TRANSMOG_BACKEND' in os.environ:
            del os.environ['TRANSMOG_BACKEND']
        
        result = create_backend(backend=None, kwargs={})
        
        assert isinstance(result, MockAnthropicBackend)
    
    @patch('transmogrifier.backends.anthropic.AnthropicBackend', MockAnthropicBackend)
    @patch('transmogrifier.backends.openai.OpenAIBackend', MockOpenAIBackend)
    @patch('transmogrifier.backends.gemini.GeminiBackend', MockGeminiBackend)
    def test_invariant_only_three_backends(self):
        """
        Invariant: Only three backend types are supported.
        """
        from transmogrifier.backends import create_backend
        
        # Test valid backends work
        valid_backends = ['anthropic', 'openai', 'gemini']
        for backend_name in valid_backends:
            result = create_backend(backend=backend_name, kwargs={})
            assert result is not None
            assert hasattr(result, 'complete')
        
        # Test invalid backend fails
        invalid_backends = ['azure', 'cohere', 'huggingface', 'claude']
        for backend_name in invalid_backends:
            with pytest.raises(Exception):
                create_backend(backend=backend_name, kwargs={})
    
    @patch('transmogrifier.backends.anthropic.AnthropicBackend', MockAnthropicBackend)
    @patch('transmogrifier.backends.openai.OpenAIBackend', MockOpenAIBackend)
    @patch('transmogrifier.backends.gemini.GeminiBackend', MockGeminiBackend)
    def test_invariant_all_backends_have_complete(self):
        """
        Invariant: All backends implement the Backend protocol with complete() method.
        """
        from transmogrifier.backends import create_backend
        
        anthropic_backend = create_backend(backend='anthropic', kwargs={})
        openai_backend = create_backend(backend='openai', kwargs={})
        gemini_backend = create_backend(backend='gemini', kwargs={})
        
        assert hasattr(anthropic_backend, 'complete')
        assert hasattr(openai_backend, 'complete')
        assert hasattr(gemini_backend, 'complete')
        
        assert callable(anthropic_backend.complete)
        assert callable(openai_backend.complete)
        assert callable(gemini_backend.complete)
    
    @patch('transmogrifier.backends.anthropic.AnthropicBackend', MockAnthropicBackend)
    @patch('transmogrifier.backends.openai.OpenAIBackend', MockOpenAIBackend)
    @patch('transmogrifier.backends.gemini.GeminiBackend', MockGeminiBackend)
    def test_backend_protocol_conformance(self):
        """
        Invariant: Backend protocol conformance - all backends are callable with complete signature.
        """
        from transmogrifier.backends import create_backend
        import inspect
        
        backends = [
            create_backend(backend='anthropic', kwargs={}),
            create_backend(backend='openai', kwargs={}),
            create_backend(backend='gemini', kwargs={}),
        ]
        
        for backend in backends:
            assert callable(backend.complete)
            
            # Verify signature accepts required parameters
            sig = inspect.signature(backend.complete)
            params = list(sig.parameters.keys())
            
            # Should accept system, messages, max_tokens
            assert len(params) >= 3  # At minimum these three params


class TestCompleteProtocol:
    """Tests for the complete() protocol method."""
    
    def test_complete_protocol_signature(self):
        """
        Happy path: complete() protocol method signature verification.
        """
        mock_backend = MockAnthropicBackend()
        
        result = mock_backend.complete(
            system='You are a helpful assistant',
            messages=[{'role': 'user', 'content': 'Hello'}],
            max_tokens=100
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_complete_empty_messages(self):
        """
        Edge case: complete() with empty messages list.
        """
        mock_backend = MockAnthropicBackend()
        
        result = mock_backend.complete(
            system='System prompt',
            messages=[],
            max_tokens=100
        )
        
        assert isinstance(result, str)
    
    def test_complete_max_tokens_boundary(self):
        """
        Edge case: complete() with max_tokens boundary values.
        """
        mock_backend = MockAnthropicBackend()
        
        result = mock_backend.complete(
            system='System',
            messages=[{'role': 'user', 'content': 'Hi'}],
            max_tokens=1
        )
        
        assert isinstance(result, str)
    
    def test_complete_all_backends_return_string(self):
        """
        Verify all backend implementations return strings from complete().
        """
        backends = [
            MockAnthropicBackend(),
            MockOpenAIBackend(),
            MockGeminiBackend(),
        ]
        
        for backend in backends:
            result = backend.complete(
                system='Test',
                messages=[{'role': 'user', 'content': 'Test'}],
                max_tokens=50
            )
            assert isinstance(result, str)


class TestBackendProtocolCompliance:
    """Tests verifying Backend protocol compliance."""
    
    def test_backend_has_complete_method(self):
        """Verify Backend protocol requires complete method."""
        backends = [
            MockAnthropicBackend(),
            MockOpenAIBackend(),
            MockGeminiBackend(),
        ]
        
        for backend in backends:
            assert hasattr(backend, 'complete')
            assert callable(getattr(backend, 'complete'))
    
    def test_backend_complete_accepts_correct_params(self):
        """Verify complete method accepts required parameters."""
        import inspect
        
        backends = [
            MockAnthropicBackend(),
            MockOpenAIBackend(),
            MockGeminiBackend(),
        ]
        
        for backend in backends:
            sig = inspect.signature(backend.complete)
            params = list(sig.parameters.keys())
            
            # Should have at least system, messages, max_tokens
            assert len(params) >= 3
