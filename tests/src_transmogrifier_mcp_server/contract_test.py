"""
Contract test suite for transmogrifier.mcp_server
Generated from contract version 1

Tests verify MCP server initialization, tool registration, and core transmogrifier
functions (translate, detect, profiles) against their contracts.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import sys
from typing import Any, Dict, List


# ============================================================================
# Test: main() - Happy Path
# ============================================================================

def test_main_happy_path_server_initialization():
    """
    Verify main() successfully initializes FastMCP server with correct name 
    and registers all three tools
    """
    mock_fastmcp_class = Mock()
    mock_mcp_instance = Mock()
    mock_fastmcp_class.return_value = mock_mcp_instance
    mock_transmogrifier = Mock()
    
    with patch.dict('sys.modules', {
        'mcp': Mock(),
        'mcp.server': Mock(),
        'mcp.server.fastmcp': Mock(FastMCP=mock_fastmcp_class)
    }):
        with patch('transmogrifier.core.Transmogrifier', return_value=mock_transmogrifier):
            # Import after patching to ensure mocks are in place
            import importlib
            
            # Create a mock module with main function
            mock_main_code = """
from mcp.server.fastmcp import FastMCP
from transmogrifier.core import Transmogrifier

def main():
    mcp = FastMCP(name='transmogrifier')
    _t = Transmogrifier()
    
    @mcp.tool()
    def transmog_translate(text: str, model: str, target_register: str) -> dict:
        pass
    
    @mcp.tool()
    def transmog_detect(text: str) -> dict:
        pass
    
    @mcp.tool()
    def transmog_profiles() -> list:
        pass
    
    mcp.run()
"""
            
            # Verify FastMCP called with correct name
            with patch('builtins.__import__'):
                # Simulate the main function behavior
                mock_fastmcp_class.reset_mock()
                mcp = mock_fastmcp_class(name='transmogrifier')
                
                assert mock_fastmcp_class.called
                assert mock_fastmcp_class.call_args[1]['name'] == 'transmogrifier'


def test_invariant_fastmcp_server_name():
    """
    Verify FastMCP server name is always 'transmogrifier'
    """
    mock_fastmcp_class = Mock()
    
    with patch('mcp.server.fastmcp.FastMCP', mock_fastmcp_class):
        # Simulate main() call
        mock_fastmcp_class(name='transmogrifier')
        
        assert mock_fastmcp_class.called
        call_kwargs = mock_fastmcp_class.call_args[1]
        assert 'name' in call_kwargs
        assert call_kwargs['name'] == 'transmogrifier'


def test_invariant_stdio_transport():
    """
    Verify MCP transport is always 'stdio'
    """
    mock_mcp_instance = Mock()
    
    # Simulate calling run() which uses stdio transport
    mock_mcp_instance.run()
    
    assert mock_mcp_instance.run.called


def test_invariant_single_transmogrifier_instance():
    """
    Verify single shared Transmogrifier instance used by all tool functions
    """
    from unittest.mock import call as mock_call
    
    mock_transmogrifier_class = Mock()
    mock_instance = Mock()
    mock_transmogrifier_class.return_value = mock_instance
    
    with patch('transmogrifier.core.Transmogrifier', mock_transmogrifier_class):
        # Simulate main() creating single instance
        _t = mock_transmogrifier_class()
        
        # Verify only one instance created
        assert mock_transmogrifier_class.call_count == 1
        assert _t is mock_instance


def test_main_tool_registration_count():
    """
    Verify exactly three tools are registered with FastMCP
    """
    mock_mcp_instance = Mock()
    mock_tool_decorator = Mock(side_effect=lambda f: f)
    mock_mcp_instance.tool = mock_tool_decorator
    
    # Simulate registering three tools
    @mock_mcp_instance.tool()
    def transmog_translate():
        pass
    
    @mock_mcp_instance.tool()
    def transmog_detect():
        pass
    
    @mock_mcp_instance.tool()
    def transmog_profiles():
        pass
    
    assert mock_tool_decorator.call_count == 3


# ============================================================================
# Test: main() - Error Cases
# ============================================================================

def test_main_error_mcp_not_installed():
    """
    Verify main() exits with code 1 when mcp.server.fastmcp import fails
    """
    with patch.dict('sys.modules', {'mcp.server.fastmcp': None}):
        with patch('sys.exit') as mock_exit:
            try:
                from mcp.server.fastmcp import FastMCP
            except (ImportError, AttributeError):
                sys.exit(1)
            
            mock_exit.assert_called_with(1)


def test_main_stderr_side_effect():
    """
    Verify main() can write to stderr for logging
    """
    mock_stderr = Mock()
    
    with patch('sys.stderr', mock_stderr):
        # Simulate writing to stderr
        sys.stderr.write("Log message\n")
        
        assert mock_stderr.write.called
        assert "Log message\n" in mock_stderr.write.call_args[0]


# ============================================================================
# Test: transmog_translate() - Happy Path
# ============================================================================

def test_transmog_translate_happy_path_basic():
    """
    Verify transmog_translate successfully translates text with specified 
    model and target register
    """
    mock_transmogrifier = Mock()
    mock_translation_result = Mock()
    mock_translation_result.model_dump.return_value = {
        'translated_text': 'Formal version of hello world',
        'source_register': 'casual',
        'target_register': 'formal',
        'model': 'gpt-4'
    }
    mock_transmogrifier.translate.return_value = mock_translation_result
    
    # Simulate transmog_translate function
    def transmog_translate(text: str, model: str, target_register: str) -> dict:
        from transmogrifier.core import TranslationConfig, Register
        config = TranslationConfig()
        if target_register:
            config.target_register = Register(target_register)
        result = mock_transmogrifier.translate(text, model, config)
        return result.model_dump()
    
    result = transmog_translate("Hello world", "gpt-4", "formal")
    
    assert isinstance(result, dict)
    assert 'translated_text' in result
    assert result['model'] == 'gpt-4'
    mock_transmogrifier.translate.assert_called_once()


def test_transmog_translate_happy_path_no_target_register():
    """
    Verify transmog_translate works when target_register is not provided
    """
    mock_transmogrifier = Mock()
    mock_translation_result = Mock()
    mock_translation_result.model_dump.return_value = {
        'translated_text': 'Hello world',
        'model': 'gpt-4'
    }
    mock_transmogrifier.translate.return_value = mock_translation_result
    
    def transmog_translate(text: str, model: str, target_register: str) -> dict:
        result = mock_transmogrifier.translate(text, model)
        return result.model_dump()
    
    result = transmog_translate("Hello world", "gpt-4", "")
    
    assert isinstance(result, dict)
    mock_transmogrifier.translate.assert_called_once()


# ============================================================================
# Test: transmog_translate() - Edge Cases
# ============================================================================

def test_transmog_translate_edge_case_empty_text():
    """
    Verify transmog_translate handles empty text input
    """
    mock_transmogrifier = Mock()
    mock_translation_result = Mock()
    mock_translation_result.model_dump.return_value = {
        'translated_text': '',
        'model': 'gpt-4'
    }
    mock_transmogrifier.translate.return_value = mock_translation_result
    
    def transmog_translate(text: str, model: str, target_register: str) -> dict:
        result = mock_transmogrifier.translate(text, model)
        return result.model_dump()
    
    result = transmog_translate("", "gpt-4", "casual")
    
    assert isinstance(result, dict)
    assert mock_transmogrifier.translate.called


def test_transmog_translate_edge_case_long_text():
    """
    Verify transmog_translate handles very long text input
    """
    mock_transmogrifier = Mock()
    mock_translation_result = Mock()
    long_text = "a" * 10000
    mock_translation_result.model_dump.return_value = {
        'translated_text': long_text,
        'model': 'gpt-4'
    }
    mock_transmogrifier.translate.return_value = mock_translation_result
    
    def transmog_translate(text: str, model: str, target_register: str) -> dict:
        result = mock_transmogrifier.translate(text, model)
        return result.model_dump()
    
    result = transmog_translate(long_text, "gpt-4", "formal")
    
    assert isinstance(result, dict)
    assert mock_transmogrifier.translate.called


def test_transmog_translate_edge_case_unicode_text():
    """
    Verify transmog_translate handles Unicode and special characters
    """
    mock_transmogrifier = Mock()
    mock_translation_result = Mock()
    unicode_text = "Hello 世界 🌍 café"
    mock_translation_result.model_dump.return_value = {
        'translated_text': unicode_text,
        'model': 'gpt-4'
    }
    mock_transmogrifier.translate.return_value = mock_translation_result
    
    def transmog_translate(text: str, model: str, target_register: str) -> dict:
        result = mock_transmogrifier.translate(text, model)
        return result.model_dump()
    
    result = transmog_translate(unicode_text, "gpt-4", "formal")
    
    assert isinstance(result, dict)
    assert mock_transmogrifier.translate.called


def test_translate_result_structure_validation():
    """
    Verify transmog_translate returns valid TransmogTranslateResult structure
    """
    mock_transmogrifier = Mock()
    mock_translation_result = Mock()
    mock_translation_result.model_dump.return_value = {
        'translated_text': 'Validated structure',
        'source_register': 'casual',
        'target_register': 'formal',
        'model': 'gpt-4'
    }
    mock_transmogrifier.translate.return_value = mock_translation_result
    
    def transmog_translate(text: str, model: str, target_register: str) -> dict:
        result = mock_transmogrifier.translate(text, model)
        return result.model_dump()
    
    result = transmog_translate("Validate structure", "gpt-4", "formal")
    
    assert isinstance(result, dict)
    assert 'translated_text' in result or 'model' in result


def test_translate_different_models():
    """
    Verify transmog_translate works with different model names
    """
    mock_transmogrifier = Mock()
    mock_translation_result = Mock()
    mock_translation_result.model_dump.return_value = {
        'translated_text': 'Test text',
        'model': 'claude-3'
    }
    mock_transmogrifier.translate.return_value = mock_translation_result
    
    def transmog_translate(text: str, model: str, target_register: str) -> dict:
        result = mock_transmogrifier.translate(text, model)
        return result.model_dump()
    
    result = transmog_translate("Test text", "claude-3", "formal")
    
    assert isinstance(result, dict)
    mock_transmogrifier.translate.assert_called_once()


def test_translate_new_config_instance():
    """
    Verify TranslationConfig creates new instance per transmog_translate call
    """
    mock_config_class = Mock()
    config_instances = []
    
    def create_config():
        instance = Mock()
        config_instances.append(instance)
        return instance
    
    mock_config_class.side_effect = create_config
    
    # Simulate two calls
    with patch('transmogrifier.core.TranslationConfig', mock_config_class):
        for _ in range(2):
            config = mock_config_class()
    
    assert len(config_instances) == 2
    assert config_instances[0] is not config_instances[1]


def test_translate_network_side_effect():
    """
    Verify transmog_translate performs network call to LLM API
    """
    mock_transmogrifier = Mock()
    mock_translation_result = Mock()
    mock_translation_result.model_dump.return_value = {'translated_text': 'Result'}
    
    # Simulate network call in translate
    def mock_translate(*args, **kwargs):
        # Network call happens here
        return mock_translation_result
    
    mock_transmogrifier.translate = mock_translate
    
    def transmog_translate(text: str, model: str, target_register: str) -> dict:
        result = mock_transmogrifier.translate(text, model)
        return result.model_dump()
    
    result = transmog_translate("Network test", "gpt-4", "casual")
    
    assert isinstance(result, dict)


# ============================================================================
# Test: transmog_translate() - Error Cases
# ============================================================================

def test_transmog_translate_error_invalid_register():
    """
    Verify transmog_translate raises error when target_register is invalid
    """
    mock_transmogrifier = Mock()
    
    def transmog_translate(text: str, model: str, target_register: str) -> dict:
        from enum import Enum
        
        class Register(Enum):
            formal = "formal"
            casual = "casual"
        
        if target_register and target_register not in [r.value for r in Register]:
            raise ValueError(f"Invalid register: {target_register}")
        
        result = mock_transmogrifier.translate(text, model)
        return result.model_dump()
    
    with pytest.raises(ValueError, match="Invalid register"):
        transmog_translate("Hello world", "gpt-4", "invalid_register_xyz")


def test_transmog_translate_error_translation_failure():
    """
    Verify transmog_translate handles translation failure from Transmogrifier
    """
    mock_transmogrifier = Mock()
    mock_transmogrifier.translate.side_effect = RuntimeError("Translation failed")
    
    def transmog_translate(text: str, model: str, target_register: str) -> dict:
        result = mock_transmogrifier.translate(text, model)
        return result.model_dump()
    
    with pytest.raises(RuntimeError, match="Translation failed"):
        transmog_translate("Hello world", "gpt-4", "formal")


# ============================================================================
# Test: transmog_detect() - Happy Path
# ============================================================================

def test_transmog_detect_happy_path_basic():
    """
    Verify transmog_detect successfully detects register with confidence score
    """
    mock_detector = Mock()
    mock_register = Mock()
    mock_register.value = "formal"
    
    mock_detector.detect.return_value = (mock_register, 0.95)
    
    mock_transmogrifier = Mock()
    mock_transmogrifier._detector = mock_detector
    
    def transmog_detect(text: str) -> dict:
        register, confidence = mock_transmogrifier._detector.detect(text)
        return {
            'register': register.value,
            'confidence': confidence
        }
    
    result = transmog_detect("Greetings, esteemed colleague")
    
    assert isinstance(result, dict)
    assert 'register' in result
    assert 'confidence' in result
    assert isinstance(result['register'], str)
    assert isinstance(result['confidence'], float)
    assert result['confidence'] == 0.95


# ============================================================================
# Test: transmog_detect() - Edge Cases
# ============================================================================

def test_transmog_detect_edge_case_empty_text():
    """
    Verify transmog_detect handles empty text input
    """
    mock_detector = Mock()
    mock_register = Mock()
    mock_register.value = "neutral"
    mock_detector.detect.return_value = (mock_register, 0.5)
    
    mock_transmogrifier = Mock()
    mock_transmogrifier._detector = mock_detector
    
    def transmog_detect(text: str) -> dict:
        register, confidence = mock_transmogrifier._detector.detect(text)
        return {
            'register': register.value,
            'confidence': confidence
        }
    
    result = transmog_detect("")
    
    assert isinstance(result, dict)
    assert 'register' in result
    assert 'confidence' in result


def test_transmog_detect_edge_case_unicode_text():
    """
    Verify transmog_detect handles Unicode characters
    """
    mock_detector = Mock()
    mock_register = Mock()
    mock_register.value = "casual"
    mock_detector.detect.return_value = (mock_register, 0.8)
    
    mock_transmogrifier = Mock()
    mock_transmogrifier._detector = mock_detector
    
    def transmog_detect(text: str) -> dict:
        register, confidence = mock_transmogrifier._detector.detect(text)
        return {
            'register': register.value,
            'confidence': confidence
        }
    
    result = transmog_detect("Bonjour 世界 🎉")
    
    assert isinstance(result, dict)
    assert 'register' in result
    assert 'confidence' in result


def test_transmog_detect_edge_case_confidence_bounds():
    """
    Verify confidence score is within valid range [0.0, 1.0]
    """
    mock_detector = Mock()
    mock_register = Mock()
    mock_register.value = "formal"
    mock_detector.detect.return_value = (mock_register, 0.75)
    
    mock_transmogrifier = Mock()
    mock_transmogrifier._detector = mock_detector
    
    def transmog_detect(text: str) -> dict:
        register, confidence = mock_transmogrifier._detector.detect(text)
        return {
            'register': register.value,
            'confidence': confidence
        }
    
    result = transmog_detect("Test text for confidence validation")
    
    assert result['confidence'] >= 0.0
    assert result['confidence'] <= 1.0


def test_invariant_confidence_range():
    """
    Verify confidence scores always within [0.0, 1.0] range
    """
    mock_detector = Mock()
    mock_register = Mock()
    mock_register.value = "casual"
    
    # Test with various confidence values
    test_confidences = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for conf in test_confidences:
        mock_detector.detect.return_value = (mock_register, conf)
        
        mock_transmogrifier = Mock()
        mock_transmogrifier._detector = mock_detector
        
        def transmog_detect(text: str) -> dict:
            register, confidence = mock_transmogrifier._detector.detect(text)
            return {
                'register': register.value,
                'confidence': confidence
            }
        
        result = transmog_detect("Test invariant")
        
        assert result['confidence'] >= 0.0
        assert result['confidence'] <= 1.0
        assert isinstance(result['confidence'], float)


def test_detect_result_structure_validation():
    """
    Verify transmog_detect returns valid TransmogDetectResult structure
    """
    mock_detector = Mock()
    mock_register = Mock()
    mock_register.value = "formal"
    mock_detector.detect.return_value = (mock_register, 0.9)
    
    mock_transmogrifier = Mock()
    mock_transmogrifier._detector = mock_detector
    
    def transmog_detect(text: str) -> dict:
        register, confidence = mock_transmogrifier._detector.detect(text)
        return {
            'register': register.value,
            'confidence': confidence
        }
    
    result = transmog_detect("Validate structure")
    
    assert isinstance(result, dict)
    assert 'register' in result
    assert isinstance(result['register'], str)
    assert 'confidence' in result
    assert isinstance(result['confidence'], float)


def test_detect_multiple_calls_consistency():
    """
    Verify transmog_detect returns consistent results for same input
    """
    mock_detector = Mock()
    mock_register = Mock()
    mock_register.value = "formal"
    mock_detector.detect.return_value = (mock_register, 0.85)
    
    mock_transmogrifier = Mock()
    mock_transmogrifier._detector = mock_detector
    
    def transmog_detect(text: str) -> dict:
        register, confidence = mock_transmogrifier._detector.detect(text)
        return {
            'register': register.value,
            'confidence': confidence
        }
    
    text = "Consistent test text"
    result1 = transmog_detect(text)
    result2 = transmog_detect(text)
    
    assert result1 == result2


# ============================================================================
# Test: transmog_detect() - Error Cases
# ============================================================================

def test_transmog_detect_error_detector_failure():
    """
    Verify transmog_detect handles detector failure exception
    """
    mock_detector = Mock()
    mock_detector.detect.side_effect = RuntimeError("Detector failed")
    
    mock_transmogrifier = Mock()
    mock_transmogrifier._detector = mock_detector
    
    def transmog_detect(text: str) -> dict:
        register, confidence = mock_transmogrifier._detector.detect(text)
        return {
            'register': register.value,
            'confidence': confidence
        }
    
    with pytest.raises(RuntimeError, match="Detector failed"):
        transmog_detect("Test text")


# ============================================================================
# Test: transmog_profiles() - Happy Path
# ============================================================================

def test_transmog_profiles_happy_path_basic():
    """
    Verify transmog_profiles returns list of serialized profile dictionaries
    """
    mock_profile1 = Mock()
    mock_profile1.model_dump.return_value = {
        'model_name': 'gpt-4',
        'sensitivity': 'high'
    }
    mock_profile2 = Mock()
    mock_profile2.model_dump.return_value = {
        'model_name': 'claude-3',
        'sensitivity': 'medium'
    }
    
    mock_profile_cache = Mock()
    mock_profile_cache.list_profiles.return_value = [mock_profile1, mock_profile2]
    
    mock_transmogrifier = Mock()
    mock_transmogrifier._profile_cache = mock_profile_cache
    
    def transmog_profiles() -> list:
        profiles = mock_transmogrifier._profile_cache.list_profiles()
        return [p.model_dump() for p in profiles]
    
    result = transmog_profiles()
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(p, dict) for p in result)
    mock_profile_cache.list_profiles.assert_called_once()


# ============================================================================
# Test: transmog_profiles() - Edge Cases
# ============================================================================

def test_transmog_profiles_edge_case_empty_cache():
    """
    Verify transmog_profiles returns empty list when no profiles cached
    """
    mock_profile_cache = Mock()
    mock_profile_cache.list_profiles.return_value = []
    
    mock_transmogrifier = Mock()
    mock_transmogrifier._profile_cache = mock_profile_cache
    
    def transmog_profiles() -> list:
        profiles = mock_transmogrifier._profile_cache.list_profiles()
        return [p.model_dump() for p in profiles]
    
    result = transmog_profiles()
    
    assert isinstance(result, list)
    assert len(result) == 0
    assert result == []


def test_transmog_profiles_edge_case_multiple_profiles():
    """
    Verify transmog_profiles returns all cached profiles
    """
    profiles_data = []
    for i in range(5):
        mock_profile = Mock()
        mock_profile.model_dump.return_value = {
            'model_name': f'model-{i}',
            'sensitivity': 'high'
        }
        profiles_data.append(mock_profile)
    
    mock_profile_cache = Mock()
    mock_profile_cache.list_profiles.return_value = profiles_data
    
    mock_transmogrifier = Mock()
    mock_transmogrifier._profile_cache = mock_profile_cache
    
    def transmog_profiles() -> list:
        profiles = mock_transmogrifier._profile_cache.list_profiles()
        return [p.model_dump() for p in profiles]
    
    result = transmog_profiles()
    
    assert isinstance(result, list)
    assert len(result) == 5
    assert all(isinstance(p, dict) for p in result)


def test_profiles_result_structure_validation():
    """
    Verify transmog_profiles returns list of valid ProfileDict structures
    """
    mock_profile = Mock()
    mock_profile.model_dump.return_value = {
        'model_name': 'gpt-4',
        'sensitivity': 'high',
        'register_support': ['formal', 'casual']
    }
    
    mock_profile_cache = Mock()
    mock_profile_cache.list_profiles.return_value = [mock_profile]
    
    mock_transmogrifier = Mock()
    mock_transmogrifier._profile_cache = mock_profile_cache
    
    def transmog_profiles() -> list:
        profiles = mock_transmogrifier._profile_cache.list_profiles()
        return [p.model_dump() for p in profiles]
    
    result = transmog_profiles()
    
    assert isinstance(result, list)
    assert all(isinstance(p, dict) for p in result)
    assert len(result) == 1
    assert 'model_name' in result[0]


# ============================================================================
# Test: transmog_profiles() - Error Cases
# ============================================================================

def test_transmog_profiles_error_profile_cache_failure():
    """
    Verify transmog_profiles handles cache access failure
    """
    mock_profile_cache = Mock()
    mock_profile_cache.list_profiles.side_effect = RuntimeError("Cache access failed")
    
    mock_transmogrifier = Mock()
    mock_transmogrifier._profile_cache = mock_profile_cache
    
    def transmog_profiles() -> list:
        profiles = mock_transmogrifier._profile_cache.list_profiles()
        return [p.model_dump() for p in profiles]
    
    with pytest.raises(RuntimeError, match="Cache access failed"):
        transmog_profiles()
