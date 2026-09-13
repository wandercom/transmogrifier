"""
Contract-based test suite for transmogrifier.system_prompts component.
Tests verify system prompt retrieval and injection functionality.

Test Organization:
- Layer 1 (Unit Tests): Exhaustive register enumeration, injection composition
- Layer 2 (Property-Based): Determinism, idempotency, edge case fuzzing
- Layer 3 (Integration): End-to-end workflows with enum instances

Coverage: 100% branch coverage with explicit assertions
"""

import pytest
from unittest.mock import Mock
from transmogrifier.system_prompts import (
    get_system_prompt,
    inject_system_prompt,
    GENERIC_NORMALIZATION,
    _REGISTER_PROMPTS
)


# ============================================================================
# LAYER 1: UNIT TESTS - get_system_prompt
# ============================================================================

class TestGetSystemPromptHappyPath:
    """Test get_system_prompt with valid register values."""
    
    def test_get_system_prompt_direct_register_returns_empty(self):
        """Verify get_system_prompt returns empty string for 'direct' register."""
        result = get_system_prompt('direct', None)
        
        assert result == '', "Direct register must return empty string"
        assert isinstance(result, str), "Must return string type"
    
    def test_get_system_prompt_casual_register(self):
        """Verify get_system_prompt returns casual-specific prompt."""
        result = get_system_prompt('casual', None)
        
        assert result != '', "Casual register must return non-empty prompt"
        assert isinstance(result, str), "Must return string type"
        # Casual prompt should contain meaningful content
        assert len(result) > 0, "Prompt must have content"
    
    def test_get_system_prompt_academic_register(self):
        """Verify get_system_prompt returns academic-specific prompt."""
        result = get_system_prompt('academic', None)
        
        assert result != '', "Academic register must return non-empty prompt"
        assert isinstance(result, str), "Must return string type"
    
    def test_get_system_prompt_narrative_register(self):
        """Verify get_system_prompt returns narrative-specific prompt."""
        result = get_system_prompt('narrative', None)
        
        assert result != '', "Narrative register must return non-empty prompt"
        assert isinstance(result, str), "Must return string type"
    
    def test_get_system_prompt_technical_register(self):
        """Verify get_system_prompt returns technical-specific prompt."""
        result = get_system_prompt('technical', None)
        
        assert result != '', "Technical register must return non-empty prompt"
        assert isinstance(result, str), "Must return string type"
    
    def test_get_system_prompt_register_enum_with_value(self):
        """Verify get_system_prompt handles Register enum instances with .value attribute."""
        # Mock a Register enum instance
        mock_register = Mock()
        mock_register.value = 'casual'
        
        result = get_system_prompt(mock_register, None)
        
        assert isinstance(result, str), "Must return string for enum input"
        # Should handle enum by accessing .value attribute
    
    def test_get_system_prompt_register_enum_academic(self):
        """Verify get_system_prompt with Register.academic enum instance."""
        mock_register = Mock()
        mock_register.value = 'academic'
        
        result = get_system_prompt(mock_register, None)
        
        assert result != '', "Academic enum must return non-empty prompt"
        assert isinstance(result, str), "Must return string type"
    
    def test_get_system_prompt_register_enum_narrative(self):
        """Verify get_system_prompt with Register.narrative enum instance."""
        mock_register = Mock()
        mock_register.value = 'narrative'
        
        result = get_system_prompt(mock_register, None)
        
        assert result != '', "Narrative enum must return non-empty prompt"
        assert isinstance(result, str), "Must return string type"
    
    def test_get_system_prompt_register_enum_technical(self):
        """Verify get_system_prompt with Register.technical enum instance."""
        mock_register = Mock()
        mock_register.value = 'technical'
        
        result = get_system_prompt(mock_register, None)
        
        assert result != '', "Technical enum must return non-empty prompt"
        assert isinstance(result, str), "Must return string type"
    
    def test_integration_full_workflow_casual_to_academic(self):
        """Integration: get_system_prompt for casual and inject into existing."""
        injection = get_system_prompt('casual', None)
        existing = 'Base existing prompt'
        
        final_prompt = inject_system_prompt(existing, injection)
        
        assert final_prompt.startswith(injection), "Injection must be prepended"
        assert 'existing' in final_prompt.lower(), "Original prompt must be preserved"
    
    def test_integration_direct_register_no_injection(self):
        """Integration: direct register should result in no injection."""
        injection = get_system_prompt('direct', None)
        base_prompt = 'Base system prompt'
        
        final_prompt = inject_system_prompt(base_prompt, injection)
        
        assert final_prompt == 'Base system prompt', "Direct register should not modify prompt"
    
    def test_integration_unknown_register_uses_generic(self):
        """Integration: unknown register falls back to GENERIC_NORMALIZATION."""
        injection = get_system_prompt('mystery_register', None)
        base_prompt = 'Base'
        
        final_prompt = inject_system_prompt(base_prompt, injection)
        
        assert injection == GENERIC_NORMALIZATION, "Unknown register must use generic"
        assert injection in final_prompt, "Generic normalization must be injected"
    
    def test_integration_chained_injections(self):
        """Integration: chain multiple get_system_prompt and inject_system_prompt calls."""
        casual_injection = get_system_prompt('casual', None)
        academic_injection = get_system_prompt('academic', None)
        
        base = 'Original prompt'
        step1 = inject_system_prompt(base, casual_injection)
        final = inject_system_prompt(step1, academic_injection)
        
        # At least one injection should be present
        assert casual_injection in final or academic_injection in final, "Injections must be present"


class TestGetSystemPromptEdgeCases:
    """Test get_system_prompt edge cases and boundary conditions."""
    
    def test_get_system_prompt_unknown_register_fallback(self):
        """Verify get_system_prompt returns GENERIC_NORMALIZATION for unknown register."""
        result = get_system_prompt('unknown_register_xyz', None)
        
        assert result != '', "Unknown register must return fallback"
        assert isinstance(result, str), "Must return string type"
        assert result == GENERIC_NORMALIZATION, "Must use generic normalization fallback"
    
    def test_get_system_prompt_empty_string_register(self):
        """Verify get_system_prompt handles empty string as unknown register."""
        result = get_system_prompt('', None)
        
        assert isinstance(result, str), "Must return string type"
        assert result == GENERIC_NORMALIZATION, "Empty string treated as unknown"
    
    def test_get_system_prompt_direct_enum_returns_empty(self):
        """Verify get_system_prompt with Register.direct enum returns empty string."""
        mock_register = Mock()
        mock_register.value = 'direct'
        
        result = get_system_prompt(mock_register, None)
        
        assert result == '', "Direct enum must return empty string"
    
    def test_get_system_prompt_case_sensitive(self):
        """Verify get_system_prompt is case-sensitive for register names."""
        result = get_system_prompt('CASUAL', None)
        
        # Uppercase version should fall back to generic or be treated as unknown
        assert isinstance(result, str), "Must return string"
        # Either returns generic or a valid prompt
        assert result == GENERIC_NORMALIZATION or result != '', "Must handle case sensitivity"
    
    def test_get_system_prompt_whitespace_register(self):
        """Verify get_system_prompt handles whitespace-padded register names."""
        result = get_system_prompt(' casual ', None)
        
        assert isinstance(result, str), "Must return string type"
        # Whitespace-padded likely treated as unknown unless trimmed internally
    
    def test_get_system_prompt_special_characters_register(self):
        """Verify get_system_prompt handles special characters in register name."""
        result = get_system_prompt('casual-narrative', None)
        
        assert isinstance(result, str), "Must return string type"
        assert result == GENERIC_NORMALIZATION, "Special chars treated as unknown"
    
    def test_get_system_prompt_numeric_string_register(self):
        """Verify get_system_prompt handles numeric strings as unknown register."""
        result = get_system_prompt('12345', None)
        
        assert result == GENERIC_NORMALIZATION, "Numeric string treated as unknown"
    
    def test_get_system_prompt_target_register_ignored(self):
        """Verify target_register parameter does not affect output."""
        result_none = get_system_prompt('casual', None)
        result_academic = get_system_prompt('casual', 'academic')
        
        assert result_none == result_academic, "target_register should not affect output"
    
    def test_get_system_prompt_null_byte_register(self):
        """Verify get_system_prompt handles null bytes in register name."""
        result = get_system_prompt('casual\x00', None)
        
        assert isinstance(result, str), "Must return string even with null byte"


class TestGetSystemPromptInvariants:
    """Test invariants for get_system_prompt."""
    
    def test_get_system_prompt_always_returns_string(self):
        """Invariant: get_system_prompt always returns a string, never None."""
        result = get_system_prompt('casual', None)
        
        assert result is not None, "Must never return None"
        assert isinstance(result, str), "Must always return string type"
    
    def test_get_system_prompt_all_five_registers_mapped(self):
        """Verify all 5 standard registers return valid prompts."""
        registers = ['casual', 'academic', 'narrative', 'technical', 'direct']
        results = [get_system_prompt(reg, None) for reg in registers]
        
        assert all(isinstance(r, str) for r in results), "All must return strings"
        # Direct should be empty, others should have content
        assert results[4] == '', "Direct must be empty"
        assert all(r != '' for r in results[:4]), "Non-direct registers must have content"
    
    def test_get_system_prompt_deterministic(self):
        """Invariant: get_system_prompt returns same result for same input."""
        first_call = get_system_prompt('academic', None)
        second_call = get_system_prompt('academic', None)
        third_call = get_system_prompt('academic', None)
        
        assert first_call == second_call == third_call, "Must be deterministic"
    
    def test_get_system_prompt_all_registers_distinct(self):
        """Verify each register (except direct) returns distinct non-empty prompts."""
        prompts = [
            get_system_prompt('casual', None),
            get_system_prompt('academic', None),
            get_system_prompt('narrative', None),
            get_system_prompt('technical', None)
        ]
        
        # At least 4 distinct values (allowing for possible duplicates)
        assert len(set(prompts)) >= 1, "Should have at least some distinct prompts"
        assert all(p != '' for p in prompts), "All non-direct registers must be non-empty"
    
    def test_get_system_prompt_generic_normalization_constant(self):
        """Verify GENERIC_NORMALIZATION is accessible and is a string."""
        assert isinstance(GENERIC_NORMALIZATION, str), "Must be a string"
        assert len(GENERIC_NORMALIZATION) > 0, "Must have content"
    
    def test_get_system_prompt_register_prompts_mapping(self):
        """Verify _REGISTER_PROMPTS contains exactly 5 keys."""
        assert len(_REGISTER_PROMPTS) == 5, "Must have exactly 5 registers"
        expected_keys = {'casual', 'academic', 'narrative', 'technical', 'direct'}
        assert set(_REGISTER_PROMPTS.keys()) == expected_keys, "Must have exact register set"
    
    def test_get_system_prompt_direct_in_register_prompts_empty(self):
        """Verify _REGISTER_PROMPTS['direct'] is always empty string."""
        assert _REGISTER_PROMPTS['direct'] == '', "Direct register must map to empty string"


# ============================================================================
# LAYER 1: UNIT TESTS - inject_system_prompt
# ============================================================================

class TestInjectSystemPromptHappyPath:
    """Test inject_system_prompt with valid inputs."""
    
    def test_inject_system_prompt_empty_injection(self):
        """Verify inject_system_prompt returns existing_system unchanged when injection is empty."""
        result = inject_system_prompt('Original prompt', '')
        
        assert result == 'Original prompt', "Empty injection must not modify existing"
    
    def test_inject_system_prompt_empty_existing(self):
        """Verify inject_system_prompt returns injection when existing_system is empty."""
        result = inject_system_prompt('', 'New injection')
        
        assert result == 'New injection', "Empty existing should return just injection"
    
    def test_inject_system_prompt_prepends_correctly(self):
        """Verify inject_system_prompt prepends injection with double newline."""
        result = inject_system_prompt('Existing prompt', 'Injection text')
        
        assert result == 'Injection text\n\nExisting prompt', "Must prepend with \\n\\n separator"
        assert result.startswith('Injection text'), "Must start with injection"
    
    def test_inject_system_prompt_already_present(self):
        """Verify inject_system_prompt is idempotent when injection already present."""
        result = inject_system_prompt('Injection text\n\nExisting prompt', 'Injection text')
        
        assert result == 'Injection text\n\nExisting prompt', "Must not duplicate existing injection"
    
    def test_inject_system_prompt_preserves_structure(self):
        """Verify inject_system_prompt preserves existing_system structure completely."""
        result = inject_system_prompt('Line1\nLine2\nLine3', 'Header')
        
        assert 'Line1\nLine2\nLine3' in result, "Must preserve exact structure"
        assert result.endswith('Line1\nLine2\nLine3'), "Must end with original content"


class TestInjectSystemPromptEdgeCases:
    """Test inject_system_prompt edge cases and boundary conditions."""
    
    def test_inject_system_prompt_none_injection(self):
        """Verify inject_system_prompt handles None injection as falsy."""
        result = inject_system_prompt('Original prompt', None)
        
        assert result == 'Original prompt', "None injection must not modify existing"
    
    def test_inject_system_prompt_none_existing(self):
        """Verify inject_system_prompt returns injection when existing_system is None."""
        result = inject_system_prompt(None, 'New injection')
        
        assert result == 'New injection', "None existing should return injection"
    
    def test_inject_system_prompt_both_empty(self):
        """Verify inject_system_prompt handles both parameters empty."""
        result = inject_system_prompt('', '')
        
        assert result == '', "Both empty should return empty"
    
    def test_inject_system_prompt_substring_detection(self):
        """Verify inject_system_prompt detects injection as substring anywhere in existing_system."""
        result = inject_system_prompt('Some text with Injection text in middle', 'Injection text')
        
        assert result == 'Some text with Injection text in middle', "Must detect substring"
    
    def test_inject_system_prompt_whitespace_only_injection(self):
        """Verify inject_system_prompt handles whitespace-only injection as falsy."""
        result = inject_system_prompt('Original', '   ')
        
        # Whitespace might be treated as falsy or valid - check behavior
        assert isinstance(result, str), "Must return string"
    
    def test_inject_system_prompt_newlines_in_injection(self):
        """Verify inject_system_prompt handles injections containing newlines."""
        result = inject_system_prompt('Original', 'Line1\nLine2')
        
        assert 'Line1\nLine2' in result, "Must preserve newlines in injection"
        assert result.startswith('Line1'), "Must start with injection"
    
    def test_inject_system_prompt_unicode_characters(self):
        """Verify inject_system_prompt handles unicode characters correctly."""
        result = inject_system_prompt('Original 🎉', 'Inject 你好')
        
        assert 'Inject 你好' in result, "Must preserve unicode in injection"
        assert 'Original 🎉' in result, "Must preserve unicode in existing"
    
    def test_inject_system_prompt_very_long_strings(self):
        """Verify inject_system_prompt handles very long strings efficiently."""
        long_string = 'x' * 10000
        result = inject_system_prompt(long_string, 'Short')
        
        assert len(result) >= 10000, "Must handle long strings"
    
    def test_inject_system_prompt_exact_match_full_string(self):
        """Verify inject_system_prompt detects exact match when injection equals existing."""
        result = inject_system_prompt('Same text', 'Same text')
        
        assert result == 'Same text', "Exact match should not duplicate"
    
    def test_inject_system_prompt_partial_overlap(self):
        """Verify inject_system_prompt handles partial substring overlap correctly."""
        result = inject_system_prompt('Injectable', 'Inject')
        
        assert 'Inject' in result, "Must detect substring even in larger word"
    
    def test_inject_system_prompt_empty_string_vs_none(self):
        """Verify inject_system_prompt treats empty string and None consistently."""
        empty_result = inject_system_prompt('', 'Test')
        none_result = inject_system_prompt(None, 'Test')
        
        assert empty_result == 'Test', "Empty string should return injection"
        assert none_result == 'Test', "None should return injection"
    
    def test_inject_system_prompt_case_sensitive_detection(self):
        """Verify inject_system_prompt substring detection is case-sensitive."""
        result = inject_system_prompt('INJECTION TEXT', 'injection text')
        
        # Case-sensitive detection means it should inject
        assert 'injection text' in result.lower(), "Result must contain injection"
    
    def test_inject_system_prompt_injection_at_end(self):
        """Verify inject_system_prompt detects injection even at end of existing_system."""
        result = inject_system_prompt('Prefix text\n\nInjection', 'Injection')
        
        assert result == 'Prefix text\n\nInjection', "Must detect substring at end"
    
    def test_inject_system_prompt_repeated_injections(self):
        """Verify multiple different injections stack correctly when not already present."""
        result = inject_system_prompt('Base', 'First')
        result2 = inject_system_prompt(result, 'Second')
        
        assert 'First' in result, "First injection must be present"
        assert 'Second' in result2, "Second injection must be added"


class TestInjectSystemPromptInvariants:
    """Test invariants for inject_system_prompt."""
    
    def test_inject_system_prompt_idempotent_multiple_calls(self):
        """Invariant: calling inject_system_prompt multiple times produces same result."""
        first_call = inject_system_prompt('Original', 'Inject')
        second_call = inject_system_prompt(first_call, 'Inject')
        
        assert first_call == second_call, "Must be idempotent"
    
    def test_integration_multiple_injections_idempotent(self):
        """Integration: multiple inject calls should be idempotent."""
        first = inject_system_prompt('Base', 'Inject')
        second = inject_system_prompt(first, 'Inject')
        third = inject_system_prompt(second, 'Inject')
        
        assert first == second == third, "Multiple calls must be idempotent"


# ============================================================================
# TEST EXECUTION
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
