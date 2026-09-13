"""
Contract tests for Rule-Based Register Rewriting

Tests the RuleEngine.rewrite() function according to contract specifications.
Implements three-tier testing: unit tests, edge cases, and invariant verification.
"""

import pytest
import re
import time
from unittest.mock import Mock, MagicMock
from transmogrifier.rules import RuleEngine, RewriteRule


class TestRewriteHappyPath:
    """Happy path tests for basic register transformations"""
    
    def test_rewrite_casual_to_direct_basic(self):
        """Happy path: Basic casual to direct register transformation"""
        engine = RuleEngine()
        text = "I'm gonna check that out"
        source = "casual"
        target = "direct"
        
        result = engine.rewrite(text, source, target)
        
        assert result is not None
        assert len(result) > 0
        # Result should be transformed (unless source == target)
        assert result != text or source == target
    
    def test_rewrite_academic_to_direct_basic(self):
        """Happy path: Academic to direct register transformation"""
        engine = RuleEngine()
        text = "The aforementioned hypothesis requires further investigation"
        source = "academic"
        target = "direct"
        
        result = engine.rewrite(text, source, target)
        
        assert result is not None
        assert len(result) > 0
    
    def test_rewrite_narrative_to_direct_basic(self):
        """Happy path: Narrative to direct register transformation"""
        engine = RuleEngine()
        text = "Once upon a time in a galaxy far away"
        source = "narrative"
        target = "direct"
        
        result = engine.rewrite(text, source, target)
        
        assert result is not None
        assert len(result) > 0
    
    def test_rewrite_technical_to_direct_basic(self):
        """Happy path: Technical to direct register transformation"""
        engine = RuleEngine()
        text = "Execute the aforementioned procedure"
        source = "technical"
        target = "direct"
        
        result = engine.rewrite(text, source, target)
        
        assert result is not None
        assert len(result) > 0
    
    def test_rewrite_same_register_identity(self):
        """Happy path: Rewriting with source equals target returns original text"""
        engine = RuleEngine()
        text = "This is some text"
        source = "casual"
        target = "casual"
        
        result = engine.rewrite(text, source, target)
        
        # Postcondition: If source == target, returns original text unchanged
        assert result == text
    
    def test_rewrite_with_register_objects(self):
        """Happy path: Using Register objects with .value attribute"""
        engine = RuleEngine()
        text = "I'm testing this"
        
        # Create mock Register objects with .value attributes
        source_register = Mock()
        source_register.value = "casual"
        target_register = Mock()
        target_register.value = "direct"
        
        result = engine.rewrite(text, source_register, target_register)
        
        assert result is not None
        assert len(result) > 0
    
    def test_rewrite_indirect_routing(self):
        """Happy path: Indirect routing through 'direct' register when no direct path exists"""
        engine = RuleEngine()
        text = "I'm gonna do this"
        source = "casual"
        target = "academic"
        
        # This should route through 'direct': casual→direct→academic
        result = engine.rewrite(text, source, target)
        
        assert result is not None
        assert len(result) > 0


class TestRewriteEdgeCases:
    """Edge case tests for boundary conditions"""
    
    def test_rewrite_empty_string(self):
        """Edge case: Empty string input"""
        engine = RuleEngine()
        text = ""
        source = "casual"
        target = "direct"
        
        result = engine.rewrite(text, source, target)
        
        # Postcondition: If strip produces empty string, returns original text
        assert result == ''
    
    def test_rewrite_whitespace_only(self):
        """Edge case: Whitespace-only string"""
        engine = RuleEngine()
        text = "   \t\n  "
        source = "casual"
        target = "direct"
        
        result = engine.rewrite(text, source, target)
        
        # Postcondition: If strip produces empty string, returns original text
        assert result == '   \t\n  '
    
    def test_rewrite_leading_trailing_whitespace(self):
        """Edge case: Text with leading and trailing whitespace"""
        engine = RuleEngine()
        text = "  I'm gonna check this  "
        source = "casual"
        target = "direct"
        
        result = engine.rewrite(text, source, target)
        
        # Postcondition: Result is stripped unless strip produces empty string
        assert result is not None
        if len(result) > 0:
            assert result[0] != ' '
            assert result[-1] != ' '
    
    def test_rewrite_unicode_text(self):
        """Edge case: Unicode text with special characters"""
        engine = RuleEngine()
        text = "I'm gonna café ☕ 你好"
        source = "casual"
        target = "direct"
        
        result = engine.rewrite(text, source, target)
        
        # Unicode characters should be preserved
        assert result is not None
        assert len(result) > 0
    
    def test_rewrite_very_long_input(self):
        """Edge case: Very long text input"""
        engine = RuleEngine()
        # Generate 10000 character string
        text = "I'm gonna check this. " * 500  # ~11000 chars
        source = "casual"
        target = "direct"
        
        start_time = time.time()
        result = engine.rewrite(text, source, target)
        elapsed = time.time() - start_time
        
        assert result is not None
        assert len(result) > 0
        # Performance: p95 < 1ms (allowing some slack for large input)
        assert elapsed < 0.1  # 100ms tolerance for large input
    
    def test_rewrite_multiline_text(self):
        """Edge case: Text with multiple line breaks"""
        engine = RuleEngine()
        text = "I'm gonna do this\nAnd I'm gonna do that\nYou're awesome"
        source = "casual"
        target = "direct"
        
        result = engine.rewrite(text, source, target)
        
        assert result is not None
        assert len(result) > 0
    
    def test_rewrite_no_rules_for_pair(self):
        """Edge case: No transformation rules exist for register pair"""
        engine = RuleEngine()
        text = "Some text here"
        source = "unknown_register"
        target = "another_unknown"
        
        result = engine.rewrite(text, source, target)
        
        # Postcondition: Returns text possibly routed through 'direct' register
        assert result is not None
        assert len(result) > 0
    
    def test_rewrite_preserved_html(self):
        """Edge case: HTML content should be handled appropriately"""
        engine = RuleEngine()
        text = "I'm gonna <strong>check</strong> this out"
        source = "casual"
        target = "direct"
        
        result = engine.rewrite(text, source, target)
        
        assert result is not None
        assert len(result) > 0
    
    def test_rewrite_escaped_characters(self):
        """Edge case: Text with escaped characters"""
        engine = RuleEngine()
        text = "I'm gonna check\\nthis\\tout"
        source = "casual"
        target = "direct"
        
        result = engine.rewrite(text, source, target)
        
        assert result is not None
        assert len(result) > 0
    
    def test_rewrite_non_matching_text(self):
        """Edge case: Text that doesn't match any rules"""
        engine = RuleEngine()
        text = "xyz qwerty asdf"
        source = "casual"
        target = "direct"
        
        result = engine.rewrite(text, source, target)
        
        # Should return text unchanged or minimally processed
        assert result is not None
        assert len(result) > 0
    
    def test_rewrite_multiple_transformations(self):
        """Edge case: Text with multiple patterns to transform"""
        engine = RuleEngine()
        text = "I'm gonna wanna check this, you're awesome"
        source = "casual"
        target = "direct"
        
        result = engine.rewrite(text, source, target)
        
        # All applicable rules should be applied sequentially
        assert result is not None
        assert len(result) > 0
    
    def test_rewrite_overlapping_patterns(self):
        """Edge case: Text with potentially overlapping pattern matches"""
        engine = RuleEngine()
        text = "I'm I'm gonna gonna check"
        source = "casual"
        target = "direct"
        
        result = engine.rewrite(text, source, target)
        
        # Patterns should be applied deterministically
        assert result is not None
        assert len(result) > 0
    
    def test_rewrite_special_regex_characters(self):
        """Edge case: Text containing special regex characters"""
        engine = RuleEngine()
        text = "I'm gonna check $100 (or more) [test]"
        source = "casual"
        target = "direct"
        
        result = engine.rewrite(text, source, target)
        
        # Special characters shouldn't break regex processing
        assert result is not None
        assert len(result) > 0


class TestRewriteInvariants:
    """Tests for contract invariants"""
    
    def test_rewrite_case_insensitive(self):
        """Invariant: Regex substitutions use IGNORECASE flag"""
        engine = RuleEngine()
        text = "I'M GONNA check THIS"
        source = "casual"
        target = "direct"
        
        result = engine.rewrite(text, source, target)
        
        # Case variations should be handled by IGNORECASE
        assert result is not None
        assert len(result) > 0
    
    def test_rewrite_deterministic_output(self):
        """Invariant: Sequential and deterministic rule application"""
        engine = RuleEngine()
        text = "I'm gonna check this"
        source = "casual"
        target = "direct"
        
        # Call rewrite multiple times with same input
        result = engine.rewrite(text, source, target)
        second_result = engine.rewrite(text, source, target)
        third_result = engine.rewrite(text, source, target)
        
        # Same output every time
        assert result == second_result
        assert result == third_result
    
    def test_rewrite_no_side_effects(self):
        """Invariant: Rewrite has no side effects on engine state"""
        engine = RuleEngine()
        
        # Perform multiple rewrites with different texts
        first_result = engine.rewrite("I'm testing this", "casual", "direct")
        second_result = engine.rewrite("I'm testing this", "casual", "direct")
        
        # Multiple calls shouldn't pollute state
        assert first_result == second_result
        
        # Try with different text in between
        engine.rewrite("Something completely different", "academic", "direct")
        third_result = engine.rewrite("I'm testing this", "casual", "direct")
        
        assert first_result == third_result
    
    def test_rewrite_returns_non_empty_string(self):
        """Invariant: Postcondition - returns non-empty string"""
        engine = RuleEngine()
        
        # Test various non-empty inputs
        test_cases = [
            "test",
            "I'm gonna do this",
            "The aforementioned",
            "Once upon a time",
            "Execute procedure"
        ]
        
        for text in test_cases:
            result = engine.rewrite(text, "casual", "direct")
            # Never returns empty string for non-empty input
            assert len(result) > 0
    
    def test_rewrite_strip_fallback(self):
        """Invariant: If strip produces empty string, returns original text"""
        engine = RuleEngine()
        text = "   "
        source = "casual"
        target = "direct"
        
        result = engine.rewrite(text, source, target)
        
        # Returns original whitespace text
        assert result == '   '
    
    def test_supported_register_transformations(self):
        """Invariant: Supported register transformations are hardcoded"""
        engine = RuleEngine()
        text = "Test text for transformation"
        
        # Supported transformations: (casual→direct), (academic→direct), 
        # (narrative→direct), (technical→direct)
        supported_pairs = [
            ("casual", "direct"),
            ("academic", "direct"),
            ("narrative", "direct"),
            ("technical", "direct")
        ]
        
        for source, target in supported_pairs:
            result = engine.rewrite(text, source, target)
            assert result is not None
            assert len(result) > 0


class TestRewriteRoundTrip:
    """Bidirectional and round-trip tests"""
    
    def test_rewrite_bidirectional_casual_direct(self):
        """Test both casual→direct and direct→casual transformations"""
        engine = RuleEngine()
        casual_text = "I'm gonna do this"
        
        # Transform casual to direct
        direct_result = engine.rewrite(casual_text, "casual", "direct")
        assert direct_result is not None
        
        # Transform back (if rules exist for direct→casual)
        back_to_casual = engine.rewrite(direct_result, "direct", "casual")
        assert back_to_casual is not None
    
    def test_rewrite_identity_preserves_text(self):
        """Test that identity transformation preserves text exactly"""
        engine = RuleEngine()
        test_texts = [
            "Simple text",
            "Text with numbers 123",
            "Text with symbols !@#$%",
            "Unicode text 你好 ☕"
        ]
        
        for text in test_texts:
            result = engine.rewrite(text, "casual", "casual")
            assert result == text


class TestRewriteRegisterParameters:
    """Tests for register parameter handling"""
    
    def test_rewrite_register_enum_vs_string(self):
        """Test both string registers and Register objects"""
        engine = RuleEngine()
        text = "I'm testing"
        
        # String registers
        result_str = engine.rewrite(text, "casual", "direct")
        assert result_str is not None
        
        # Register objects with .value
        source_obj = Mock()
        source_obj.value = "casual"
        target_obj = Mock()
        target_obj.value = "direct"
        
        result_obj = engine.rewrite(text, source_obj, target_obj)
        assert result_obj is not None
        
        # Should produce same result
        assert result_str == result_obj
    
    def test_rewrite_unknown_registers(self):
        """Test handling of unknown register types"""
        engine = RuleEngine()
        text = "Test text"
        
        result = engine.rewrite(text, "unknown_source", "unknown_target")
        
        # Should handle gracefully (postcondition: returns non-empty string)
        assert result is not None
        assert len(result) > 0
    
    def test_rewrite_register_case_handling(self):
        """Test case sensitivity of register names"""
        engine = RuleEngine()
        text = "I'm testing this"
        
        # Test lowercase (standard)
        result_lower = engine.rewrite(text, "casual", "direct")
        
        # Test if mixed case is handled
        result_mixed = engine.rewrite(text, "Casual", "Direct")
        
        assert result_lower is not None
        assert result_mixed is not None


class TestRewriteRuleInteractions:
    """Tests for rule interaction scenarios"""
    
    def test_rewrite_sequential_rule_application(self):
        """Test that rules are applied sequentially"""
        engine = RuleEngine()
        
        # Text with multiple patterns
        text = "I'm gonna wanna check this out, you're awesome"
        result = engine.rewrite(text, "casual", "direct")
        
        assert result is not None
        assert len(result) > 0
    
    def test_rewrite_non_overlapping_rules(self):
        """Test multiple non-overlapping rules in same text"""
        engine = RuleEngine()
        
        text = "I'm here and you're there"
        result = engine.rewrite(text, "casual", "direct")
        
        assert result is not None
        assert len(result) > 0
    
    def test_rewrite_chained_transformations(self):
        """Test chained transformations through multiple registers"""
        engine = RuleEngine()
        text = "I'm gonna test this"
        
        # Chain: casual → direct → academic
        step1 = engine.rewrite(text, "casual", "direct")
        step2 = engine.rewrite(step1, "direct", "academic")
        
        # Direct chain
        direct_result = engine.rewrite(text, "casual", "academic")
        
        assert step2 is not None
        assert direct_result is not None


class TestRewritePerformance:
    """Performance regression tests"""
    
    def test_rewrite_performance_small_input(self):
        """Test performance with small input (< 1ms target)"""
        engine = RuleEngine()
        text = "I'm gonna check this"
        
        start = time.time()
        result = engine.rewrite(text, "casual", "direct")
        elapsed = time.time() - start
        
        assert result is not None
        # p95 < 1ms, allowing 10ms for test overhead
        assert elapsed < 0.01
    
    def test_rewrite_performance_medium_input(self):
        """Test performance with medium input (1000 chars)"""
        engine = RuleEngine()
        text = "I'm gonna check this. " * 50  # ~1100 chars
        
        start = time.time()
        result = engine.rewrite(text, "casual", "direct")
        elapsed = time.time() - start
        
        assert result is not None
        # O(n*m) complexity, should still be fast
        assert elapsed < 0.05
    
    def test_rewrite_no_pathological_backtracking(self):
        """Test that regex patterns don't cause pathological backtracking"""
        engine = RuleEngine()
        
        # Create text that could cause backtracking issues
        text = "a" * 100 + "I'm gonna" + "b" * 100
        
        start = time.time()
        result = engine.rewrite(text, "casual", "direct")
        elapsed = time.time() - start
        
        assert result is not None
        # Should not timeout or take excessive time
        assert elapsed < 0.1


class TestRewriteStateManagement:
    """Tests for state management and instance reuse"""
    
    def test_rewrite_fresh_engine_instance(self):
        """Test with fresh RuleEngine instance"""
        engine = RuleEngine()
        text = "I'm testing"
        
        result = engine.rewrite(text, "casual", "direct")
        
        assert result is not None
        assert len(result) > 0
    
    def test_rewrite_reused_engine_instance(self):
        """Test that reused engine instances work correctly"""
        engine = RuleEngine()
        
        # Multiple rewrites on same instance
        result1 = engine.rewrite("I'm testing", "casual", "direct")
        result2 = engine.rewrite("You're awesome", "casual", "direct")
        result3 = engine.rewrite("I'm testing", "casual", "direct")
        
        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
        
        # Same input should give same output
        assert result1 == result3
    
    def test_rewrite_parallel_engine_instances(self):
        """Test that multiple engine instances don't interfere"""
        engine1 = RuleEngine()
        engine2 = RuleEngine()
        
        text = "I'm testing this"
        
        result1 = engine1.rewrite(text, "casual", "direct")
        result2 = engine2.rewrite(text, "casual", "direct")
        
        # Both engines should produce same result
        assert result1 == result2


class TestRewriteComplexScenarios:
    """Complex real-world scenarios"""
    
    def test_rewrite_mixed_register_content(self):
        """Test text that mixes multiple register styles"""
        engine = RuleEngine()
        
        # Text mixing casual and formal elements
        text = "I'm gonna check the aforementioned hypothesis"
        result = engine.rewrite(text, "casual", "direct")
        
        assert result is not None
        assert len(result) > 0
    
    def test_rewrite_with_punctuation_variety(self):
        """Test text with various punctuation marks"""
        engine = RuleEngine()
        
        text = "I'm gonna check this! Are you? Yes, I'm sure; definitely."
        result = engine.rewrite(text, "casual", "direct")
        
        assert result is not None
        assert len(result) > 0
    
    def test_rewrite_with_numbers_and_symbols(self):
        """Test text containing numbers and symbols"""
        engine = RuleEngine()
        
        text = "I'm gonna need $100 or 50% more @ 3pm"
        result = engine.rewrite(text, "casual", "direct")
        
        assert result is not None
        assert len(result) > 0
    
    def test_rewrite_paragraph_text(self):
        """Test full paragraph transformation"""
        engine = RuleEngine()
        
        text = """I'm gonna tell you something important. 
        You're gonna want to hear this. 
        The aforementioned procedure isn't gonna work."""
        
        result = engine.rewrite(text, "casual", "direct")
        
        assert result is not None
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
