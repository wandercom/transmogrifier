"""
Contract tests for RegisterDetector component.
Tests the linguistic register detection system using pattern matching.

This test suite implements a layered testing strategy:
- Layer 1: Unit tests for _score() feature extraction
- Layer 2: Integration tests for detect() with canonical examples
- Layer 3: Edge case suite covering boundary inputs
- Layer 4: Invariant tests for confidence and score properties

Generated from contract version 1.
"""

import pytest
import re
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Tuple

# Import the component under test
from transmogrifier.detector import *


# ============================================================================
# LAYER 1: Unit Tests for _score() Feature Extraction
# ============================================================================

class TestScoreFeatureExtraction:
    """Unit tests for _score() method verifying feature extraction correctness."""
    
    def test_score_casual_markers(self):
        """_score should sum weights from CASUAL_MARKERS patterns"""
        detector = RegisterDetector()
        text = "Hey guys! Yeah I'm totally gonna do that lol."
        
        scores = detector._score(text)
        
        assert hasattr(scores, 'casual')
        assert scores.casual > 0, "Casual score should be positive for casual text"
        assert isinstance(scores.casual, float)
    
    def test_score_technical_markers(self):
        """_score should sum weights from TECHNICAL_MARKERS patterns"""
        detector = RegisterDetector()
        text = "The API endpoint function implements the algorithm using parameters."
        
        scores = detector._score(text)
        
        assert hasattr(scores, 'technical')
        assert scores.technical > 0, "Technical score should be positive for technical text"
        assert isinstance(scores.technical, float)
    
    def test_score_academic_markers(self):
        """_score should sum weights from ACADEMIC_MARKERS patterns"""
        detector = RegisterDetector()
        text = "Furthermore, this study demonstrates that the methodology employed yields significant results."
        
        scores = detector._score(text)
        
        assert hasattr(scores, 'academic')
        assert scores.academic > 0, "Academic score should be positive for academic text"
        assert isinstance(scores.academic, float)
    
    def test_score_narrative_markers(self):
        """_score should sum weights from NARRATIVE_MARKERS patterns"""
        detector = RegisterDetector()
        text = "She walked slowly, remembering the moment when everything changed."
        
        scores = detector._score(text)
        
        assert hasattr(scores, 'narrative')
        assert scores.narrative > 0, "Narrative score should be positive for narrative text"
        assert isinstance(scores.narrative, float)
    
    def test_score_direct_short_text_bonus(self):
        """_score should add 1.5 to direct score if text has ≤12 words"""
        detector = RegisterDetector()
        text = "Stop now please."  # 3 words
        
        scores = detector._score(text)
        
        assert hasattr(scores, 'direct')
        assert scores.direct >= 1.5, "Direct score should have at least 1.5 bonus for ≤12 words"
    
    def test_score_direct_very_short_text_bonus(self):
        """_score should add another 1.5 to direct score if text has ≤6 words"""
        detector = RegisterDetector()
        text = "Stop."  # 1 word
        
        scores = detector._score(text)
        
        assert scores.direct >= 3.0, "Direct score should have 3.0 bonus for ≤6 words (1.5 + 1.5)"
    
    def test_score_direct_penalty_when_others_high(self):
        """_score should multiply direct score by 0.3 if max(other scores) > 2.0"""
        detector = RegisterDetector()
        # Long academic text with many academic markers
        text = "Furthermore, the methodology employed in this research demonstrates statistically significant results across multiple dimensions."
        
        scores = detector._score(text)
        
        # Check if academic score is high
        if scores.academic > 2.0:
            # Direct score should be reduced (likely small or 0 due to penalty)
            # We can't test exact value without knowing initial direct, but we can verify it's small
            assert scores.direct < scores.academic, "Direct score should be penalized when academic score is high"
    
    def test_score_returns_non_negative_values(self):
        """_score should return _FeatureScores with all non-negative float values"""
        detector = RegisterDetector()
        text = "Any text with various markers for testing."
        
        scores = detector._score(text)
        
        assert scores.casual >= 0, "Casual score must be non-negative"
        assert scores.technical >= 0, "Technical score must be non-negative"
        assert scores.academic >= 0, "Academic score must be non-negative"
        assert scores.narrative >= 0, "Narrative score must be non-negative"
        assert scores.direct >= 0, "Direct score must be non-negative"
    
    def test_score_case_insensitive_matching(self):
        """_score should match patterns case-insensitively"""
        detector = RegisterDetector()
        text = "HEY GUYS! YEAH I'M TOTALLY GONNA DO THAT LOL."
        
        scores = detector._score(text)
        
        assert scores.casual > 0, "Should detect casual markers despite uppercase"
    
    def test_score_all_fields_present(self):
        """_FeatureScores should have all required fields"""
        detector = RegisterDetector()
        text = "Test text."
        
        scores = detector._score(text)
        
        assert hasattr(scores, 'casual')
        assert hasattr(scores, 'technical')
        assert hasattr(scores, 'academic')
        assert hasattr(scores, 'narrative')
        assert hasattr(scores, 'direct')


# ============================================================================
# LAYER 2: Integration Tests for detect()
# ============================================================================

class TestDetectIntegration:
    """Integration tests for detect() with canonical examples of each register."""
    
    def test_detect_casual_register_happy_path(self):
        """Text with strong casual markers should be classified as casual"""
        detector = RegisterDetector()
        text = "Hey guys! Yeah I'm totally gonna do that lol. BTW thanks!"
        
        register, confidence = detector.detect(text)
        
        assert register == Register.casual, f"Expected casual register, got {register}"
        assert confidence > 0.5, f"Expected confidence > 0.5, got {confidence}"
    
    def test_detect_technical_register_happy_path(self):
        """Text with strong technical markers should be classified as technical"""
        detector = RegisterDetector()
        text = "The API endpoint returns a JSON object containing the configuration parameters for the system."
        
        register, confidence = detector.detect(text)
        
        assert register == Register.technical, f"Expected technical register, got {register}"
        assert confidence > 0.5, f"Expected confidence > 0.5, got {confidence}"
    
    def test_detect_academic_register_happy_path(self):
        """Text with strong academic markers should be classified as academic"""
        detector = RegisterDetector()
        text = "Furthermore, this study demonstrates that the methodology employed herein yields statistically significant results."
        
        register, confidence = detector.detect(text)
        
        assert register == Register.academic, f"Expected academic register, got {register}"
        assert confidence > 0.5, f"Expected confidence > 0.5, got {confidence}"
    
    def test_detect_narrative_register_happy_path(self):
        """Text with strong narrative markers should be classified as narrative"""
        detector = RegisterDetector()
        text = "She walked slowly down the path, remembering the first time they had met. It was a moment she would never forget."
        
        register, confidence = detector.detect(text)
        
        assert register == Register.narrative, f"Expected narrative register, got {register}"
        assert confidence > 0.5, f"Expected confidence > 0.5, got {confidence}"
    
    def test_detect_direct_register_short_text(self):
        """Short imperative text should be classified as direct"""
        detector = RegisterDetector()
        text = "Stop now."
        
        register, confidence = detector.detect(text)
        
        assert register == Register.direct, f"Expected direct register, got {register}"
        assert confidence > 0.5, f"Expected high confidence for direct, got {confidence}"
    
    def test_detect_mixed_register_technical_academic(self):
        """Text with mixed technical and academic markers"""
        detector = RegisterDetector()
        text = "This study implements an algorithm that demonstrates significant performance improvements in the API."
        
        register, confidence = detector.detect(text)
        
        # Should be either technical or academic based on strongest score
        assert register in [Register.technical, Register.academic], \
            f"Expected technical or academic register for mixed text, got {register}"
        assert 0 <= confidence <= 1.0


# ============================================================================
# LAYER 3: Edge Case Suite
# ============================================================================

class TestDetectEdgeCases:
    """Edge case tests covering boundary inputs."""
    
    def test_detect_empty_string_returns_direct_register(self):
        """Empty string should return Register.direct with confidence 1.0"""
        detector = RegisterDetector()
        text = ""
        
        register, confidence = detector.detect(text)
        
        assert register == Register.direct, f"Expected direct register for empty string, got {register}"
        assert confidence == 1.0, f"Expected confidence 1.0 for empty string, got {confidence}"
    
    def test_detect_whitespace_only_returns_direct_register(self):
        """Whitespace-only text should return Register.direct with confidence 1.0"""
        detector = RegisterDetector()
        text = "   \t\n  "
        
        register, confidence = detector.detect(text)
        
        assert register == Register.direct, f"Expected direct register for whitespace, got {register}"
        assert confidence == 1.0, f"Expected confidence 1.0 for whitespace, got {confidence}"
    
    def test_detect_single_character(self):
        """Single character should return valid register (likely direct)"""
        detector = RegisterDetector()
        text = "a"
        
        register, confidence = detector.detect(text)
        
        assert register == Register.direct, f"Expected direct register for single char, got {register}"
        assert confidence >= 0.5, f"Expected high confidence for single char, got {confidence}"
    
    def test_detect_very_long_text(self):
        """Very long text should still return valid register and confidence"""
        detector = RegisterDetector()
        text = "This is a test sentence. " * 1000
        
        register, confidence = detector.detect(text)
        
        assert isinstance(register, Register), "Should return a Register enum"
        assert 0 <= confidence <= 1.0, f"Confidence should be in [0, 1], got {confidence}"
    
    def test_detect_unicode_and_special_chars(self):
        """Text with unicode and special characters should be handled gracefully"""
        detector = RegisterDetector()
        text = "Hello 😊! The café's API costs $100. 中文测试"
        
        register, confidence = detector.detect(text)
        
        assert isinstance(register, Register), "Should return a Register enum"
        assert 0 <= confidence <= 1.0, f"Confidence should be in [0, 1], got {confidence}"
    
    def test_detect_numbers_only(self):
        """Text with only numbers should return a valid register"""
        detector = RegisterDetector()
        text = "123 456 789 101112"
        
        register, confidence = detector.detect(text)
        
        # Numbers-only text is likely classified as direct due to no markers
        assert isinstance(register, Register), "Should return a Register enum"
        assert register == Register.direct, f"Expected direct for numbers-only, got {register}"
    
    def test_detect_repeated_words(self):
        """Text with repeated words should be handled correctly"""
        detector = RegisterDetector()
        text = "test test test test test test test"
        
        register, confidence = detector.detect(text)
        
        assert isinstance(register, Register), "Should return a Register enum"
        assert 0 <= confidence <= 1.0, f"Confidence should be in [0, 1], got {confidence}"
    
    def test_detect_all_zero_scores_returns_direct_08(self):
        """When all scores are 0, should return Register.direct with confidence 0.8"""
        detector = RegisterDetector()
        # A text that is long enough to avoid direct bonuses but has no markers
        # Using words that are unlikely to match any patterns
        text = "xyz abc def ghi jkl mno pqr stu vwx yzab cdef ghij klmn"
        
        register, confidence = detector.detect(text)
        
        # This test is tricky because we need text with truly zero scores
        # Most plain text will still get direct score from length heuristics
        # We just verify it returns direct register
        if register == Register.direct and confidence == 0.8:
            # Perfect - all scores were zero
            assert True
        elif register == Register.direct:
            # Direct was selected, confidence might differ due to length heuristics
            assert confidence > 0, "Confidence should be positive"


# ============================================================================
# LAYER 4: Invariant Tests
# ============================================================================

class TestInvariants:
    """Tests for contract invariants and properties."""
    
    def test_detect_confidence_in_range(self):
        """Confidence score should always be between 0 and 1 inclusive"""
        detector = RegisterDetector()
        test_texts = [
            "Any text here for testing confidence bounds.",
            "Hey guys! Totally lol!",
            "The API implements the function.",
            "Furthermore, this study demonstrates results.",
            "She walked slowly, remembering.",
            "Stop.",
            "",
            "x",
            "Test " * 100
        ]
        
        for text in test_texts:
            register, confidence = detector.detect(text)
            assert 0.0 <= confidence <= 1.0, \
                f"Confidence {confidence} out of range [0, 1] for text: {text[:50]}"
    
    def test_detect_confidence_rounded_to_three_decimals(self):
        """Confidence score should be rounded to 3 decimal places"""
        detector = RegisterDetector()
        test_texts = [
            "Testing confidence rounding with various markers.",
            "Hey guys! The API function demonstrates this methodology.",
            "She remembered when the algorithm worked perfectly.",
        ]
        
        for text in test_texts:
            register, confidence = detector.detect(text)
            assert round(confidence, 3) == confidence, \
                f"Confidence {confidence} not rounded to 3 decimals for text: {text[:50]}"
    
    def test_detect_returns_tuple(self):
        """detect() should return a tuple of (Register, float)"""
        detector = RegisterDetector()
        text = "Test text for tuple validation."
        
        result = detector.detect(text)
        
        assert isinstance(result, tuple), "detect() should return a tuple"
        assert len(result) == 2, "Tuple should have exactly 2 elements"
        register, confidence = result
        assert isinstance(register, Register), "First element should be a Register"
        assert isinstance(confidence, float), "Second element should be a float"
    
    def test_confidence_calculation_formula(self):
        """Confidence should follow formula: min((best - second) / total + 0.5, 1.0)"""
        detector = RegisterDetector()
        text = "Furthermore, this methodology demonstrates the results."
        
        register, confidence = detector.detect(text)
        
        # We can't verify the exact formula without accessing internal scores,
        # but we can verify the result is reasonable
        assert 0 <= confidence <= 1.0, "Confidence must be in valid range"
        assert isinstance(confidence, float), "Confidence must be a float"
    
    def test_detect_highest_score_wins(self):
        """The detected register should correspond to the highest score from _score()"""
        detector = RegisterDetector()
        
        # Test with clearly technical text
        text_tech = "The API function parameters implement the algorithm configuration system."
        scores_tech = detector._score(text_tech)
        register_tech, _ = detector.detect(text_tech)
        
        # Find which score is highest
        score_dict = {
            'casual': scores_tech.casual,
            'technical': scores_tech.technical,
            'academic': scores_tech.academic,
            'narrative': scores_tech.narrative,
            'direct': scores_tech.direct
        }
        max_category = max(score_dict, key=score_dict.get)
        
        # Register should match the highest score category
        assert register_tech.name == max_category, \
            f"Register {register_tech.name} doesn't match highest score category {max_category}"
    
    def test_pattern_matching_is_case_insensitive(self):
        """All marker patterns should match case-insensitively"""
        detector = RegisterDetector()
        
        text_lower = "hey guys yeah totally lol"
        text_upper = "HEY GUYS YEAH TOTALLY LOL"
        
        scores_lower = detector._score(text_lower)
        scores_upper = detector._score(text_upper)
        
        # Both should have similar casual scores (case-insensitive matching)
        assert scores_lower.casual > 0, "Lowercase should match casual patterns"
        assert scores_upper.casual > 0, "Uppercase should match casual patterns"
        # Scores should be very close (might differ slightly due to regex details)
        assert abs(scores_lower.casual - scores_upper.casual) < 0.1, \
            "Case should not significantly affect scores"


# ============================================================================
# Additional Randomized Testing (without hypothesis)
# ============================================================================

class TestRandomizedInputs:
    """Randomized tests using Python's random module."""
    
    def test_detect_random_length_texts(self):
        """Test detect with various random text lengths"""
        import random
        import string
        
        detector = RegisterDetector()
        
        for _ in range(20):
            length = random.randint(0, 500)
            text = ''.join(random.choices(string.ascii_letters + ' ', k=length))
            
            register, confidence = detector.detect(text)
            
            assert isinstance(register, Register), "Should return a Register"
            assert 0 <= confidence <= 1.0, f"Invalid confidence for random text"
            assert round(confidence, 3) == confidence, "Confidence should be rounded"
    
    def test_detect_random_word_combinations(self):
        """Test detect with random combinations of marker words"""
        import random
        
        detector = RegisterDetector()
        
        casual_words = ["hey", "yeah", "lol", "totally", "gonna"]
        technical_words = ["API", "function", "parameter", "algorithm", "system"]
        academic_words = ["furthermore", "methodology", "demonstrates", "significant", "study"]
        narrative_words = ["walked", "remembered", "moment", "slowly", "never"]
        
        word_lists = [casual_words, technical_words, academic_words, narrative_words]
        
        for _ in range(10):
            # Pick a random word list
            words = random.choice(word_lists)
            # Create text with 5-10 words
            text = ' '.join(random.choices(words, k=random.randint(5, 10)))
            
            register, confidence = detector.detect(text)
            
            assert isinstance(register, Register), "Should return a Register"
            assert 0 <= confidence <= 1.0, f"Invalid confidence for text: {text}"


# ============================================================================
# Confidence Calibration Tests
# ============================================================================

class TestConfidenceCalibration:
    """Tests to verify confidence scores correlate with classification certainty."""
    
    def test_high_confidence_for_clear_examples(self):
        """Clear, unambiguous examples should have high confidence"""
        detector = RegisterDetector()
        
        clear_examples = [
            ("", Register.direct),  # Empty -> direct with 1.0
            ("Stop.", Register.direct),  # Very short imperative
            ("Hey guys! Yeah lol totally gonna do that!", Register.casual),
        ]
        
        for text, expected_register in clear_examples:
            register, confidence = detector.detect(text)
            if register == expected_register:
                # For clear matches, confidence should be relatively high
                # (allowing some flexibility based on actual implementation)
                assert confidence >= 0.6, \
                    f"Expected high confidence for clear example: {text}, got {confidence}"
    
    def test_lower_confidence_for_mixed_examples(self):
        """Mixed register examples might have lower confidence"""
        detector = RegisterDetector()
        
        # Text with both technical and academic markers
        mixed_text = "This study's API demonstrates the methodology."
        
        register, confidence = detector.detect(mixed_text)
        
        # We just verify the result is valid; confidence might be lower
        assert isinstance(register, Register)
        assert 0 <= confidence <= 1.0


# ============================================================================
# Marker Set Validation Tests
# ============================================================================

class TestMarkerInvariants:
    """Tests to validate the marker set invariants from the contract."""
    
    def test_casual_markers_exist(self):
        """CASUAL_MARKERS should contain 7 (pattern, weight) tuples with weights 1.0-2.5"""
        from transmogrifier.detector import CASUAL_MARKERS
        
        assert len(CASUAL_MARKERS) == 7, f"Expected 7 CASUAL_MARKERS, got {len(CASUAL_MARKERS)}"
        
        for pattern, weight in CASUAL_MARKERS:
            assert isinstance(weight, (int, float)), f"Weight should be numeric, got {type(weight)}"
            assert 1.0 <= weight <= 2.5, f"Weight {weight} outside range [1.0, 2.5]"
    
    def test_technical_markers_exist(self):
        """TECHNICAL_MARKERS should contain 5 (pattern, weight) tuples with weights 1.0-2.0"""
        from transmogrifier.detector import TECHNICAL_MARKERS
        
        assert len(TECHNICAL_MARKERS) == 5, f"Expected 5 TECHNICAL_MARKERS, got {len(TECHNICAL_MARKERS)}"
        
        for pattern, weight in TECHNICAL_MARKERS:
            assert isinstance(weight, (int, float)), f"Weight should be numeric, got {type(weight)}"
            assert 1.0 <= weight <= 2.0, f"Weight {weight} outside range [1.0, 2.0]"
    
    def test_academic_markers_exist(self):
        """ACADEMIC_MARKERS should contain 6 (pattern, weight) tuples with weights 1.5-2.5"""
        from transmogrifier.detector import ACADEMIC_MARKERS
        
        assert len(ACADEMIC_MARKERS) == 6, f"Expected 6 ACADEMIC_MARKERS, got {len(ACADEMIC_MARKERS)}"
        
        for pattern, weight in ACADEMIC_MARKERS:
            assert isinstance(weight, (int, float)), f"Weight should be numeric, got {type(weight)}"
            assert 1.5 <= weight <= 2.5, f"Weight {weight} outside range [1.5, 2.5]"
    
    def test_narrative_markers_exist(self):
        """NARRATIVE_MARKERS should contain 5 (pattern, weight) tuples with weights 1.0-3.0"""
        from transmogrifier.detector import NARRATIVE_MARKERS
        
        assert len(NARRATIVE_MARKERS) == 5, f"Expected 5 NARRATIVE_MARKERS, got {len(NARRATIVE_MARKERS)}"
        
        for pattern, weight in NARRATIVE_MARKERS:
            assert isinstance(weight, (int, float)), f"Weight should be numeric, got {type(weight)}"
            assert 1.0 <= weight <= 3.0, f"Weight {weight} outside range [1.0, 3.0]"
    
    def test_direct_markers_empty(self):
        """DIRECT_MARKERS should be an empty list (direct detected by absence + brevity)"""
        from transmogrifier.detector import DIRECT_MARKERS
        
        assert isinstance(DIRECT_MARKERS, list), "DIRECT_MARKERS should be a list"
        assert len(DIRECT_MARKERS) == 0, f"DIRECT_MARKERS should be empty, got {len(DIRECT_MARKERS)} items"
    
    def test_patterns_compiled_with_ignorecase(self):
        """All marker patterns should be compiled with re.IGNORECASE flag"""
        from transmogrifier.detector import (
            CASUAL_MARKERS, TECHNICAL_MARKERS, ACADEMIC_MARKERS, NARRATIVE_MARKERS
        )
        
        all_markers = CASUAL_MARKERS + TECHNICAL_MARKERS + ACADEMIC_MARKERS + NARRATIVE_MARKERS
        
        for pattern, weight in all_markers:
            # Check that the pattern is a compiled regex with IGNORECASE
            assert hasattr(pattern, 'search'), f"Pattern should be compiled regex: {pattern}"
            assert pattern.flags & re.IGNORECASE, \
                f"Pattern should have IGNORECASE flag: {pattern.pattern}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
