"""
Contract tests for TaskClassifier.classify() method.
Tests verify behavior at boundaries, covering happy paths, edge cases, and invariants.
"""

import pytest
import random
import string
from transmogrifier.task_classifier import TaskClassifier, TaskType


# Fixtures
@pytest.fixture
def classifier():
    """Provides a TaskClassifier instance for testing."""
    return TaskClassifier()


@pytest.fixture
def factual_samples():
    """Representative factual question samples."""
    return [
        "What is the capital of France?",
        "Who invented the telephone?",
        "When did World War II end?",
        "Where is the Eiffel Tower located?",
        "How many planets are in our solar system?"
    ]


@pytest.fixture
def reasoning_samples():
    """Representative reasoning task samples."""
    return [
        "If all cats are mammals and all mammals are animals, then are all cats animals?",
        "Given that A implies B and B implies C, what can we conclude about A and C?",
        "If it rains, the ground gets wet. The ground is wet. Did it rain?"
    ]


@pytest.fixture
def code_samples():
    """Representative code-related task samples."""
    return [
        "Write a function to sort a list of integers in Python",
        "Implement a binary search algorithm",
        "Debug this code snippet for me"
    ]


@pytest.fixture
def analysis_samples():
    """Representative analysis task samples."""
    return [
        "Analyze the trends in global temperature over the past century",
        "Examine the impact of social media on teenage mental health",
        "Evaluate the economic effects of trade policies"
    ]


@pytest.fixture
def creative_samples():
    """Representative creative writing task samples."""
    return [
        "Write a poem about the ocean at sunset",
        "Create a short story about a time traveler",
        "Compose a song about friendship"
    ]


@pytest.fixture
def instruction_samples():
    """Representative instruction-based task samples."""
    return [
        "Please explain how to bake a cake step by step",
        "Describe the process of changing a car tire",
        "Tell me how to set up a WordPress website"
    ]


# Happy Path Tests - Layer 1
class TestClassifyHappyPath:
    """Test successful classification of clear task type examples."""
    
    def test_classify_factual_happy_path(self, classifier):
        """Classifies a clear factual question with high confidence"""
        text = "What is the capital of France?"
        result = classifier.classify(text)
        
        assert result[0] == TaskType.factual
        assert 0.0 <= result[1] <= 1.0
        assert isinstance(result[1], float)
        assert round(result[1], 3) == result[1]
    
    def test_classify_reasoning_happy_path(self, classifier):
        """Classifies a reasoning task with appropriate confidence"""
        text = "If all cats are mammals and all mammals are animals, then are all cats animals?"
        result = classifier.classify(text)
        
        assert result[0] == TaskType.reasoning
        assert 0.0 <= result[1] <= 1.0
        assert round(result[1], 3) == result[1]
    
    def test_classify_code_happy_path(self, classifier):
        """Classifies a code-related prompt with high confidence"""
        text = "Write a function to sort a list of integers in Python"
        result = classifier.classify(text)
        
        assert result[0] == TaskType.code
        assert 0.0 <= result[1] <= 1.0
        assert round(result[1], 3) == result[1]
    
    def test_classify_analysis_happy_path(self, classifier):
        """Classifies an analytical task with appropriate confidence"""
        text = "Analyze the trends in global temperature over the past century"
        result = classifier.classify(text)
        
        assert result[0] == TaskType.analysis
        assert 0.0 <= result[1] <= 1.0
        assert round(result[1], 3) == result[1]
    
    def test_classify_creative_happy_path(self, classifier):
        """Classifies a creative writing task with high confidence"""
        text = "Write a poem about the ocean at sunset"
        result = classifier.classify(text)
        
        assert result[0] == TaskType.creative
        assert 0.0 <= result[1] <= 1.0
        assert round(result[1], 3) == result[1]
    
    def test_classify_instruction_happy_path(self, classifier):
        """Classifies an instruction-based task with appropriate confidence"""
        text = "Please explain how to bake a cake step by step"
        result = classifier.classify(text)
        
        assert result[0] == TaskType.instruction
        assert 0.0 <= result[1] <= 1.0
        assert round(result[1], 3) == result[1]
    
    def test_classify_factual_multiple_samples(self, classifier, factual_samples):
        """Multiple factual samples all classify as factual"""
        results = [classifier.classify(text) for text in factual_samples]
        
        assert all(task_type == TaskType.factual for task_type, _ in results)
    
    def test_classify_reasoning_multiple_samples(self, classifier, reasoning_samples):
        """Multiple reasoning samples classify as reasoning"""
        results = [classifier.classify(text) for text in reasoning_samples]
        
        assert all(task_type == TaskType.reasoning for task_type, _ in results)
    
    def test_classify_code_multiple_samples(self, classifier, code_samples):
        """Multiple code samples classify as code"""
        results = [classifier.classify(text) for text in code_samples]
        
        assert all(task_type == TaskType.code for task_type, _ in results)
    
    def test_classify_analysis_multiple_samples(self, classifier, analysis_samples):
        """Multiple analysis samples classify as analysis"""
        results = [classifier.classify(text) for text in analysis_samples]
        
        assert all(task_type == TaskType.analysis for task_type, _ in results)
    
    def test_classify_creative_multiple_samples(self, classifier, creative_samples):
        """Multiple creative samples classify as creative"""
        results = [classifier.classify(text) for text in creative_samples]
        
        assert all(task_type == TaskType.creative for task_type, _ in results)
    
    def test_classify_instruction_multiple_samples(self, classifier, instruction_samples):
        """Multiple instruction samples classify as instruction"""
        results = [classifier.classify(text) for text in instruction_samples]
        
        assert all(task_type == TaskType.instruction for task_type, _ in results)


# Edge Case Tests - Layer 3
class TestClassifyEdgeCases:
    """Test boundary conditions and edge cases."""
    
    def test_classify_empty_string(self, classifier):
        """Empty string returns unknown with 0.0 confidence"""
        text = ""
        result = classifier.classify(text)
        
        assert result[0] == TaskType.unknown
        assert result[1] == 0.0
    
    def test_classify_whitespace_only(self, classifier):
        """Whitespace-only input returns unknown with 0.0 confidence"""
        text = "   \t\n  "
        result = classifier.classify(text)
        
        assert result[0] == TaskType.unknown
        assert result[1] == 0.0
    
    def test_classify_no_patterns_match(self, classifier):
        """Input with no matching patterns returns unknown with 0.5 confidence"""
        text = "zzz xyz abc def"
        result = classifier.classify(text)
        
        assert result[0] == TaskType.unknown
        assert result[1] == 0.5
    
    def test_classify_unicode_text(self, classifier):
        """Unicode characters are handled correctly"""
        text = "Quelle est la capitale de la France? 🇫🇷"
        result = classifier.classify(text)
        
        assert isinstance(result[0], TaskType)
        assert 0.0 <= result[1] <= 1.0
        assert round(result[1], 3) == result[1]
    
    def test_classify_very_long_text(self, classifier):
        """Very long text is processed correctly"""
        text = "What is the meaning of life? " * 100
        result = classifier.classify(text)
        
        assert isinstance(result[0], TaskType)
        assert 0.0 <= result[1] <= 1.0
        assert round(result[1], 3) == result[1]
    
    def test_classify_single_character(self, classifier):
        """Single character input is handled"""
        text = "?"
        result = classifier.classify(text)
        
        assert isinstance(result[0], TaskType)
        assert 0.0 <= result[1] <= 1.0
        assert round(result[1], 3) == result[1]
    
    def test_classify_special_characters(self, classifier):
        """Special characters are handled correctly"""
        text = "!@#$%^&*()_+-=[]{}|;':,.<>?/~`"
        result = classifier.classify(text)
        
        assert isinstance(result[0], TaskType)
        assert 0.0 <= result[1] <= 1.0
        assert round(result[1], 3) == result[1]
    
    def test_classify_numerical_only(self, classifier):
        """Numerical-only input is handled"""
        text = "123456789"
        result = classifier.classify(text)
        
        assert isinstance(result[0], TaskType)
        assert 0.0 <= result[1] <= 1.0
        assert round(result[1], 3) == result[1]
    
    def test_classify_case_insensitive(self, classifier):
        """Pattern matching is case-insensitive"""
        text = "WHAT IS THE CAPITAL OF FRANCE?"
        result = classifier.classify(text)
        
        assert result[0] == TaskType.factual
        assert 0.0 <= result[1] <= 1.0
    
    def test_classify_confidence_rounded_three_places(self, classifier):
        """Confidence is rounded to exactly 3 decimal places"""
        text = "What is the meaning of life?"
        result = classifier.classify(text)
        
        assert round(result[1], 3) == result[1]
        assert len(str(result[1]).split('.')[-1]) <= 3 or result[1] in [0.0, 1.0]
    
    def test_classify_mixed_markers(self, classifier):
        """Text with markers from multiple categories"""
        text = "What is the best way to write code that analyzes data creatively?"
        result = classifier.classify(text)
        
        assert isinstance(result[0], TaskType)
        assert 0.0 <= result[1] <= 1.0
        assert round(result[1], 3) == result[1]
    
    def test_classify_relative_confidence_clear_vs_ambiguous(self, classifier):
        """Clear samples have higher confidence than ambiguous ones"""
        clear_text = "What is the capital of France?"
        ambiguous_text = "Tell me something about France maybe"
        
        clear_result = classifier.classify(clear_text)
        ambiguous_result = classifier.classify(ambiguous_text)
        
        clear_confidence = clear_result[1]
        ambiguous_confidence = ambiguous_result[1]
        
        # Clear factual question should have higher confidence
        assert clear_confidence > ambiguous_confidence


# Invariant Tests - Layer 2 (without Hypothesis)
class TestClassifyInvariants:
    """Test invariants and properties that must hold."""
    
    def test_classify_determinism(self, classifier):
        """Same input produces same output (determinism)"""
        text = "What is the capital of Spain?"
        result = classifier.classify(text)
        result_second = classifier.classify(text)
        result_third = classifier.classify(text)
        
        assert result == result_second
        assert result == result_third
    
    def test_classify_idempotence(self, classifier):
        """Classifier behavior is idempotent for same input"""
        text = "Explain how to solve a quadratic equation"
        result = classifier.classify(text)
        
        repeated_calls = [classifier.classify(text) for _ in range(10)]
        
        assert all(call == result for call in repeated_calls)
    
    def test_classify_confidence_bounds_invariant(self, classifier):
        """Confidence is always within [0.0, 1.0] bounds"""
        # Generate random test inputs
        random_inputs = [
            ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=random.randint(1, 100)))
            for _ in range(50)
        ]
        
        results = [classifier.classify(text) for text in random_inputs]
        
        assert all(0.0 <= conf <= 1.0 for _, conf in results)
    
    def test_classify_all_task_types_covered(self, classifier, factual_samples, 
                                             reasoning_samples, code_samples,
                                             analysis_samples, creative_samples, 
                                             instruction_samples):
        """All TaskType enum values are testable"""
        all_samples = (factual_samples + reasoning_samples + code_samples + 
                      analysis_samples + creative_samples + instruction_samples)
        
        results = [classifier.classify(text) for text in all_samples]
        classified_types = {task_type for task_type, _ in results}
        
        assert TaskType.factual in classified_types
        assert TaskType.reasoning in classified_types
        assert TaskType.code in classified_types
        assert TaskType.analysis in classified_types
        assert TaskType.creative in classified_types
        assert TaskType.instruction in classified_types
    
    def test_classify_return_tuple_structure(self, classifier):
        """Return value is always a tuple of (TaskType, float)"""
        text = "Sample text for type checking"
        result = classifier.classify(text)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], TaskType)
        assert isinstance(result[1], float)
    
    def test_classify_confidence_precision_invariant(self, classifier):
        """All confidence values are rounded to 3 decimal places"""
        test_inputs = [
            "What is AI?",
            "Explain quantum mechanics",
            "Write a sorting algorithm",
            "Analyze market trends",
            "Create a haiku",
            "How do I cook pasta?",
            "",
            "xyz abc",
            "123",
            "!@#$%"
        ]
        
        results = [classifier.classify(text) for text in test_inputs]
        
        for _, confidence in results:
            assert round(confidence, 3) == confidence
    
    def test_classify_task_type_enum_validity(self, classifier):
        """All returned TaskTypes are valid enum members"""
        test_inputs = [
            "What is this?",
            "Why does that happen?",
            "def foo(): pass",
            "Compare A and B",
            "Imagine a world where...",
            "Do this step by step",
            "",
            "random text here"
        ]
        
        valid_task_types = set(TaskType)
        
        for text in test_inputs:
            task_type, _ = classifier.classify(text)
            assert task_type in valid_task_types
    
    def test_classify_whitespace_normalization_invariant(self, classifier):
        """Leading/trailing whitespace doesn't affect classification of meaningful text"""
        base_text = "What is the capital of Germany?"
        
        result_base = classifier.classify(base_text)
        result_leading = classifier.classify("   " + base_text)
        result_trailing = classifier.classify(base_text + "   ")
        result_both = classifier.classify("   " + base_text + "   ")
        
        # All should classify to the same TaskType
        assert result_base[0] == result_leading[0] == result_trailing[0] == result_both[0]
    
    def test_classify_case_invariance(self, classifier):
        """Case variations produce same TaskType classification"""
        base_text = "what is the capital of italy?"
        upper_text = base_text.upper()
        title_text = base_text.title()
        
        result_base = classifier.classify(base_text)
        result_upper = classifier.classify(upper_text)
        result_title = classifier.classify(title_text)
        
        # All should classify to the same TaskType (factual)
        assert result_base[0] == result_upper[0] == result_title[0]
        assert result_base[0] == TaskType.factual
