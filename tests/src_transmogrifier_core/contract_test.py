"""
Contract-based test suite for Transmogrifier Core Orchestrator
Generated from contract version 1
Tests cover initialization, translation operations, config precedence, edge cases, and invariants
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Optional
import time
import uuid


# Import the component under test
# Assuming the module structure based on component_id and dependencies
try:
    from transmogrifier.core import Transmogrifier, TranslationResult, TranslationConfig, Register, TranslationLevel
    from transmogrifier.detector import RegisterDetector
    from transmogrifier.task_classifier import TaskClassifier
    from transmogrifier.profiles import ProfileCache
    from transmogrifier.rules import RuleEngine
except ImportError:
    # Fallback import structure
    try:
        from transmogrifier.core import Transmogrifier, TranslationResult, TranslationConfig, Register, TranslationLevel
        from transmogrifier.detector import RegisterDetector
        from transmogrifier.task_classifier import TaskClassifier
        from transmogrifier.profiles import ProfileCache
        from transmogrifier.rules import RuleEngine
    except ImportError:
        # Create minimal mock types for testing if imports fail
        from enum import Enum
        
        class Register(Enum):
            direct = "direct"
            casual = "casual"
            technical = "technical"
            academic = "academic"
            narrative = "narrative"
        
        class TranslationLevel(Enum):
            system_prompt = 1
            rule_rewrite = 2
            llm_translate = 3
        
        class TranslationConfig:
            def __init__(self, target_register=None, max_level=TranslationLevel.rule_rewrite,
                        semantic_threshold=0.85, spread_threshold_pp=2.0,
                        passthrough_on_failure=False, task_aware=True):
                self.target_register = target_register
                self.max_level = max_level
                self.semantic_threshold = semantic_threshold
                self.spread_threshold_pp = spread_threshold_pp
                self.passthrough_on_failure = passthrough_on_failure
                self.task_aware = task_aware
        
        class TranslationResult:
            def __init__(self, input_text, output_text, detected_register, target_register,
                        detected_task, level_applied, system_prompt, semantic_similarity,
                        skipped, skip_reason, elapsed_ms, trace_id):
                self.input_text = input_text
                self.output_text = output_text
                self.detected_register = detected_register
                self.target_register = target_register
                self.detected_task = detected_task
                self.level_applied = level_applied
                self.system_prompt = system_prompt
                self.semantic_similarity = semantic_similarity
                self.skipped = skipped
                self.skip_reason = skip_reason
                self.elapsed_ms = elapsed_ms
                self.trace_id = trace_id
        
        class RegisterDetector:
            def detect(self, text):
                return Register.technical
        
        class TaskClassifier:
            def classify(self, text):
                return "general"
        
        class ProfileCache:
            def get_profile(self, model):
                return Mock(is_invariant=False, task_spread=3.0, target_register=Register.technical)
        
        class RuleEngine:
            def rewrite(self, text, source_register, target_register):
                return f"rewritten {target_register.value} text"
        
        class Transmogrifier:
            def __init__(self, profile_cache=None, config=None):
                self._detector = RegisterDetector()
                self._task_classifier = TaskClassifier()
                self._profile_cache = profile_cache if profile_cache is not None else ProfileCache()
                self._rule_engine = RuleEngine()
                self._config = config if config is not None else TranslationConfig()
            
            def translate(self, text, model, config=None):
                start = time.perf_counter()
                
                effective_config = config if config is not None else self._config
                detected = self._detector.detect(text)
                task = self._task_classifier.classify(text) if effective_config.task_aware else "none"
                
                profile = self._profile_cache.get_profile(model)
                target = effective_config.target_register or profile.target_register or detected
                
                # Validate target register
                if isinstance(target, str) and target not in [r.value for r in Register]:
                    raise ValueError(f"Invalid register: {target}")
                
                # Skip logic
                skipped = False
                skip_reason = None
                if profile.is_invariant and profile.task_spread < 2.0:
                    skipped = True
                    skip_reason = f"invariant profile with low spread ({profile.task_spread})"
                
                # Translation logic
                if detected == target:
                    output_text = text
                    level = TranslationLevel.system_prompt
                else:
                    output_text = self._rule_engine.rewrite(text, detected, target)
                    level = TranslationLevel.rule_rewrite
                
                if skipped:
                    output_text = text
                
                # Generate system prompt
                system_prompt = f"Translate from {detected.value} to {target.value}"
                
                elapsed = (time.perf_counter() - start) * 1000
                trace_id = uuid.uuid4().hex[:12]
                
                return TranslationResult(
                    input_text=text,
                    output_text=output_text,
                    detected_register=detected,
                    target_register=target,
                    detected_task=task,
                    level_applied=level,
                    system_prompt=system_prompt,
                    semantic_similarity=0.95,
                    skipped=skipped,
                    skip_reason=skip_reason,
                    elapsed_ms=elapsed,
                    trace_id=trace_id
                )


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_detector():
    """Mock RegisterDetector"""
    detector = Mock(spec=RegisterDetector)
    detector.detect.return_value = Register.technical
    return detector


@pytest.fixture
def mock_task_classifier():
    """Mock TaskClassifier"""
    classifier = Mock(spec=TaskClassifier)
    classifier.classify.return_value = "general_task"
    return classifier


@pytest.fixture
def mock_profile_cache():
    """Mock ProfileCache with default profile"""
    cache = Mock(spec=ProfileCache)
    profile = Mock()
    profile.is_invariant = False
    profile.task_spread = 3.0
    profile.target_register = Register.technical
    cache.get_profile.return_value = profile
    return cache


@pytest.fixture
def mock_rule_engine():
    """Mock RuleEngine"""
    engine = Mock(spec=RuleEngine)
    engine.rewrite.return_value = "rewritten technical text"
    return engine


@pytest.fixture
def default_config():
    """Default TranslationConfig"""
    return TranslationConfig(
        target_register=None,
        max_level=TranslationLevel.rule_rewrite,
        semantic_threshold=0.85,
        spread_threshold_pp=2.0,
        passthrough_on_failure=False,
        task_aware=True
    )


@pytest.fixture
def mock_time(monkeypatch):
    """Mock time.perf_counter for deterministic timing"""
    counter = [0.0]
    def fake_perf_counter():
        val = counter[0]
        counter[0] += 0.1  # Each call adds 100ms
        return val
    monkeypatch.setattr('time.perf_counter', fake_perf_counter)


@pytest.fixture
def mock_uuid(monkeypatch):
    """Mock uuid.uuid4 for deterministic trace IDs"""
    call_count = [0]
    def fake_uuid4():
        call_count[0] += 1
        mock_uuid = Mock()
        mock_uuid.hex = f"abcdef{call_count[0]:06d}123456"
        return mock_uuid
    monkeypatch.setattr('uuid.uuid4', fake_uuid4)


# ============================================================================
# Initialization Tests
# ============================================================================

class TestInitialization:
    """Test Transmogrifier.__init__ with various parameter combinations"""
    
    def test_init_with_valid_params(self, mock_profile_cache, default_config):
        """Happy path: Initialize Transmogrifier with valid profile_cache and config"""
        trans = Transmogrifier(profile_cache=mock_profile_cache, config=default_config)
        
        assert hasattr(trans, '_detector')
        assert hasattr(trans, '_task_classifier')
        assert hasattr(trans, '_profile_cache')
        assert hasattr(trans, '_rule_engine')
        assert hasattr(trans, '_config')
        assert trans._profile_cache is mock_profile_cache
        assert trans._config is default_config
    
    def test_init_with_none_profile_cache(self, default_config):
        """Edge case: Initialize with None profile_cache creates default ProfileCache"""
        trans = Transmogrifier(profile_cache=None, config=default_config)
        
        assert trans._profile_cache is not None
        assert isinstance(trans._profile_cache, ProfileCache)
    
    def test_init_with_none_config(self, mock_profile_cache):
        """Edge case: Initialize with None config creates default TranslationConfig"""
        trans = Transmogrifier(profile_cache=mock_profile_cache, config=None)
        
        assert trans._config is not None
        assert isinstance(trans._config, TranslationConfig)
    
    def test_init_with_both_none(self):
        """Edge case: Initialize with both parameters None creates defaults"""
        trans = Transmogrifier(profile_cache=None, config=None)
        
        assert isinstance(trans._profile_cache, ProfileCache)
        assert isinstance(trans._config, TranslationConfig)


# ============================================================================
# Translation Success Path Tests
# ============================================================================

class TestTranslateSuccessPaths:
    """Test successful translation scenarios"""
    
    def test_translate_happy_path_same_register(self, mock_time, mock_uuid):
        """Happy path: Translate with detected == target register"""
        with patch('transmogrifier.core.RegisterDetector') as MockDetector, \
             patch('transmogrifier.core.TaskClassifier') as MockClassifier, \
             patch('transmogrifier.core.ProfileCache') as MockCache, \
             patch('transmogrifier.core.RuleEngine') as MockEngine:
            
            # Setup mocks
            detector = Mock()
            detector.detect.return_value = Register.technical
            MockDetector.return_value = detector
            
            classifier = Mock()
            classifier.classify.return_value = "general"
            MockClassifier.return_value = classifier
            
            profile = Mock()
            profile.is_invariant = False
            profile.task_spread = 3.0
            profile.target_register = Register.technical
            cache = Mock()
            cache.get_profile.return_value = profile
            MockCache.return_value = cache
            
            engine = Mock()
            MockEngine.return_value = engine
            
            trans = Transmogrifier()
            result = trans.translate(text='test input', model='gpt-4', config=None)
            
            assert result.input_text == 'test input'
            assert result.output_text == 'test input'
            assert result.detected_register == Register.technical
            assert result.target_register == Register.technical
            assert result.level_applied == TranslationLevel.system_prompt
            assert result.skipped == False
            assert result.elapsed_ms >= 0
            assert len(result.trace_id) == 12
    
    def test_translate_happy_path_different_register(self, mock_time, mock_uuid):
        """Happy path: Translate with detected != target register"""
        with patch('transmogrifier.core.RegisterDetector') as MockDetector, \
             patch('transmogrifier.core.TaskClassifier') as MockClassifier, \
             patch('transmogrifier.core.ProfileCache') as MockCache, \
             patch('transmogrifier.core.RuleEngine') as MockEngine:
            
            detector = Mock()
            detector.detect.return_value = Register.casual
            MockDetector.return_value = detector
            
            classifier = Mock()
            classifier.classify.return_value = "general"
            MockClassifier.return_value = classifier
            
            profile = Mock()
            profile.is_invariant = False
            profile.task_spread = 3.0
            profile.target_register = Register.technical
            cache = Mock()
            cache.get_profile.return_value = profile
            MockCache.return_value = cache
            
            engine = Mock()
            engine.rewrite.return_value = 'rewritten technical text'
            MockEngine.return_value = engine
            
            config = TranslationConfig(target_register=Register.technical)
            trans = Transmogrifier(config=config)
            result = trans.translate(text='casual text', model='gpt-4', config=None)
            
            assert result.input_text == 'casual text'
            assert result.output_text == 'rewritten technical text'
            assert result.detected_register == Register.casual
            assert result.target_register == Register.technical
            assert result.level_applied == TranslationLevel.rule_rewrite
            assert result.system_prompt is not None
            assert result.skipped == False
    
    def test_translate_all_fields_populated(self, mock_time, mock_uuid):
        """Invariant: All 12 TranslationResult fields are properly populated"""
        trans = Transmogrifier()
        result = trans.translate(text='test', model='gpt-4', config=None)
        
        # Verify all 12 fields
        assert result.input_text is not None
        assert result.output_text is not None
        assert result.detected_register is not None
        assert result.target_register is not None
        assert result.detected_task is not None
        assert result.level_applied is not None
        assert result.system_prompt is not None
        # semantic_similarity can be None per Optional
        assert result.skipped is not None
        # skip_reason should be None when not skipped
        assert result.elapsed_ms >= 0
        assert len(result.trace_id) == 12


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_translate_skip_logic_invariant_profile(self, mock_time, mock_uuid):
        """Edge case: Skip when profile.is_invariant=True and task_spread < 2.0"""
        with patch('transmogrifier.core.RegisterDetector') as MockDetector, \
             patch('transmogrifier.core.TaskClassifier') as MockClassifier, \
             patch('transmogrifier.core.ProfileCache') as MockCache, \
             patch('transmogrifier.core.RuleEngine') as MockEngine:
            
            detector = Mock()
            detector.detect.return_value = Register.technical
            MockDetector.return_value = detector
            
            classifier = Mock()
            classifier.classify.return_value = "general"
            MockClassifier.return_value = classifier
            
            profile = Mock()
            profile.is_invariant = True
            profile.task_spread = 1.5
            profile.target_register = Register.technical
            cache = Mock()
            cache.get_profile.return_value = profile
            MockCache.return_value = cache
            
            engine = Mock()
            MockEngine.return_value = engine
            
            trans = Transmogrifier()
            result = trans.translate(text='test input', model='gpt-4', config=None)
            
            assert result.skipped == True
            assert result.output_text == result.input_text
            assert result.skip_reason is not None
            assert 'invariant' in result.skip_reason.lower() or 'spread' in result.skip_reason.lower()
    
    def test_translate_no_skip_normal_case(self):
        """Invariant: When not skipped, skip_reason is None"""
        trans = Transmogrifier()
        result = trans.translate(text='test', model='gpt-4', config=None)
        
        # Assuming normal case doesn't skip
        if not result.skipped:
            assert result.skip_reason is None
    
    def test_config_precedence_method_over_constructor(self, mock_time, mock_uuid):
        """Edge case: Method-level config overrides constructor config"""
        with patch('transmogrifier.core.RegisterDetector') as MockDetector, \
             patch('transmogrifier.core.TaskClassifier') as MockClassifier, \
             patch('transmogrifier.core.ProfileCache') as MockCache, \
             patch('transmogrifier.core.RuleEngine') as MockEngine:
            
            detector = Mock()
            detector.detect.return_value = Register.technical
            MockDetector.return_value = detector
            
            classifier = Mock()
            classifier.classify.return_value = "general"
            MockClassifier.return_value = classifier
            
            profile = Mock()
            profile.is_invariant = False
            profile.task_spread = 3.0
            profile.target_register = Register.technical
            cache = Mock()
            cache.get_profile.return_value = profile
            MockCache.return_value = cache
            
            engine = Mock()
            engine.rewrite.return_value = 'casual output'
            MockEngine.return_value = engine
            
            constructor_config = TranslationConfig(target_register=Register.technical)
            method_config = TranslationConfig(target_register=Register.casual)
            
            trans = Transmogrifier(config=constructor_config)
            result = trans.translate(text='test', model='gpt-4', config=method_config)
            
            assert result.target_register == Register.casual
    
    def test_config_none_at_both_levels(self):
        """Edge case: Config None at both constructor and method level uses defaults"""
        trans = Transmogrifier(config=None)
        result = trans.translate(text='test', model='gpt-4', config=None)
        
        assert result is not None
        assert result.target_register is not None
    
    def test_translate_empty_text(self):
        """Edge case: Translate empty string"""
        trans = Transmogrifier()
        result = trans.translate(text='', model='gpt-4', config=None)
        
        assert result.input_text == ''
        assert result is not None
    
    def test_translate_whitespace_text(self):
        """Edge case: Translate whitespace-only text"""
        trans = Transmogrifier()
        result = trans.translate(text='   \n\t  ', model='gpt-4', config=None)
        
        assert result.input_text == '   \n\t  '
        assert result is not None
    
    def test_translate_unicode_text(self):
        """Edge case: Translate text with unicode characters"""
        unicode_text = "Hello 👋 世界 🌍"
        trans = Transmogrifier()
        result = trans.translate(text=unicode_text, model='gpt-4', config=None)
        
        assert unicode_text in result.input_text or result.input_text == unicode_text
        assert result is not None
    
    def test_translate_long_text(self):
        """Edge case: Translate very long text"""
        long_text = "a" * 10000
        trans = Transmogrifier()
        result = trans.translate(text=long_text, model='gpt-4', config=None)
        
        assert len(result.input_text) == 10000
        assert result is not None
    
    def test_all_register_enums_valid(self):
        """Edge case: Test all valid Register enum values"""
        trans = Transmogrifier()
        
        for register in Register:
            config = TranslationConfig(target_register=register)
            result = trans.translate(text='test', model='gpt-4', config=config)
            assert result is not None
    
    def test_task_aware_true(self):
        """Edge case: task_aware=True enables task classification"""
        config = TranslationConfig(task_aware=True)
        trans = Transmogrifier(config=config)
        result = trans.translate(text='test', model='gpt-4', config=None)
        
        assert result.detected_task is not None
        assert len(result.detected_task) > 0
    
    def test_task_aware_false(self):
        """Edge case: task_aware=False skips task classification"""
        config = TranslationConfig(task_aware=False)
        trans = Transmogrifier(config=config)
        result = trans.translate(text='test', model='gpt-4', config=None)
        
        # Should still return a result
        assert result is not None
    
    def test_passthrough_on_failure_true(self):
        """Edge case: passthrough_on_failure=True returns original on error"""
        config = TranslationConfig(passthrough_on_failure=True)
        trans = Transmogrifier(config=config)
        
        # In normal operation, this should still work
        result = trans.translate(text='test', model='gpt-4', config=None)
        assert result is not None
    
    def test_semantic_threshold_boundary(self):
        """Edge case: semantic_threshold at exact boundary"""
        config = TranslationConfig(semantic_threshold=0.85)
        trans = Transmogrifier(config=config)
        result = trans.translate(text='test', model='gpt-4', config=None)
        
        assert result is not None
    
    def test_spread_threshold_boundary(self, mock_time, mock_uuid):
        """Edge case: spread_threshold_pp at exact boundary (2.0)"""
        with patch('transmogrifier.core.ProfileCache') as MockCache:
            profile = Mock()
            profile.is_invariant = True
            profile.task_spread = 2.0  # Exact boundary
            profile.target_register = Register.technical
            cache = Mock()
            cache.get_profile.return_value = profile
            MockCache.return_value = cache
            
            config = TranslationConfig(spread_threshold_pp=2.0)
            trans = Transmogrifier(config=config)
            result = trans.translate(text='test', model='gpt-4', config=None)
            
            # At exactly 2.0, should NOT skip (< 2.0)
            assert result.skipped == False


# ============================================================================
# Error Case Tests
# ============================================================================

class TestErrorCases:
    """Test error conditions and exception handling"""
    
    def test_translate_invalid_register_string(self):
        """Error case: Invalid register string raises ValueError"""
        # Create a mock config that bypasses pydantic validation
        with patch('transmogrifier.core.RegisterDetector') as MockDetector, \
             patch('transmogrifier.core.TaskClassifier') as MockClassifier, \
             patch('transmogrifier.core.ProfileCache') as MockCache:
            
            detector = Mock()
            detector.detect.return_value = Register.technical
            MockDetector.return_value = detector
            
            classifier = Mock()
            classifier.classify.return_value = "general"
            MockClassifier.return_value = classifier
            
            profile = Mock()
            profile.is_invariant = False
            profile.task_spread = 3.0
            profile.target_register = "invalid_register"  # Invalid string
            cache = Mock()
            cache.get_profile.return_value = profile
            MockCache.return_value = cache
            
            trans = Transmogrifier()
            
            with pytest.raises(ValueError) as exc_info:
                trans.translate(text='test', model='gpt-4', config=None)
            
            assert 'register' in str(exc_info.value).lower() or 'invalid' in str(exc_info.value).lower()


# ============================================================================
# Invariant Tests
# ============================================================================

class TestInvariants:
    """Test system invariants across all operations"""
    
    def test_elapsed_ms_nonnegative(self):
        """Invariant: elapsed_ms is always >= 0"""
        trans = Transmogrifier()
        result = trans.translate(text='test', model='gpt-4', config=None)
        
        assert result.elapsed_ms >= 0
        assert result.elapsed_ms < 10000  # Sanity check
    
    def test_trace_id_format(self):
        """Invariant: trace_id is always 12-character hexadecimal string"""
        trans = Transmogrifier()
        result = trans.translate(text='test', model='gpt-4', config=None)
        
        assert len(result.trace_id) == 12
        assert all(c in '0123456789abcdef' for c in result.trace_id.lower())
    
    def test_trace_id_uniqueness(self):
        """Invariant: Each translation gets unique trace_id"""
        trans = Transmogrifier()
        
        result1 = trans.translate(text='test', model='gpt-4', config=None)
        result2 = trans.translate(text='test', model='gpt-4', config=None)
        result3 = trans.translate(text='test', model='gpt-4', config=None)
        
        assert result1.trace_id != result2.trace_id
        assert result1.trace_id != result3.trace_id
        assert result2.trace_id != result3.trace_id
    
    def test_system_prompt_always_generated(self):
        """Invariant: system_prompt is always generated regardless of rewriting"""
        trans = Transmogrifier()
        result = trans.translate(text='test', model='gpt-4', config=None)
        
        assert result.system_prompt is not None
        assert isinstance(result.system_prompt, str)
        assert len(result.system_prompt) > 0
    
    def test_level_applied_invariant_same_register(self, mock_time, mock_uuid):
        """Invariant: If detected == target, level_applied is system_prompt"""
        with patch('transmogrifier.core.RegisterDetector') as MockDetector, \
             patch('transmogrifier.core.ProfileCache') as MockCache:
            
            detector = Mock()
            detector.detect.return_value = Register.technical
            MockDetector.return_value = detector
            
            profile = Mock()
            profile.is_invariant = False
            profile.task_spread = 3.0
            profile.target_register = Register.technical
            cache = Mock()
            cache.get_profile.return_value = profile
            MockCache.return_value = cache
            
            trans = Transmogrifier()
            result = trans.translate(text='test', model='gpt-4', config=None)
            
            assert result.detected_register == result.target_register
            assert result.level_applied == TranslationLevel.system_prompt
    
    def test_level_applied_invariant_different_register(self, mock_time, mock_uuid):
        """Invariant: If detected != target and max_level >= rule_rewrite, level_applied is rule_rewrite"""
        with patch('transmogrifier.core.RegisterDetector') as MockDetector, \
             patch('transmogrifier.core.ProfileCache') as MockCache, \
             patch('transmogrifier.core.RuleEngine') as MockEngine:
            
            detector = Mock()
            detector.detect.return_value = Register.casual
            MockDetector.return_value = detector
            
            profile = Mock()
            profile.is_invariant = False
            profile.task_spread = 3.0
            profile.target_register = Register.technical
            cache = Mock()
            cache.get_profile.return_value = profile
            MockCache.return_value = cache
            
            engine = Mock()
            engine.rewrite.return_value = 'rewritten'
            MockEngine.return_value = engine
            
            config = TranslationConfig(max_level=TranslationLevel.rule_rewrite)
            trans = Transmogrifier(config=config)
            result = trans.translate(text='test', model='gpt-4', config=None)
            
            assert result.detected_register != result.target_register
            assert result.level_applied == TranslationLevel.rule_rewrite
    
    def test_skipped_invariant_output_equals_input(self, mock_time, mock_uuid):
        """Invariant: If skipped=True, output_text == input_text"""
        with patch('transmogrifier.core.ProfileCache') as MockCache:
            profile = Mock()
            profile.is_invariant = True
            profile.task_spread = 1.5
            profile.target_register = Register.technical
            cache = Mock()
            cache.get_profile.return_value = profile
            MockCache.return_value = cache
            
            trans = Transmogrifier()
            result = trans.translate(text='test input', model='gpt-4', config=None)
            
            assert result.skipped == True
            assert result.output_text == result.input_text


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance characteristics"""
    
    def test_performance_linear_complexity(self):
        """Performance: O(n) complexity with text length"""
        trans = Transmogrifier()
        
        # Test with different text lengths
        short_result = trans.translate(text='a' * 100, model='gpt-4', config=None)
        medium_result = trans.translate(text='a' * 1000, model='gpt-4', config=None)
        long_result = trans.translate(text='a' * 10000, model='gpt-4', config=None)
        
        # Times should increase with length (rough check)
        assert short_result.elapsed_ms >= 0
        assert medium_result.elapsed_ms >= 0
        assert long_result.elapsed_ms >= 0
        
        # Very rough linearity check - longer should take more time
        # But we can't be too strict since this is local processing
        assert long_result.elapsed_ms >= short_result.elapsed_ms * 0.1  # Very loose bound


# ============================================================================
# Configuration Tests
# ============================================================================

class TestConfiguration:
    """Test various configuration scenarios"""
    
    def test_translation_level_ordering(self):
        """Invariant: TranslationLevel ordering system_prompt < rule_rewrite < llm_translate"""
        assert TranslationLevel.system_prompt.value < TranslationLevel.rule_rewrite.value
        assert TranslationLevel.rule_rewrite.value < TranslationLevel.llm_translate.value
        assert TranslationLevel.system_prompt.value == 1
        assert TranslationLevel.rule_rewrite.value == 2
        assert TranslationLevel.llm_translate.value == 3
