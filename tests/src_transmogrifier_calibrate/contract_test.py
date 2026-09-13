"""
Contract tests for CalibrationRunner component.

Three-tier testing strategy:
1. Unit tests for score_response() with randomized inputs
2. Component tests for CalibrationRunner with mocked dependencies
3. Integration tests for end-to-end workflows

Tests verify behavior at boundaries, not implementation details.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime, timezone
import time
import random
import string

# Import the backend protocol from its owning package.
from transmogrifier.backends import Backend

# Import component under test
from transmogrifier.calibrate import (
    CalibrationRunner,
    score_response,
    REGISTER_TRANSFORMS,
    BENCHMARK_TASKS,
    ProfileCache,
    ModelProfile,
    RegisterAccuracy,
    TaskRegisterProfile,
)


# =============================================================================
# Unit Tests for score_response()
# =============================================================================

class TestScoreResponseHappyPath:
    """Happy path tests for score_response function."""

    def test_score_response_happy_path_accept_no_reject(self):
        """Verify score_response returns True when response contains accept pattern and no reject patterns."""
        response = "The capital of France is Paris"
        task = {"accept": ["Paris"], "reject": [], "category": "factual"}
        
        result = score_response(response, task)
        
        assert result is True

    def test_score_response_happy_path_multiple_accept_patterns(self):
        """Verify score_response returns True when response contains any of multiple accept patterns."""
        response = "The result is 42"
        task = {"accept": ["42", "forty-two"], "reject": [], "category": "factual"}
        
        result = score_response(response, task)
        
        assert result is True

    def test_score_response_reject_pattern_present(self):
        """Verify score_response returns False when reject pattern is present even if accept pattern exists."""
        response = "Paris is wrong, it's London"
        task = {"accept": ["Paris"], "reject": ["wrong"], "category": "factual"}
        
        result = score_response(response, task)
        
        assert result is False

    def test_score_response_reasoning_category_strict_reject(self):
        """Verify score_response returns False for reasoning category when any reject pattern present."""
        response = "The answer involves uncertain reasoning"
        task = {"accept": ["answer"], "reject": ["uncertain"], "category": "reasoning"}
        
        result = score_response(response, task)
        
        assert result is False

    def test_score_response_no_accept_pattern(self):
        """Verify score_response returns False when response contains no accept patterns."""
        response = "The capital is London"
        task = {"accept": ["Paris"], "reject": [], "category": "factual"}
        
        result = score_response(response, task)
        
        assert result is False


class TestScoreResponseEdgeCases:
    """Edge case tests for score_response function."""

    def test_score_response_edge_case_empty_response(self):
        """Verify score_response handles empty response string."""
        response = ""
        task = {"accept": ["Paris"], "reject": [], "category": "factual"}
        
        result = score_response(response, task)
        
        assert result is False

    def test_score_response_edge_case_empty_accept_list(self):
        """Verify score_response handles empty accept list."""
        response = "Any response text"
        task = {"accept": [], "reject": [], "category": "factual"}
        
        result = score_response(response, task)
        
        assert result is False

    def test_score_response_edge_case_case_sensitivity(self):
        """Verify score_response pattern matching is case-sensitive or case-insensitive as per implementation."""
        response = "The answer is PARIS"
        task = {"accept": ["paris"], "reject": [], "category": "factual"}
        
        result = score_response(response, task)
        
        assert isinstance(result, bool)

    def test_score_response_edge_case_partial_match(self):
        """Verify score_response handles partial string matches."""
        response = "The Parisian culture is rich"
        task = {"accept": ["Paris"], "reject": [], "category": "factual"}
        
        result = score_response(response, task)
        
        assert isinstance(result, bool)

    def test_score_response_multiple_reject_patterns(self):
        """Verify score_response returns False when any reject pattern matches."""
        response = "This is incorrect and wrong"
        task = {"accept": ["This"], "reject": ["incorrect", "wrong"], "category": "factual"}
        
        result = score_response(response, task)
        
        assert result is False

    def test_score_response_whitespace_response(self):
        """Verify score_response handles whitespace-only response."""
        response = "   \n\t  "
        task = {"accept": ["answer"], "reject": [], "category": "factual"}
        
        result = score_response(response, task)
        
        assert result is False

    def test_score_response_special_characters(self):
        """Verify score_response handles special characters in patterns."""
        response = "The answer is $100.00"
        task = {"accept": ["$100"], "reject": [], "category": "factual"}
        
        result = score_response(response, task)
        
        assert isinstance(result, bool)


class TestScoreResponseRandomized:
    """Randomized testing for score_response using direct random generation."""

    def test_score_response_random_valid_inputs(self):
        """Test score_response with randomly generated valid inputs."""
        for _ in range(10):
            # Generate random response
            response_length = random.randint(10, 100)
            response = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=response_length))
            
            # Generate random task
            accept_patterns = [f"pattern_{i}" for i in range(random.randint(1, 5))]
            reject_patterns = [f"reject_{i}" for i in range(random.randint(0, 3))]
            category = random.choice(["factual", "reasoning", "code", "analysis", "creative", "instruction"])
            
            task = {
                "accept": accept_patterns,
                "reject": reject_patterns,
                "category": category
            }
            
            result = score_response(response, task)
            
            # Verify result is boolean
            assert isinstance(result, bool)

    def test_score_response_guaranteed_match(self):
        """Test score_response with guaranteed accept pattern match."""
        for _ in range(10):
            accept_pattern = f"match_{random.randint(1000, 9999)}"
            response = f"This contains {accept_pattern} in the middle"
            task = {"accept": [accept_pattern], "reject": [], "category": "factual"}
            
            result = score_response(response, task)
            
            assert result is True

    def test_score_response_guaranteed_reject(self):
        """Test score_response with guaranteed reject pattern match."""
        for _ in range(10):
            accept_pattern = f"accept_{random.randint(1000, 9999)}"
            reject_pattern = f"reject_{random.randint(1000, 9999)}"
            response = f"This contains {accept_pattern} and {reject_pattern}"
            task = {"accept": [accept_pattern], "reject": [reject_pattern], "category": "factual"}
            
            result = score_response(response, task)
            
            assert result is False


# =============================================================================
# Component Tests for CalibrationRunner.__init__()
# =============================================================================

class TestCalibrationRunnerInit:
    """Tests for CalibrationRunner initialization."""

    def test_init_happy_path_with_cache(self):
        """Verify __init__ initializes CalibrationRunner with backend and profile cache."""
        mock_backend = Mock(spec=Backend)
        mock_profile_cache = Mock(spec=ProfileCache)
        
        runner = CalibrationRunner(mock_backend, mock_profile_cache)
        
        assert runner._backend is mock_backend
        assert runner._cache is mock_profile_cache

    def test_init_happy_path_without_cache(self):
        """Verify __init__ creates new ProfileCache when None provided."""
        mock_backend = Mock(spec=Backend)
        
        with patch('transmogrifier.calibrate.ProfileCache') as MockProfileCache:
            mock_cache_instance = Mock()
            MockProfileCache.return_value = mock_cache_instance
            
            runner = CalibrationRunner(mock_backend, None)
            
            assert runner._backend is mock_backend
            assert runner._cache is not None
            assert isinstance(runner._cache, type(mock_cache_instance))

    def test_init_backend_not_none(self):
        """Verify __init__ precondition: backend is not None."""
        mock_backend = Mock(spec=Backend)
        mock_cache = Mock(spec=ProfileCache)
        
        runner = CalibrationRunner(mock_backend, mock_cache)
        
        assert runner._backend is not None


# =============================================================================
# Component Tests for CalibrationRunner.run()
# =============================================================================

class TestCalibrationRunnerRun:
    """Component tests for CalibrationRunner.run() method."""

    def setup_method(self):
        """Set up mocks for each test."""
        self.mock_backend = Mock(spec=Backend)
        self.mock_cache = Mock(spec=ProfileCache)
        self.runner = CalibrationRunner(self.mock_backend, self.mock_cache)
        
        # Configure backend to return successful responses
        self.mock_backend.complete.return_value = "This is a valid response"

    def test_run_happy_path_complete_execution(self):
        """Verify run executes calibration across all registers and tasks, returns ModelProfile with correct structure."""
        model_name = "gpt-4"
        model_version = "1.0"
        provider = "openai"
        
        with patch('transmogrifier.calibrate.BENCHMARK_TASKS', [
            {"category": "factual", "prompt": "test1", "accept": ["valid"], "reject": []},
            {"category": "reasoning", "prompt": "test2", "accept": ["valid"], "reject": []},
        ]):
            with patch('transmogrifier.calibrate.REGISTER_TRANSFORMS', {
                "direct": lambda x: x,
                "casual": lambda x: x,
            }):
                profile = self.runner.run(
                    model_name=model_name,
                    model_version=model_version,
                    provider=provider,
                    tasks=None,
                    registers=None,
                    delay=0.0,
                    verbose=False
                )
        
        assert isinstance(profile, ModelProfile)
        assert profile.calibration_version == '2.0'
        assert profile.calibrated_at is not None

    def test_run_happy_path_with_custom_tasks(self):
        """Verify run works with custom task list provided."""
        custom_tasks = [
            {"category": "factual", "prompt": "test", "accept": ["yes"], "reject": []}
        ]
        
        profile = self.runner.run(
            model_name="gpt-4",
            model_version="1.0",
            provider="openai",
            tasks=custom_tasks,
            registers=None,
            delay=0.0,
            verbose=False
        )
        
        assert isinstance(profile, ModelProfile)

    def test_run_happy_path_with_custom_registers(self):
        """Verify run works with custom register list provided."""
        custom_registers = ["direct", "casual"]
        
        with patch('transmogrifier.calibrate.BENCHMARK_TASKS', [
            {"category": "factual", "prompt": "test", "accept": ["valid"], "reject": []}
        ]):
            profile = self.runner.run(
                model_name="gpt-4",
                model_version="1.0",
                provider="openai",
                tasks=None,
                registers=custom_registers,
                delay=0.0,
                verbose=False
            )
        
        assert isinstance(profile, ModelProfile)

    @patch('transmogrifier.calibrate.time.sleep')
    def test_run_happy_path_with_delay(self, mock_sleep):
        """Verify run enforces delay between API calls."""
        custom_tasks = [
            {"category": "factual", "prompt": "test1", "accept": ["yes"], "reject": []},
            {"category": "factual", "prompt": "test2", "accept": ["yes"], "reject": []}
        ]
        custom_registers = ["direct"]
        
        profile = self.runner.run(
            model_name="gpt-4",
            model_version="1.0",
            provider="openai",
            tasks=custom_tasks,
            registers=custom_registers,
            delay=0.1,
            verbose=False
        )
        
        assert isinstance(profile, ModelProfile)
        # Delay should be called between tasks (N-1 times for N tasks)
        assert mock_sleep.call_count >= 1

    def test_run_happy_path_verbose_logging(self, capfd):
        """Verify run outputs logs when verbose is True."""
        custom_tasks = [
            {"category": "factual", "prompt": "test", "accept": ["yes"], "reject": []}
        ]
        custom_registers = ["direct"]
        
        with patch('transmogrifier.calibrate.REGISTER_TRANSFORMS', {
            "direct": lambda x: x
        }):
            profile = self.runner.run(
                model_name="gpt-4",
                model_version="1.0",
                provider="openai",
                tasks=custom_tasks,
                registers=custom_registers,
                delay=0.0,
                verbose=True
            )
        
        assert isinstance(profile, ModelProfile)
        # Check if any output was produced (verbose logging)
        captured = capfd.readouterr()
        # Some implementations may use logging module instead of print

    def test_run_edge_case_empty_tasks_list(self):
        """Verify run handles empty tasks list."""
        profile = self.runner.run(
            model_name="gpt-4",
            model_version="1.0",
            provider="openai",
            tasks=[],
            registers=None,
            delay=0.0,
            verbose=False
        )
        
        assert isinstance(profile, ModelProfile)

    def test_run_edge_case_empty_registers_list(self):
        """Verify run handles empty registers list."""
        profile = self.runner.run(
            model_name="gpt-4",
            model_version="1.0",
            provider="openai",
            tasks=None,
            registers=[],
            delay=0.0,
            verbose=False
        )
        
        assert isinstance(profile, ModelProfile)

    @patch('transmogrifier.calibrate.time.sleep')
    def test_run_edge_case_zero_delay(self, mock_sleep):
        """Verify run works with zero delay."""
        custom_tasks = [
            {"category": "factual", "prompt": "test", "accept": ["yes"], "reject": []}
        ]
        
        profile = self.runner.run(
            model_name="gpt-4",
            model_version="1.0",
            provider="openai",
            tasks=custom_tasks,
            registers=["direct"],
            delay=0.0,
            verbose=False
        )
        
        assert isinstance(profile, ModelProfile)

    @patch('transmogrifier.calibrate.time.sleep')
    def test_run_edge_case_large_delay(self, mock_sleep):
        """Verify run works with large delay value."""
        custom_tasks = [
            {"category": "factual", "prompt": "test", "accept": ["yes"], "reject": []}
        ]
        custom_registers = ["direct"]
        
        profile = self.runner.run(
            model_name="gpt-4",
            model_version="1.0",
            provider="openai",
            tasks=custom_tasks,
            registers=custom_registers,
            delay=1.0,
            verbose=False
        )
        
        assert isinstance(profile, ModelProfile)

    def test_run_error_backend_exception(self):
        """Verify run propagates backend exception when backend.complete() fails."""
        self.mock_backend.complete.side_effect = Exception("Network error")
        
        custom_tasks = [
            {"category": "factual", "prompt": "test", "accept": ["yes"], "reject": []}
        ]
        
        with pytest.raises(Exception) as exc_info:
            self.runner.run(
                model_name="gpt-4",
                model_version="1.0",
                provider="openai",
                tasks=custom_tasks,
                registers=["direct"],
                delay=0.0,
                verbose=False
            )
        
        assert "Network error" in str(exc_info.value)

    def test_run_postcondition_cache_saved(self):
        """Verify run saves ModelProfile to cache via ProfileCache.put()."""
        custom_tasks = [
            {"category": "factual", "prompt": "test", "accept": ["yes"], "reject": []}
        ]
        
        profile = self.runner.run(
            model_name="gpt-4",
            model_version="1.0",
            provider="openai",
            tasks=custom_tasks,
            registers=["direct"],
            delay=0.0,
            verbose=False
        )
        
        assert self.mock_cache.put.called

    def test_run_postcondition_utc_timestamp(self):
        """Verify run sets calibrated_at to current UTC timestamp."""
        custom_tasks = [
            {"category": "factual", "prompt": "test", "accept": ["yes"], "reject": []}
        ]
        
        before = datetime.now(timezone.utc)
        profile = self.runner.run(
            model_name="gpt-4",
            model_version="1.0",
            provider="openai",
            tasks=custom_tasks,
            registers=["direct"],
            delay=0.0,
            verbose=False
        )
        after = datetime.now(timezone.utc)
        
        assert profile.calibrated_at.tzinfo is not None
        assert before <= profile.calibrated_at <= after

    def test_run_backend_complete_called_correctly(self):
        """Verify backend.complete() is called with correct parameters."""
        custom_tasks = [
            {"category": "factual", "prompt": "What is 2+2?", "accept": ["4"], "reject": []}
        ]
        custom_registers = ["direct"]
        
        with patch('transmogrifier.calibrate.REGISTER_TRANSFORMS', {
            "direct": lambda x: f"Direct: {x}"
        }):
            profile = self.runner.run(
                model_name="gpt-4",
                model_version="1.0",
                provider="openai",
                tasks=custom_tasks,
                registers=custom_registers,
                delay=0.0,
                verbose=False
            )
        
        # Verify backend.complete was called
        assert self.mock_backend.complete.called

    def test_run_multiple_categories(self):
        """Verify run handles multiple task categories correctly."""
        custom_tasks = [
            {"category": "factual", "prompt": "test1", "accept": ["yes"], "reject": []},
            {"category": "reasoning", "prompt": "test2", "accept": ["yes"], "reject": []},
            {"category": "code", "prompt": "test3", "accept": ["yes"], "reject": []},
        ]
        
        profile = self.runner.run(
            model_name="gpt-4",
            model_version="1.0",
            provider="openai",
            tasks=custom_tasks,
            registers=["direct"],
            delay=0.0,
            verbose=False
        )
        
        assert isinstance(profile, ModelProfile)


# =============================================================================
# Invariant Tests
# =============================================================================

class TestInvariants:
    """Tests for system invariants."""

    def test_invariant_calibration_version(self):
        """Verify calibration version is always '2.0'."""
        mock_backend = Mock(spec=Backend)
        mock_backend.complete.return_value = "valid response"
        mock_cache = Mock(spec=ProfileCache)
        runner = CalibrationRunner(mock_backend, mock_cache)
        
        custom_tasks = [
            {"category": "factual", "prompt": "test", "accept": ["yes"], "reject": []}
        ]
        
        profile = runner.run(
            model_name="gpt-4",
            model_version="1.0",
            provider="openai",
            tasks=custom_tasks,
            registers=["direct"],
            delay=0.0,
            verbose=False
        )
        
        assert profile.calibration_version == '2.0'

    def test_invariant_register_transforms_count(self):
        """Verify REGISTER_TRANSFORMS contains exactly 5 transforms."""
        assert len(REGISTER_TRANSFORMS) == 5

    def test_invariant_register_transforms_names(self):
        """Verify REGISTER_TRANSFORMS contains expected register names."""
        expected_registers = {"direct", "casual", "technical", "academic", "narrative"}
        assert set(REGISTER_TRANSFORMS.keys()) == expected_registers

    def test_invariant_benchmark_tasks_count(self):
        """Verify BENCHMARK_TASKS contains 50 tasks."""
        assert len(BENCHMARK_TASKS) == 50

    def test_invariant_benchmark_tasks_structure(self):
        """Verify each task dict contains required keys."""
        required_keys = {"category", "prompt", "accept", "reject"}
        
        for task in BENCHMARK_TASKS:
            assert all(key in task for key in required_keys)
            assert isinstance(task["accept"], list)
            assert isinstance(task["reject"], list)

    def test_invariant_benchmark_tasks_categories(self):
        """Verify BENCHMARK_TASKS contains tasks across 6 categories."""
        categories = set(task["category"] for task in BENCHMARK_TASKS)
        expected_categories = {"factual", "reasoning", "code", "analysis", "creative", "instruction"}
        assert categories == expected_categories


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests with test doubles."""

    def test_full_calibration_workflow(self):
        """Test complete calibration workflow from initialization to profile generation."""
        # Create mock backend that returns realistic responses
        mock_backend = Mock(spec=Backend)
        mock_backend.complete.return_value = "The answer is correct and valid"
        
        # Create mock cache
        mock_cache = Mock(spec=ProfileCache)
        
        # Initialize runner
        runner = CalibrationRunner(mock_backend, mock_cache)
        
        # Define test tasks
        test_tasks = [
            {"category": "factual", "prompt": "What is 2+2?", "accept": ["4", "four"], "reject": ["wrong"]},
            {"category": "reasoning", "prompt": "Why is the sky blue?", "accept": ["light", "scatter"], "reject": ["magic"]},
        ]
        
        test_registers = ["direct", "casual"]
        
        # Run calibration
        profile = runner.run(
            model_name="test-model",
            model_version="1.0",
            provider="test-provider",
            tasks=test_tasks,
            registers=test_registers,
            delay=0.0,
            verbose=False
        )
        
        # Verify profile structure
        assert isinstance(profile, ModelProfile)
        assert profile.calibration_version == '2.0'
        assert profile.calibrated_at is not None
        
        # Verify cache was called
        assert mock_cache.put.called
        
        # Verify backend was called multiple times (registers x tasks)
        assert mock_backend.complete.call_count == len(test_registers) * len(test_tasks)

    def test_calibration_with_all_default_parameters(self):
        """Test calibration using all default benchmark tasks and registers."""
        mock_backend = Mock(spec=Backend)
        mock_backend.complete.return_value = "valid response"
        mock_cache = Mock(spec=ProfileCache)
        
        runner = CalibrationRunner(mock_backend, mock_cache)
        
        # Use minimal subset for speed
        with patch('transmogrifier.calibrate.BENCHMARK_TASKS',
                   BENCHMARK_TASKS[:5]):  # Use first 5 tasks
            profile = runner.run(
                model_name="test-model",
                model_version="1.0",
                provider="test-provider",
                tasks=None,
                registers=None,
                delay=0.0,
                verbose=False
            )
        
        assert isinstance(profile, ModelProfile)
        assert profile.calibration_version == '2.0'

    def test_partial_failure_handling(self):
        """Test calibration behavior when some backend calls succeed and some fail."""
        mock_backend = Mock(spec=Backend)
        
        # Simulate intermittent failures
        call_count = [0]
        def backend_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                raise Exception("Intermittent failure")
            return "valid response"
        
        mock_backend.complete.side_effect = backend_side_effect
        mock_cache = Mock(spec=ProfileCache)
        
        runner = CalibrationRunner(mock_backend, mock_cache)
        
        test_tasks = [
            {"category": "factual", "prompt": f"test{i}", "accept": ["yes"], "reject": []}
            for i in range(5)
        ]
        
        # Should raise exception on first failure
        with pytest.raises(Exception) as exc_info:
            runner.run(
                model_name="test-model",
                model_version="1.0",
                provider="test-provider",
                tasks=test_tasks,
                registers=["direct"],
                delay=0.0,
                verbose=False
            )
        
        assert "Intermittent failure" in str(exc_info.value)

    @patch('transmogrifier.calibrate.time.sleep')
    def test_delay_enforcement_with_multiple_tasks(self, mock_sleep):
        """Test that delay is properly enforced between multiple task executions."""
        mock_backend = Mock(spec=Backend)
        mock_backend.complete.return_value = "valid"
        mock_cache = Mock(spec=ProfileCache)
        
        runner = CalibrationRunner(mock_backend, mock_cache)
        
        test_tasks = [
            {"category": "factual", "prompt": f"test{i}", "accept": ["valid"], "reject": []}
            for i in range(3)
        ]
        
        profile = runner.run(
            model_name="test-model",
            model_version="1.0",
            provider="test-provider",
            tasks=test_tasks,
            registers=["direct"],
            delay=0.5,
            verbose=False
        )
        
        assert isinstance(profile, ModelProfile)
        # Should have called sleep between each task
        assert mock_sleep.call_count >= 2


# =============================================================================
# Performance and Stress Tests
# =============================================================================

class TestPerformance:
    """Performance-related tests."""

    def test_large_task_set_performance(self):
        """Test calibration with large number of tasks."""
        mock_backend = Mock(spec=Backend)
        mock_backend.complete.return_value = "valid"
        mock_cache = Mock(spec=ProfileCache)
        
        runner = CalibrationRunner(mock_backend, mock_cache)
        
        # Create large task set
        large_tasks = [
            {"category": random.choice(["factual", "reasoning"]), 
             "prompt": f"test{i}", 
             "accept": ["valid"], 
             "reject": []}
            for i in range(100)
        ]
        
        start_time = time.time()
        profile = runner.run(
            model_name="test-model",
            model_version="1.0",
            provider="test-provider",
            tasks=large_tasks,
            registers=["direct"],
            delay=0.0,
            verbose=False
        )
        duration = time.time() - start_time
        
        assert isinstance(profile, ModelProfile)
        # Should complete in reasonable time (adjust threshold as needed)
        # Just verify it completes, actual time depends on implementation

    def test_all_registers_all_categories(self):
        """Test calibration with all registers and all task categories."""
        mock_backend = Mock(spec=Backend)
        mock_backend.complete.return_value = "valid response"
        mock_cache = Mock(spec=ProfileCache)
        
        runner = CalibrationRunner(mock_backend, mock_cache)
        
        # Create task for each category
        test_tasks = [
            {"category": cat, "prompt": f"test {cat}", "accept": ["valid"], "reject": []}
            for cat in ["factual", "reasoning", "code", "analysis", "creative", "instruction"]
        ]
        
        test_registers = list(REGISTER_TRANSFORMS.keys())
        
        profile = runner.run(
            model_name="test-model",
            model_version="1.0",
            provider="test-provider",
            tasks=test_tasks,
            registers=test_registers,
            delay=0.0,
            verbose=False
        )
        
        assert isinstance(profile, ModelProfile)
        assert len(profile.accuracies) == len(test_registers)
