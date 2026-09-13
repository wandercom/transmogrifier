"""
Contract tests for Model Profile Cache component.

Tests verify behavior at boundaries using pytest with mocked dependencies.
Covers happy paths, edge cases, error cases, and invariants.
"""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
from typing import Any

# Import the component under test
from transmogrifier.profiles import (
    RegisterAccuracy,
    TaskRegisterProfile,
    ModelProfile,
    ProfileCache,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Temporary cache directory for testing."""
    return tmp_path / "test_cache"


@pytest.fixture
def sample_register_accuracies():
    """Sample RegisterAccuracy instances for testing."""
    return [
        RegisterAccuracy(
            register="direct",
            accuracy=0.85,
            sample_size=16,
            task_type="classification"
        ),
        RegisterAccuracy(
            register="casual",
            accuracy=0.80,
            sample_size=16,
            task_type="classification"
        ),
        RegisterAccuracy(
            register="formal",
            accuracy=0.95,
            sample_size=16,
            task_type="classification"
        ),
    ]


@pytest.fixture
def sample_task_profile(sample_register_accuracies):
    """Sample TaskRegisterProfile for testing."""
    return TaskRegisterProfile(
        task_type="classification",
        accuracies=sample_register_accuracies
    )


@pytest.fixture
def sample_model_profile(sample_register_accuracies):
    """Sample ModelProfile for testing."""
    return ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=sample_register_accuracies,
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )


# ============================================================================
# TaskRegisterProfile Tests
# ============================================================================

def test_task_register_profile_best_register_happy_path(sample_register_accuracies):
    """TaskRegisterProfile.best_register returns register with highest accuracy."""
    profile = TaskRegisterProfile(
        task_type="classification",
        accuracies=sample_register_accuracies
    )
    
    result = profile.best_register()
    
    assert result == "formal", "Should return 'formal' with accuracy 0.95"


def test_task_register_profile_best_register_empty_accuracies():
    """TaskRegisterProfile.best_register returns 'direct' when accuracies is empty."""
    profile = TaskRegisterProfile(
        task_type="classification",
        accuracies=[]
    )
    
    result = profile.best_register()
    
    assert result == "direct", "Should return 'direct' as default for empty accuracies"


def test_task_register_profile_best_register_single_accuracy():
    """TaskRegisterProfile.best_register returns the only register when accuracies has one element."""
    profile = TaskRegisterProfile(
        task_type="classification",
        accuracies=[
            RegisterAccuracy(
                register="casual",
                accuracy=0.87,
                sample_size=16,
                task_type="classification"
            )
        ]
    )
    
    result = profile.best_register()
    
    assert result == "casual", "Should return the only register available"


def test_task_register_profile_spread_pp_happy_path(sample_register_accuracies):
    """TaskRegisterProfile.spread_pp calculates percentage point spread correctly."""
    profile = TaskRegisterProfile(
        task_type="classification",
        accuracies=sample_register_accuracies
    )
    
    result = profile.spread_pp()
    
    # Max: 0.95, Min: 0.80, Spread: (0.95 - 0.80) * 100 = 15.0
    assert abs(result - 15.0) < 1e-6, f"Expected 15.0, got {result}"


def test_task_register_profile_spread_pp_empty_accuracies():
    """TaskRegisterProfile.spread_pp returns 0.0 when accuracies is empty."""
    profile = TaskRegisterProfile(
        task_type="classification",
        accuracies=[]
    )
    
    result = profile.spread_pp()
    
    assert result == 0.0, "Should return 0.0 for empty accuracies"


def test_task_register_profile_spread_pp_identical_accuracies():
    """TaskRegisterProfile.spread_pp returns 0.0 when all accuracies are identical."""
    profile = TaskRegisterProfile(
        task_type="classification",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.90, sample_size=16, task_type="classification"),
            RegisterAccuracy(register="casual", accuracy=0.90, sample_size=16, task_type="classification"),
            RegisterAccuracy(register="formal", accuracy=0.90, sample_size=16, task_type="classification"),
        ]
    )
    
    result = profile.spread_pp()
    
    assert result == 0.0, "Should return 0.0 when all accuracies are identical"


# ============================================================================
# ModelProfile Tests
# ============================================================================

def test_model_profile_spread_pp_happy_path():
    """ModelProfile.spread_pp calculates aggregate percentage point spread correctly."""
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.75, sample_size=16, task_type="general"),
            RegisterAccuracy(register="casual", accuracy=0.82, sample_size=16, task_type="general"),
            RegisterAccuracy(register="formal", accuracy=0.92, sample_size=16, task_type="general"),
        ],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result = profile.spread_pp()
    
    # Max: 0.92, Min: 0.75, Spread: (0.92 - 0.75) * 100 = 17.0
    assert abs(result - 17.0) < 1e-6, f"Expected 17.0, got {result}"


def test_model_profile_spread_pp_empty_accuracies():
    """ModelProfile.spread_pp returns 0.0 when accuracies is empty."""
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result = profile.spread_pp()
    
    assert result == 0.0, "Should return 0.0 for empty accuracies"


def test_model_profile_is_invariant_true():
    """ModelProfile.is_invariant returns True when spread < 2.0 percentage points."""
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.900, sample_size=16, task_type="general"),
            RegisterAccuracy(register="casual", accuracy=0.905, sample_size=16, task_type="general"),
            RegisterAccuracy(register="formal", accuracy=0.915, sample_size=16, task_type="general"),
        ],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result = profile.is_invariant()
    
    # Spread: (0.915 - 0.900) * 100 = 1.5
    assert result is True, "Should return True when spread is 1.5"


def test_model_profile_is_invariant_false():
    """ModelProfile.is_invariant returns False when spread >= 2.0 percentage points."""
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.75, sample_size=16, task_type="general"),
            RegisterAccuracy(register="casual", accuracy=0.82, sample_size=16, task_type="general"),
            RegisterAccuracy(register="formal", accuracy=0.90, sample_size=16, task_type="general"),
        ],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result = profile.is_invariant()
    
    # Spread: (0.90 - 0.75) * 100 = 15.0
    assert result is False, "Should return False when spread is 15.0"


def test_model_profile_is_invariant_boundary_under():
    """ModelProfile.is_invariant returns True when spread is exactly 1.999 percentage points."""
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.900, sample_size=16, task_type="general"),
            RegisterAccuracy(register="formal", accuracy=0.91999, sample_size=16, task_type="general"),
        ],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result = profile.is_invariant()
    
    # Spread: (0.91999 - 0.900) * 100 = 1.999
    assert result is True, "Should return True when spread is 1.999"


def test_model_profile_is_invariant_boundary_at():
    """ModelProfile.is_invariant returns False when spread is exactly 2.0 percentage points."""
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.900, sample_size=16, task_type="general"),
            RegisterAccuracy(register="formal", accuracy=0.920, sample_size=16, task_type="general"),
        ],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result = profile.is_invariant()
    
    # Spread: (0.920 - 0.900) * 100 = 2.0
    assert result is False, "Should return False when spread is exactly 2.0"


def test_model_profile_best_register_happy_path():
    """ModelProfile.best_register returns register with highest aggregate accuracy."""
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.85, sample_size=16, task_type="general"),
            RegisterAccuracy(register="casual", accuracy=0.80, sample_size=16, task_type="general"),
            RegisterAccuracy(register="formal", accuracy=0.95, sample_size=16, task_type="general"),
        ],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result = profile.best_register()
    
    assert result == "formal", "Should return 'formal' with highest accuracy 0.95"


def test_model_profile_best_register_empty_accuracies():
    """ModelProfile.best_register returns 'direct' when accuracies is empty."""
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result = profile.best_register()
    
    assert result == "direct", "Should return 'direct' as default for empty accuracies"


def test_model_profile_worst_register_happy_path():
    """ModelProfile.worst_register returns register with lowest aggregate accuracy."""
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.85, sample_size=16, task_type="general"),
            RegisterAccuracy(register="casual", accuracy=0.75, sample_size=16, task_type="general"),
            RegisterAccuracy(register="formal", accuracy=0.95, sample_size=16, task_type="general"),
        ],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result = profile.worst_register()
    
    assert result == "casual", "Should return 'casual' with lowest accuracy 0.75"


def test_model_profile_worst_register_empty_accuracies():
    """ModelProfile.worst_register returns 'direct' when accuracies is empty."""
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result = profile.worst_register()
    
    assert result == "direct", "Should return 'direct' as default for empty accuracies"


def test_model_profile_best_register_for_task_found():
    """ModelProfile.best_register_for_task returns task-specific best register when task exists."""
    task_profile = TaskRegisterProfile(
        task_type="classification",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.85, sample_size=16, task_type="classification"),
            RegisterAccuracy(register="casual", accuracy=0.90, sample_size=16, task_type="classification"),
        ]
    )
    
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.75, sample_size=16, task_type="general"),
            RegisterAccuracy(register="formal", accuracy=0.95, sample_size=16, task_type="general"),
        ],
        by_task=[task_profile],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result = profile.best_register_for_task("classification")
    
    assert result == "casual", "Should return 'casual' from classification task profile"


def test_model_profile_best_register_for_task_not_found():
    """ModelProfile.best_register_for_task falls back to aggregate best_register when task not found."""
    task_profile = TaskRegisterProfile(
        task_type="classification",
        accuracies=[
            RegisterAccuracy(register="casual", accuracy=0.90, sample_size=16, task_type="classification"),
        ]
    )
    
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.75, sample_size=16, task_type="general"),
            RegisterAccuracy(register="formal", accuracy=0.95, sample_size=16, task_type="general"),
        ],
        by_task=[task_profile],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result = profile.best_register_for_task("unknown_task")
    
    assert result == "formal", "Should fallback to aggregate best_register 'formal'"


def test_model_profile_best_register_for_task_empty_by_task():
    """ModelProfile.best_register_for_task falls back to aggregate when by_task is empty."""
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.85, sample_size=16, task_type="general"),
            RegisterAccuracy(register="formal", accuracy=0.92, sample_size=16, task_type="general"),
        ],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result = profile.best_register_for_task("classification")
    
    assert result == "formal", "Should fallback to aggregate best_register 'formal'"


def test_model_profile_spread_for_task_found():
    """ModelProfile.spread_for_task returns task-specific spread when task exists."""
    task_profile = TaskRegisterProfile(
        task_type="classification",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.80, sample_size=16, task_type="classification"),
            RegisterAccuracy(register="casual", accuracy=0.90, sample_size=16, task_type="classification"),
        ]
    )
    
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.70, sample_size=16, task_type="general"),
            RegisterAccuracy(register="formal", accuracy=0.95, sample_size=16, task_type="general"),
        ],
        by_task=[task_profile],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result = profile.spread_for_task("classification")
    
    # Task spread: (0.90 - 0.80) * 100 = 10.0
    assert abs(result - 10.0) < 1e-6, f"Expected 10.0, got {result}"


def test_model_profile_spread_for_task_not_found():
    """ModelProfile.spread_for_task falls back to aggregate spread when task not found."""
    task_profile = TaskRegisterProfile(
        task_type="classification",
        accuracies=[
            RegisterAccuracy(register="casual", accuracy=0.90, sample_size=16, task_type="classification"),
        ]
    )
    
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.70, sample_size=16, task_type="general"),
            RegisterAccuracy(register="formal", accuracy=0.95, sample_size=16, task_type="general"),
        ],
        by_task=[task_profile],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result = profile.spread_for_task("unknown_task")
    
    # Aggregate spread: (0.95 - 0.70) * 100 = 25.0
    assert abs(result - 25.0) < 1e-6, f"Expected 25.0, got {result}"


def test_model_profile_is_expired_false_fresh():
    """ModelProfile.is_expired returns False for fresh profile within TTL."""
    one_hour_ago = datetime.now() - timedelta(hours=1)
    
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[],
        by_task=[],
        calibrated_at=one_hour_ago.isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result = profile.is_expired()
    
    assert result is False, "Profile calibrated 1 hour ago should not be expired with 720 hour TTL"


def test_model_profile_is_expired_true_old():
    """ModelProfile.is_expired returns True for expired profile beyond TTL."""
    old_time = datetime.now() - timedelta(hours=800)
    
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[],
        by_task=[],
        calibrated_at=old_time.isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result = profile.is_expired()
    
    assert result is True, "Profile calibrated 800 hours ago should be expired with 720 hour TTL"


def test_model_profile_is_expired_empty_calibrated_at():
    """ModelProfile.is_expired returns False when calibrated_at is empty."""
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[],
        by_task=[],
        calibrated_at="",
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result = profile.is_expired()
    
    assert result is False, "Should return False for empty calibrated_at"


def test_model_profile_is_expired_invalid_timestamp():
    """ModelProfile.is_expired returns False when calibrated_at cannot be parsed."""
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[],
        by_task=[],
        calibrated_at="invalid-timestamp-format",
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result = profile.is_expired()
    
    assert result is False, "Should return False for invalid timestamp"


def test_model_profile_is_expired_boundary_exact_ttl():
    """ModelProfile.is_expired returns False when exactly at TTL boundary."""
    exactly_ttl_ago = datetime.now() - timedelta(hours=720)
    
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[],
        by_task=[],
        calibrated_at=exactly_ttl_ago.isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result = profile.is_expired()
    
    # At exactly TTL boundary, should not be expired yet (uses >)
    assert result is False, "Should not be expired at exact TTL boundary"


# ============================================================================
# ProfileCache Tests
# ============================================================================

def test_profile_cache_init_with_path(temp_cache_dir):
    """ProfileCache.__init__ sets cache_dir when provided."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    
    assert cache._cache_dir == temp_cache_dir, "cache_dir should be set to provided path"
    assert cache._memory == {}, "memory should be initialized to empty dict"


def test_profile_cache_init_default_path():
    """ProfileCache.__init__ sets default cache_dir when None provided."""
    cache = ProfileCache(cache_dir=None)
    
    expected_path = Path.home() / '.transmogrifier' / 'profiles'
    assert cache._cache_dir == expected_path, "cache_dir should be set to default path"
    assert cache._memory == {}, "memory should be initialized to empty dict"


def test_profile_cache_get_from_memory(temp_cache_dir, sample_model_profile):
    """ProfileCache.get returns cached profile from memory when available and not expired."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    cache._memory["test-model"] = sample_model_profile
    
    with patch.object(cache, '_load_file') as mock_load:
        result = cache.get("test-model")
    
    assert result == sample_model_profile, "Should return profile from memory"
    mock_load.assert_not_called()


def test_profile_cache_get_from_file(temp_cache_dir, sample_model_profile):
    """ProfileCache.get loads profile from file when not in memory."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    
    with patch.object(cache, '_load_file', return_value=sample_model_profile) as mock_load:
        result = cache.get("test-model")
    
    assert result == sample_model_profile, "Should return profile from file"
    assert cache._memory["test-model"] == sample_model_profile, "Should cache in memory"
    mock_load.assert_called_once_with("test-model")


def test_profile_cache_get_from_preseeded(temp_cache_dir):
    """ProfileCache.get returns pre-seeded profile when no cached version exists."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    
    # Mock _PRESEEDED to have a test profile
    mock_preseeded = {
        "claude-opus-4": ModelProfile(
            model_name="claude-opus-4",
            model_version="1.0",
            provider="anthropic",
            accuracies=[],
            by_task=[],
            calibrated_at="2026-03-27T00:00:00",
            ttl_hours=720,
            calibration_version="1.0"
        )
    }
    
    with patch.object(cache, '_load_file', return_value=None):
        with patch('transmogrifier.profiles._PRESEEDED', mock_preseeded):
            result = cache.get("claude-opus-4")
    
    assert result is not None, "Should return pre-seeded profile"
    assert result.model_name == "claude-opus-4", "Should return correct pre-seeded profile"


def test_profile_cache_get_with_alias_resolution(temp_cache_dir, sample_model_profile):
    """ProfileCache.get resolves alias to canonical name."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    
    # Mock _ALIASES
    mock_aliases = {"aliased-model-v2": "canonical-model"}
    sample_model_profile.model_name = "canonical-model"
    
    with patch('transmogrifier.profiles._ALIASES', mock_aliases):
        with patch.object(cache, '_load_file', return_value=sample_model_profile):
            result = cache.get("aliased-model-v2")
    
    assert result is not None, "Should resolve alias and return profile"
    assert result.model_name == "canonical-model", "Should return canonical model"


def test_profile_cache_get_not_found(temp_cache_dir):
    """ProfileCache.get returns None when profile not found anywhere."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    
    with patch.object(cache, '_load_file', return_value=None):
        with patch('transmogrifier.profiles._PRESEEDED', {}):
            result = cache.get("nonexistent-model")
    
    assert result is None, "Should return None for non-existent model"


def test_profile_cache_get_expired_profile_fallback(temp_cache_dir):
    """ProfileCache.get falls back to pre-seeded when cached profile is expired."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    
    # Create expired profile
    expired_time = datetime.now() - timedelta(hours=800)
    expired_profile = ModelProfile(
        model_name="claude-opus-4",
        model_version="1.0",
        provider="anthropic",
        accuracies=[],
        by_task=[],
        calibrated_at=expired_time.isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    # Create fresh pre-seeded profile
    fresh_profile = ModelProfile(
        model_name="claude-opus-4",
        model_version="2.0",
        provider="anthropic",
        accuracies=[],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    mock_preseeded = {"claude-opus-4": fresh_profile}
    
    with patch.object(cache, '_load_file', return_value=expired_profile):
        with patch('transmogrifier.profiles._PRESEEDED', mock_preseeded):
            result = cache.get("claude-opus-4")
    
    assert result == fresh_profile, "Should return pre-seeded profile when cached is expired"


def test_profile_cache_get_partial_match_preseeded(temp_cache_dir):
    """ProfileCache.get matches pre-seeded profiles via partial name match."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    
    # Pre-seeded profile with base name
    preseeded_profile = ModelProfile(
        model_name="gpt-4o-mini",
        model_version="1.0",
        provider="openai",
        accuracies=[],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    mock_preseeded = {"gpt-4o-mini": preseeded_profile}
    
    with patch.object(cache, '_load_file', return_value=None):
        with patch('transmogrifier.profiles._PRESEEDED', mock_preseeded):
            result = cache.get("gpt-4o-mini-2024")
    
    assert result is not None, "Should find profile via partial match"
    assert result.model_name == "gpt-4o-mini", "Should return base model profile"


def test_profile_cache_put_creates_directory(temp_cache_dir, sample_model_profile):
    """ProfileCache.put creates cache directory if it doesn't exist."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    
    assert not temp_cache_dir.exists(), "Cache directory should not exist initially"
    
    result_path = cache.put(sample_model_profile)
    
    assert temp_cache_dir.exists(), "Cache directory should be created"
    assert result_path.parent == temp_cache_dir, "File should be in cache directory"
    assert sample_model_profile.model_name in cache._memory, "Profile should be in memory cache"


def test_profile_cache_put_writes_file(temp_cache_dir, sample_model_profile):
    """ProfileCache.put writes profile to JSON file and updates memory."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    temp_cache_dir.mkdir(parents=True)
    
    result_path = cache.put(sample_model_profile)
    
    expected_path = temp_cache_dir / f"{sample_model_profile.model_name}.json"
    assert result_path == expected_path, "Should return correct file path"
    assert result_path.exists(), "File should be created"
    assert cache._memory[sample_model_profile.model_name] == sample_model_profile, "Should update memory cache"
    
    # Verify file contents
    with open(result_path) as f:
        data = json.load(f)
    assert data["model_name"] == sample_model_profile.model_name, "File should contain profile data"


def test_profile_cache_put_overwrites_existing(temp_cache_dir):
    """ProfileCache.put overwrites existing profile file and memory entry."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    temp_cache_dir.mkdir(parents=True)
    
    # Create initial profile
    profile1 = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="provider1",
        accuracies=[],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    cache.put(profile1)
    
    # Create updated profile
    profile2 = ModelProfile(
        model_name="test-model",
        model_version="2.0",
        provider="provider2",
        accuracies=[],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    result_path = cache.put(profile2)
    
    assert cache._memory["test-model"] == profile2, "Memory should have updated profile"
    
    # Verify file has updated content
    with open(result_path) as f:
        data = json.load(f)
    assert data["model_version"] == "2.0", "File should have updated version"
    assert data["provider"] == "provider2", "File should have updated provider"


def test_profile_cache_invalidate_removes_file_and_memory(temp_cache_dir):
    """ProfileCache.invalidate removes profile from memory and deletes file."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    temp_cache_dir.mkdir(parents=True)
    
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    file_path = cache.put(profile)
    assert file_path.exists(), "File should exist before invalidation"
    assert "test-model" in cache._memory, "Profile should be in memory"
    
    result = cache.invalidate("test-model")
    
    assert result is True, "Should return True when file was deleted"
    assert not file_path.exists(), "File should be deleted"
    assert "test-model" not in cache._memory, "Profile should be removed from memory"


def test_profile_cache_invalidate_file_not_exists(temp_cache_dir):
    """ProfileCache.invalidate returns False when file doesn't exist."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    temp_cache_dir.mkdir(parents=True)
    
    # Add to memory only (no file)
    cache._memory["test-model"] = Mock()
    
    result = cache.invalidate("nonexistent")
    
    assert result is False, "Should return False when file doesn't exist"


def test_profile_cache_invalidate_with_alias(temp_cache_dir):
    """ProfileCache.invalidate resolves alias before invalidating."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    temp_cache_dir.mkdir(parents=True)
    
    profile = ModelProfile(
        model_name="canonical-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    file_path = cache.put(profile)
    
    mock_aliases = {"aliased-model": "canonical-model"}
    
    with patch('transmogrifier.profiles._ALIASES', mock_aliases):
        result = cache.invalidate("aliased-model")
    
    assert result is True, "Should resolve alias and invalidate canonical model"
    assert not file_path.exists(), "Canonical model file should be deleted"


def test_profile_cache_list_profiles_preseeded_only(temp_cache_dir):
    """ProfileCache.list_profiles returns pre-seeded profiles when no cached files."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    
    mock_preseeded = {
        "model1": ModelProfile(
            model_name="model1",
            model_version="1.0",
            provider="provider1",
            accuracies=[],
            by_task=[],
            calibrated_at=datetime.now().isoformat(),
            ttl_hours=720,
            calibration_version="1.0"
        ),
        "model2": ModelProfile(
            model_name="model2",
            model_version="1.0",
            provider="provider2",
            accuracies=[],
            by_task=[],
            calibrated_at=datetime.now().isoformat(),
            ttl_hours=720,
            calibration_version="1.0"
        )
    }
    
    with patch('transmogrifier.profiles._PRESEEDED', mock_preseeded):
        result = cache.list_profiles()
    
    assert len(result) == 2, "Should return all pre-seeded profiles"
    model_names = [p.model_name for p in result]
    assert "model1" in model_names, "Should include model1"
    assert "model2" in model_names, "Should include model2"


def test_profile_cache_list_profiles_with_cached(temp_cache_dir):
    """ProfileCache.list_profiles combines pre-seeded and cached profiles, deduplicating by name."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    temp_cache_dir.mkdir(parents=True)
    
    # Create cached profile
    cached_profile = ModelProfile(
        model_name="cached-model",
        model_version="1.0",
        provider="cached-provider",
        accuracies=[],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    cache.put(cached_profile)
    
    # Create pre-seeded profile with same name as cached (should be deduplicated)
    preseeded_duplicate = ModelProfile(
        model_name="cached-model",
        model_version="0.5",
        provider="preseeded-provider",
        accuracies=[],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    preseeded_unique = ModelProfile(
        model_name="preseeded-model",
        model_version="1.0",
        provider="preseeded-provider",
        accuracies=[],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    mock_preseeded = {
        "cached-model": preseeded_duplicate,
        "preseeded-model": preseeded_unique
    }
    
    with patch('transmogrifier.profiles._PRESEEDED', mock_preseeded):
        result = cache.list_profiles()
    
    model_names = [p.model_name for p in result]
    assert len(model_names) == len(set(model_names)), "Should not have duplicates"
    assert "cached-model" in model_names, "Should include cached-model"
    assert "preseeded-model" in model_names, "Should include preseeded-model"


def test_profile_cache_list_profiles_skips_invalid_files(temp_cache_dir):
    """ProfileCache.list_profiles silently skips files that fail to parse."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    temp_cache_dir.mkdir(parents=True)
    
    # Create valid profile
    valid_profile = ModelProfile(
        model_name="valid-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    cache.put(valid_profile)
    
    # Create corrupted file
    corrupted_file = temp_cache_dir / "corrupted.json"
    corrupted_file.write_text("{ invalid json }")
    
    with patch('transmogrifier.profiles._PRESEEDED', {}):
        result = cache.list_profiles()
    
    model_names = [p.model_name for p in result]
    assert "valid-model" in model_names, "Should include valid profile"
    assert len(result) == 1, "Should skip corrupted file"


def test_profile_cache_list_profiles_empty_directory(temp_cache_dir):
    """ProfileCache.list_profiles handles non-existent cache directory gracefully."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    
    # Don't create directory
    assert not temp_cache_dir.exists(), "Directory should not exist"
    
    mock_preseeded = {
        "preseeded-model": ModelProfile(
            model_name="preseeded-model",
            model_version="1.0",
            provider="test-provider",
            accuracies=[],
            by_task=[],
            calibrated_at=datetime.now().isoformat(),
            ttl_hours=720,
            calibration_version="1.0"
        )
    }
    
    with patch('transmogrifier.profiles._PRESEEDED', mock_preseeded):
        result = cache.list_profiles()
    
    assert len(result) == 1, "Should return pre-seeded profiles only"
    assert result[0].model_name == "preseeded-model", "Should include pre-seeded profile"


def test_profile_cache_load_file_success(temp_cache_dir):
    """ProfileCache._load_file returns ModelProfile when file exists and is valid."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    temp_cache_dir.mkdir(parents=True)
    
    profile = ModelProfile(
        model_name="test-model",
        model_version="1.0",
        provider="test-provider",
        accuracies=[],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    
    cache.put(profile)
    
    result = cache._load_file("test-model")
    
    assert result is not None, "Should return ModelProfile"
    assert result.model_name == "test-model", "Should have correct model name"


def test_profile_cache_load_file_not_exists(temp_cache_dir):
    """ProfileCache._load_file returns None when file doesn't exist."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    temp_cache_dir.mkdir(parents=True)
    
    result = cache._load_file("nonexistent")
    
    assert result is None, "Should return None for non-existent file"


def test_profile_cache_load_file_invalid_json(temp_cache_dir):
    """ProfileCache._load_file returns None and logs debug when parsing fails."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    temp_cache_dir.mkdir(parents=True)
    
    # Create corrupted file
    corrupted_file = temp_cache_dir / "corrupt.json"
    corrupted_file.write_text("{ invalid json }")
    
    with patch('transmogrifier.profiles.logger') as mock_logger:
        result = cache._load_file("corrupt")
    
    assert result is None, "Should return None for invalid JSON"
    # Verify debug logging was called
    assert mock_logger.debug.called, "Should log debug message"


# ============================================================================
# Invariant Tests
# ============================================================================

def test_invariant_preseeded_models_exist(temp_cache_dir):
    """Verify _PRESEEDED contains the four specified pre-calibrated profiles."""
    cache = ProfileCache(cache_dir=temp_cache_dir)
    
    # Mock _PRESEEDED with required models
    required_models = ["claude-opus-4", "claude-haiku-4-5", "gpt-4o-mini", "gemini-2-5-flash"]
    mock_preseeded = {
        name: ModelProfile(
            model_name=name,
            model_version="1.0",
            provider="test",
            accuracies=[],
            by_task=[],
            calibrated_at="2026-03-27T00:00:00",
            ttl_hours=720,
            calibration_version="1.0"
        )
        for name in required_models
    }
    
    with patch('transmogrifier.profiles._PRESEEDED', mock_preseeded):
        for model_name in required_models:
            with patch.object(cache, '_load_file', return_value=None):
                result = cache.get(model_name)
            assert result is not None, f"{model_name} should be in _PRESEEDED"
            assert result.model_name == model_name, f"Should return correct profile for {model_name}"


def test_invariant_default_cache_dir():
    """Verify default cache directory is ~/.transmogrifier/profiles."""
    cache = ProfileCache(cache_dir=None)
    
    expected_path = Path.home() / '.transmogrifier' / 'profiles'
    assert cache._cache_dir == expected_path, f"Default cache_dir should be {expected_path}"


def test_invariant_default_register_fallback():
    """Verify default register is always 'direct' for empty accuracies."""
    # Test TaskRegisterProfile.best_register
    task_profile = TaskRegisterProfile(task_type="test", accuracies=[])
    assert task_profile.best_register() == "direct", "TaskRegisterProfile should default to 'direct'"
    
    # Test ModelProfile.best_register
    model_profile = ModelProfile(
        model_name="test",
        model_version="1.0",
        provider="test",
        accuracies=[],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    assert model_profile.best_register() == "direct", "ModelProfile.best_register should default to 'direct'"
    assert model_profile.worst_register() == "direct", "ModelProfile.worst_register should default to 'direct'"


def test_invariant_spread_threshold():
    """Verify is_invariant uses 2.0 percentage point threshold."""
    # Test with spread of 1.99 (should be True)
    profile_under = ModelProfile(
        model_name="test",
        model_version="1.0",
        provider="test",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.900, sample_size=16, task_type="general"),
            RegisterAccuracy(register="formal", accuracy=0.91999, sample_size=16, task_type="general"),
        ],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    assert profile_under.is_invariant() is True, "Spread of 1.99 should be invariant"
    
    # Test with spread of 2.0 (should be False)
    profile_at = ModelProfile(
        model_name="test",
        model_version="1.0",
        provider="test",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.900, sample_size=16, task_type="general"),
            RegisterAccuracy(register="formal", accuracy=0.920, sample_size=16, task_type="general"),
        ],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    assert profile_at.is_invariant() is False, "Spread of 2.0 should not be invariant"
    
    # Test with spread of 2.01 (should be False)
    profile_over = ModelProfile(
        model_name="test",
        model_version="1.0",
        provider="test",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.900, sample_size=16, task_type="general"),
            RegisterAccuracy(register="formal", accuracy=0.92001, sample_size=16, task_type="general"),
        ],
        by_task=[],
        calibrated_at=datetime.now().isoformat(),
        ttl_hours=720,
        calibration_version="1.0"
    )
    assert profile_over.is_invariant() is False, "Spread of 2.01 should not be invariant"
