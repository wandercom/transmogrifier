"""
Contract test suite for transmogrifier.kindex_integration

This test suite verifies the Kindex integration module against its contract,
testing initialization, availability checking, error handling, state management,
and cleanup operations.

Test Organization:
1. Initialization & Error Paths: Tests for _init_store() covering success and
   both error cases (kindex_import_failure, kindex_initialization_failure)
2. State & Availability: Tests for is_available() covering memoization, state
   transitions, and _checked flag mutation
3. Cleanup & Idempotency: Tests for close() covering initialized/uninitialized
   states and idempotent behavior
"""

import pytest
import sys
from unittest.mock import Mock, MagicMock, patch, call
from typing import Any


# Import the module under test
# The actual module path should be adjusted based on the project structure
import transmogrifier.kindex_integration as kindex_integration


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset module state before and after each test."""
    # Save original state
    original_store = getattr(kindex_integration, '_store', None)
    original_checked = getattr(kindex_integration, '_checked', False)
    
    # Reset to uninitialized state
    kindex_integration._store = None
    kindex_integration._checked = False
    
    yield
    
    # Restore original state after test
    kindex_integration._store = original_store
    kindex_integration._checked = original_checked


@pytest.fixture
def mock_logger():
    """Mock the logger to verify logging behavior."""
    with patch.object(kindex_integration, 'logger') as mock_log:
        yield mock_log


@pytest.fixture
def mock_kindex_available(monkeypatch):
    """Mock kindex modules as available and functional."""
    mock_config_module = MagicMock()
    mock_store_module = MagicMock()
    mock_config_obj = MagicMock()
    mock_store_instance = MagicMock()
    
    mock_config_module.load_config.return_value = mock_config_obj
    mock_store_module.Store.return_value = mock_store_instance
    
    # Simulate successful imports
    monkeypatch.setitem(sys.modules, 'kindex.config', mock_config_module)
    monkeypatch.setitem(sys.modules, 'kindex.store', mock_store_module)
    
    return {
        'config_module': mock_config_module,
        'store_module': mock_store_module,
        'config_obj': mock_config_obj,
        'store_instance': mock_store_instance
    }


@pytest.fixture
def mock_kindex_config_import_failure(monkeypatch):
    """Mock kindex.config as unavailable (import failure)."""
    # Remove kindex modules if they exist
    if 'kindex.config' in sys.modules:
        monkeypatch.delitem(sys.modules, 'kindex.config')
    if 'kindex.store' in sys.modules:
        monkeypatch.delitem(sys.modules, 'kindex.store')
    
    # Make import fail by not setting the module
    return None


@pytest.fixture
def mock_kindex_store_import_failure(monkeypatch):
    """Mock kindex.store as unavailable (import failure)."""
    mock_config_module = MagicMock()
    
    # config is available but store is not
    monkeypatch.setitem(sys.modules, 'kindex.config', mock_config_module)
    
    if 'kindex.store' in sys.modules:
        monkeypatch.delitem(sys.modules, 'kindex.store')
    
    return None


# =============================================================================
# Suite 1: Initialization & Error Paths
# =============================================================================

def test_init_store_success(mock_kindex_available, mock_logger):
    """
    Happy path: _init_store successfully imports kindex modules and creates Store singleton.
    
    Verifies:
    - Returns True
    - _store is set to Store instance
    - logger.debug may be called for informational purposes
    """
    result = kindex_integration._init_store()
    
    assert result is True, "_init_store() should return True on success"
    assert kindex_integration._store is not None, "_store should be set"
    assert kindex_integration._store == mock_kindex_available['store_instance'], \
        "_store should be the Store instance"
    
    # Verify load_config and Store were called
    mock_kindex_available['config_module'].load_config.assert_called_once()
    mock_kindex_available['store_module'].Store.assert_called_once()


def test_init_store_import_failure_config(mock_logger):
    """
    Error case: _init_store fails when kindex.config cannot be imported.
    
    Verifies:
    - Returns False
    - _store remains None
    - logger.debug called with exception info
    """
    # Ensure kindex modules are not available
    if 'kindex.config' in sys.modules:
        del sys.modules['kindex.config']
    if 'kindex.store' in sys.modules:
        del sys.modules['kindex.store']
    
    result = kindex_integration._init_store()
    
    assert result is False, "_init_store() should return False on import failure"
    assert kindex_integration._store is None, "_store should remain None"
    
    # Verify debug logging occurred
    assert mock_logger.debug.called, "logger.debug should be called on failure"


def test_init_store_import_failure_store(mock_logger, monkeypatch):
    """
    Error case: _init_store fails when kindex.store cannot be imported.
    
    Verifies:
    - Returns False
    - _store remains None
    - logger.debug called with exception info
    """
    # Make config available but not store
    mock_config_module = MagicMock()
    monkeypatch.setitem(sys.modules, 'kindex.config', mock_config_module)
    
    # Ensure store module is not available
    if 'kindex.store' in sys.modules:
        del sys.modules['kindex.store']
    
    result = kindex_integration._init_store()
    
    assert result is False, "_init_store() should return False on store import failure"
    assert kindex_integration._store is None, "_store should remain None"
    
    # Verify debug logging occurred
    assert mock_logger.debug.called, "logger.debug should be called on failure"


def test_init_store_initialization_failure_load_config(mock_logger, monkeypatch):
    """
    Error case: _init_store fails when load_config() raises exception.
    
    Verifies:
    - Returns False
    - _store remains None
    - logger.debug called with exception info
    """
    mock_config_module = MagicMock()
    mock_store_module = MagicMock()
    
    # Make load_config raise an exception
    mock_config_module.load_config.side_effect = RuntimeError("Config load failed")
    
    monkeypatch.setitem(sys.modules, 'kindex.config', mock_config_module)
    monkeypatch.setitem(sys.modules, 'kindex.store', mock_store_module)
    
    result = kindex_integration._init_store()
    
    assert result is False, "_init_store() should return False on load_config failure"
    assert kindex_integration._store is None, "_store should remain None"
    
    # Verify debug logging occurred
    assert mock_logger.debug.called, "logger.debug should be called on failure"
    
    # Verify Store was not instantiated
    mock_store_module.Store.assert_not_called()


def test_init_store_initialization_failure_store_constructor(mock_logger, monkeypatch):
    """
    Error case: _init_store fails when Store() constructor raises exception.
    
    Verifies:
    - Returns False
    - _store remains None
    - logger.debug called with exception info
    """
    mock_config_module = MagicMock()
    mock_store_module = MagicMock()
    mock_config_obj = MagicMock()
    
    mock_config_module.load_config.return_value = mock_config_obj
    # Make Store constructor raise an exception
    mock_store_module.Store.side_effect = RuntimeError("Store init failed")
    
    monkeypatch.setitem(sys.modules, 'kindex.config', mock_config_module)
    monkeypatch.setitem(sys.modules, 'kindex.store', mock_store_module)
    
    result = kindex_integration._init_store()
    
    assert result is False, "_init_store() should return False on Store() failure"
    assert kindex_integration._store is None, "_store should remain None"
    
    # Verify debug logging occurred
    assert mock_logger.debug.called, "logger.debug should be called on failure"


def test_init_store_preserves_existing_store_on_failure(mock_logger):
    """
    Edge case: _init_store preserves existing _store value on failure.
    
    Verifies:
    - Returns False on failure
    - _store unchanged from pre-call value
    """
    # Set up existing store
    existing_store = MagicMock()
    kindex_integration._store = existing_store
    
    # Ensure kindex modules are not available
    if 'kindex.config' in sys.modules:
        del sys.modules['kindex.config']
    if 'kindex.store' in sys.modules:
        del sys.modules['kindex.store']
    
    result = kindex_integration._init_store()
    
    assert result is False, "_init_store() should return False on failure"
    assert kindex_integration._store == existing_store, \
        "_store should be unchanged from pre-call value"


# =============================================================================
# Suite 2: State & Availability
# =============================================================================

def test_is_available_first_call_success(mock_kindex_available, mock_logger):
    """
    Happy path: is_available on first call successfully initializes store.
    
    Verifies:
    - Returns True
    - _checked set to True
    - _store initialized
    """
    assert kindex_integration._checked is False, "Initial state: _checked should be False"
    assert kindex_integration._store is None, "Initial state: _store should be None"
    
    result = kindex_integration.is_available()
    
    assert result is True, "is_available() should return True on success"
    assert kindex_integration._checked is True, "_checked should be set to True"
    assert kindex_integration._store is not None, "_store should be initialized"
    assert kindex_integration._store == mock_kindex_available['store_instance'], \
        "_store should be the Store instance"


def test_is_available_first_call_failure(mock_logger):
    """
    Happy path: is_available on first call when kindex unavailable.
    
    Verifies:
    - Returns False
    - _checked set to True
    - _store remains None
    """
    # Ensure kindex modules are not available
    if 'kindex.config' in sys.modules:
        del sys.modules['kindex.config']
    if 'kindex.store' in sys.modules:
        del sys.modules['kindex.store']
    
    assert kindex_integration._checked is False, "Initial state: _checked should be False"
    assert kindex_integration._store is None, "Initial state: _store should be None"
    
    result = kindex_integration.is_available()
    
    assert result is False, "is_available() should return False on failure"
    assert kindex_integration._checked is True, "_checked should be set to True"
    assert kindex_integration._store is None, "_store should remain None"


def test_is_available_memoization_after_success(mock_kindex_available, mock_logger):
    """
    Invariant: is_available memoizes result after successful initialization.
    
    Verifies:
    - First call returns True and initializes
    - Second call returns True without re-attempting initialization
    - _init_store effectively called only once
    """
    # Patch _init_store to track calls
    with patch.object(kindex_integration, '_init_store', 
                     wraps=kindex_integration._init_store) as mock_init:
        # First call
        result1 = kindex_integration.is_available()
        assert result1 is True, "First call should return True"
        assert mock_init.call_count == 1, "_init_store should be called once"
        
        # Second call
        result2 = kindex_integration.is_available()
        assert result2 is True, "Second call should return True"
        assert mock_init.call_count == 1, "_init_store should not be called again"
        
        # Verify state
        assert kindex_integration._checked is True, "_checked should remain True"
        assert kindex_integration._store is not None, "_store should remain initialized"


def test_is_available_memoization_after_failure(mock_logger):
    """
    Invariant: is_available memoizes result after initialization failure.
    
    Verifies:
    - First call returns False
    - Second call returns False without re-attempting initialization
    - _init_store effectively called only once
    """
    # Ensure kindex modules are not available
    if 'kindex.config' in sys.modules:
        del sys.modules['kindex.config']
    if 'kindex.store' in sys.modules:
        del sys.modules['kindex.store']
    
    # Patch _init_store to track calls
    with patch.object(kindex_integration, '_init_store', 
                     wraps=kindex_integration._init_store) as mock_init:
        # First call
        result1 = kindex_integration.is_available()
        assert result1 is False, "First call should return False"
        assert mock_init.call_count == 1, "_init_store should be called once"
        
        # Second call
        result2 = kindex_integration.is_available()
        assert result2 is False, "Second call should return False"
        assert mock_init.call_count == 1, "_init_store should not be called again"
        
        # Verify state
        assert kindex_integration._checked is True, "_checked should remain True"
        assert kindex_integration._store is None, "_store should remain None"


def test_is_available_after_close_re_initializes(mock_kindex_available, mock_logger):
    """
    Edge case: is_available re-attempts initialization after close().
    
    Verifies:
    - is_available returns True before close
    - close() resets state
    - is_available returns True after close and re-initializes
    - _init_store called twice
    """
    # Patch _init_store to track calls
    with patch.object(kindex_integration, '_init_store', 
                     wraps=kindex_integration._init_store) as mock_init:
        # First availability check
        result1 = kindex_integration.is_available()
        assert result1 is True, "First is_available() should return True"
        assert mock_init.call_count == 1, "_init_store should be called once"
        assert kindex_integration._checked is True, "_checked should be True"
        
        # Close
        kindex_integration.close()
        assert kindex_integration._checked is False, "_checked should be reset to False"
        assert kindex_integration._store is None, "_store should be reset to None"
        
        # Second availability check after close
        result2 = kindex_integration.is_available()
        assert result2 is True, "Second is_available() should return True"
        assert mock_init.call_count == 2, "_init_store should be called twice"
        assert kindex_integration._checked is True, "_checked should be True again"
        assert kindex_integration._store is not None, "_store should be re-initialized"


def test_is_available_checked_flag_mutation(mock_kindex_available, mock_logger):
    """
    Invariant: _checked always set to True after first is_available() call.
    
    Verifies _checked is True regardless of success/failure outcome.
    """
    # Test success case
    assert kindex_integration._checked is False, "Initial: _checked should be False"
    kindex_integration.is_available()
    assert kindex_integration._checked is True, \
        "_checked should be True after is_available() (success)"
    
    # Reset and test failure case
    kindex_integration._checked = False
    kindex_integration._store = None
    
    # Remove kindex modules to force failure
    if 'kindex.config' in sys.modules:
        del sys.modules['kindex.config']
    if 'kindex.store' in sys.modules:
        del sys.modules['kindex.store']
    
    assert kindex_integration._checked is False, "After reset: _checked should be False"
    kindex_integration.is_available()
    assert kindex_integration._checked is True, \
        "_checked should be True after is_available() (failure)"


# =============================================================================
# Suite 3: Cleanup & Idempotency
# =============================================================================

def test_close_on_initialized_store(mock_kindex_available, mock_logger):
    """
    Happy path: close() properly closes initialized store and resets state.
    
    Verifies:
    - _store.close() is called
    - _store set to None
    - _checked set to False
    """
    # Initialize the store
    kindex_integration.is_available()
    assert kindex_integration._store is not None, "Store should be initialized"
    assert kindex_integration._checked is True, "_checked should be True"
    
    store_instance = kindex_integration._store
    
    # Close
    kindex_integration.close()
    
    # Verify close was called on the store
    store_instance.close.assert_called_once()
    
    # Verify state reset
    assert kindex_integration._store is None, "_store should be None after close()"
    assert kindex_integration._checked is False, "_checked should be False after close()"


def test_close_idempotent(mock_kindex_available, mock_logger):
    """
    Edge case: close() can be called multiple times safely.
    
    Verifies:
    - First close() succeeds
    - Second close() succeeds without error
    - _store.close() called only once
    - State remains reset
    """
    # Initialize the store
    kindex_integration.is_available()
    store_instance = kindex_integration._store
    
    # First close
    kindex_integration.close()
    store_instance.close.assert_called_once()
    assert kindex_integration._store is None, "_store should be None after first close()"
    assert kindex_integration._checked is False, "_checked should be False after first close()"
    
    # Second close (should not raise error)
    kindex_integration.close()
    
    # Verify close was still only called once (on the original instance)
    store_instance.close.assert_called_once()
    
    # Verify state remains reset
    assert kindex_integration._store is None, "_store should remain None"
    assert kindex_integration._checked is False, "_checked should remain False"


def test_close_on_uninitialized(mock_logger):
    """
    Edge case: close() on uninitialized module state.
    
    Verifies:
    - No errors occur
    - State remains None/False
    """
    # Ensure uninitialized state
    assert kindex_integration._store is None, "_store should be None initially"
    assert kindex_integration._checked is False, "_checked should be False initially"
    
    # Close should not raise any errors
    kindex_integration.close()
    
    # Verify state remains uninitialized
    assert kindex_integration._store is None, "_store should remain None"
    assert kindex_integration._checked is False, "_checked should remain False"
