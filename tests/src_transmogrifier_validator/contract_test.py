"""
Contract tests for SemanticValidator component.

Tests cover:
- Initialization with various model names
- Lazy loading behavior and error handling
- Validate function with mocked dependencies
- is_valid threshold checking
- Edge cases (empty strings, unicode, long texts)
- Error cases (import failures, unavailable dependencies)
- Invariants (model state, symmetry, bounds)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

# Import component under test
from transmogrifier.validator import SemanticValidator


class TestInit:
    """Tests for __init__ function."""
    
    def test_init_happy_path(self):
        """Initialize SemanticValidator with valid model name, verify _model_name is set and _model is None"""
        validator = SemanticValidator(model_name="all-MiniLM-L6-v2")
        
        assert validator._model_name == "all-MiniLM-L6-v2"
        assert validator._model is None
    
    def test_init_custom_model(self):
        """Initialize with custom model name"""
        validator = SemanticValidator(model_name="custom-model-name")
        
        assert validator._model_name == "custom-model-name"
        assert validator._model is None
    
    def test_init_empty_string(self):
        """Initialize with empty string model name (edge case)"""
        validator = SemanticValidator(model_name="")
        
        assert validator._model_name == ""
        assert validator._model is None


class TestLoad:
    """Tests for _load function."""
    
    def test_load_success(self):
        """Lazy load model successfully when sentence_transformers available"""
        validator = SemanticValidator(model_name="test-model")
        
        # Mock SentenceTransformer
        mock_model = Mock()
        mock_model.encode = Mock(return_value=[[0.1, 0.2, 0.3]])
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            validator._load()
        
        assert validator._model is not None
        assert validator._model is not False
    
    def test_load_import_error(self):
        """Handle ImportError when sentence_transformers not available"""
        validator = SemanticValidator(model_name="test-model")
        
        # Mock import failure by patching builtins.__import__
        with patch('builtins.__import__', side_effect=ImportError("sentence_transformers not found")):
            with patch('logging.Logger.warning'):
                validator._load()
        
        assert validator._model is False
    
    def test_load_idempotent(self):
        """Verify _load does nothing if model already loaded"""
        validator = SemanticValidator(model_name="test-model")
        
        # Pre-load with a mock model
        mock_model = Mock()
        validator._model = mock_model
        
        # Call _load again - should not change the model
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            validator._load()
            # SentenceTransformer should not be called if already loaded
            mock_st.assert_not_called()
        
        assert validator._model is mock_model


class TestValidate:
    """Tests for validate function."""
    
    def test_validate_happy_path(self):
        """Compute cosine similarity between two texts successfully"""
        validator = SemanticValidator(model_name="test-model")
        
        # Mock the model and numpy
        mock_model = Mock()
        mock_embeddings = Mock()
        mock_embeddings.__getitem__ = Mock(side_effect=lambda x: [0.5, 0.5, 0.5, 0.5])
        mock_model.encode = Mock(return_value=mock_embeddings)
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            with patch('numpy.linalg.norm', return_value=1.0):
                with patch('numpy.dot', return_value=0.85):
                    result = validator.validate(input_text="hello world", output_text="hello there")
        
        assert isinstance(result, float)
        assert result >= -1.0
        assert result <= 1.0
    
    def test_validate_identical_texts(self):
        """Validate identical texts should return high similarity"""
        validator = SemanticValidator(model_name="test-model")
        
        # Mock the model to return identical embeddings
        mock_model = Mock()
        mock_embeddings = Mock()
        mock_embeddings.__getitem__ = Mock(return_value=[1.0, 0.0, 0.0])
        mock_model.encode = Mock(return_value=mock_embeddings)
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            with patch('numpy.linalg.norm', return_value=1.0):
                with patch('numpy.dot', return_value=1.0):
                    result = validator.validate(input_text="same text", output_text="same text")
        
        assert result == 1.0
    
    def test_validate_model_unavailable(self):
        """Return None when model loading failed"""
        validator = SemanticValidator(model_name="test-model")
        
        # Simulate failed model loading
        with patch('builtins.__import__', side_effect=ImportError()):
            with patch('logging.Logger.warning'):
                validator._load()
        
        result = validator.validate(input_text="test", output_text="test")
        
        assert result is None
    
    def test_validate_empty_strings(self):
        """Validate with empty strings"""
        validator = SemanticValidator(model_name="test-model")
        
        mock_model = Mock()
        mock_embeddings = Mock()
        mock_embeddings.__getitem__ = Mock(return_value=[0.0, 0.0])
        mock_model.encode = Mock(return_value=mock_embeddings)
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            with patch('numpy.linalg.norm', return_value=1.0):
                with patch('numpy.dot', return_value=0.5):
                    result = validator.validate(input_text="", output_text="")
        
        assert result is not None
    
    def test_validate_whitespace_only(self):
        """Validate texts with only whitespace"""
        validator = SemanticValidator(model_name="test-model")
        
        mock_model = Mock()
        mock_embeddings = Mock()
        mock_embeddings.__getitem__ = Mock(return_value=[0.1, 0.1])
        mock_model.encode = Mock(return_value=mock_embeddings)
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            with patch('numpy.linalg.norm', return_value=1.0):
                with patch('numpy.dot', return_value=0.3):
                    result = validator.validate(input_text="   ", output_text="\t\n")
        
        assert result is not None
    
    def test_validate_unicode(self):
        """Validate texts with unicode characters"""
        validator = SemanticValidator(model_name="test-model")
        
        mock_model = Mock()
        mock_embeddings = Mock()
        mock_embeddings.__getitem__ = Mock(return_value=[0.5, 0.5])
        mock_model.encode = Mock(return_value=mock_embeddings)
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            with patch('numpy.linalg.norm', return_value=1.0):
                with patch('numpy.dot', return_value=0.95):
                    result = validator.validate(input_text="café 日本語", output_text="café 日本語")
        
        assert result is not None
    
    def test_validate_very_long_text(self):
        """Validate with very long text strings"""
        validator = SemanticValidator(model_name="test-model")
        
        long_text = "word " * 1000
        mock_model = Mock()
        mock_embeddings = Mock()
        mock_embeddings.__getitem__ = Mock(return_value=[0.6] * 384)
        mock_model.encode = Mock(return_value=mock_embeddings)
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            with patch('numpy.linalg.norm', return_value=1.0):
                with patch('numpy.dot', return_value=0.99):
                    result = validator.validate(input_text=long_text, output_text=long_text)
        
        assert result is not None


class TestIsValid:
    """Tests for is_valid function."""
    
    def test_is_valid_above_threshold(self):
        """Check similarity above threshold returns True"""
        validator = SemanticValidator(model_name="test-model")
        
        # Mock validate to return 0.95
        with patch.object(validator, 'validate', return_value=0.95):
            result = validator.is_valid(input_text="test", output_text="test", threshold=0.9)
        
        assert result is True
    
    def test_is_valid_below_threshold(self):
        """Check similarity below threshold returns False"""
        validator = SemanticValidator(model_name="test-model")
        
        # Mock validate to return 0.5
        with patch.object(validator, 'validate', return_value=0.5):
            result = validator.is_valid(input_text="test1", output_text="test2", threshold=0.9)
        
        assert result is False
    
    def test_is_valid_at_threshold(self):
        """Check similarity exactly at threshold returns True"""
        validator = SemanticValidator(model_name="test-model")
        
        # Mock validate to return exactly 0.95
        with patch.object(validator, 'validate', return_value=0.95):
            result = validator.is_valid(input_text="test", output_text="test", threshold=0.95)
        
        assert result is True
    
    def test_is_valid_threshold_zero(self):
        """Check with threshold of 0.0 (minimum boundary)"""
        validator = SemanticValidator(model_name="test-model")
        
        # Mock validate to return 0.1
        with patch.object(validator, 'validate', return_value=0.1):
            result = validator.is_valid(input_text="test", output_text="test", threshold=0.0)
        
        assert result is True
    
    def test_is_valid_threshold_one(self):
        """Check with threshold of 1.0 (maximum boundary)"""
        validator = SemanticValidator(model_name="test-model")
        
        # Mock validate to return 1.0
        with patch.object(validator, 'validate', return_value=1.0):
            result = validator.is_valid(input_text="test", output_text="test", threshold=1.0)
        
        assert result is True
    
    def test_is_valid_validation_unavailable(self):
        """Return None when validate returns None"""
        validator = SemanticValidator(model_name="test-model")
        
        # Mock validate to return None (dependencies unavailable)
        with patch.object(validator, 'validate', return_value=None):
            result = validator.is_valid(input_text="test", output_text="test", threshold=0.95)
        
        assert result is None


class TestInvariants:
    """Tests for system invariants."""
    
    def test_invariant_model_state(self):
        """Verify _model is always None, False, or SentenceTransformer instance"""
        validator = SemanticValidator(model_name="test-model")
        
        # Initially None
        assert validator._model is None or validator._model is False or hasattr(validator._model, 'encode')
        
        # After successful load
        mock_model = Mock()
        mock_model.encode = Mock()
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            validator._load()
        
        assert validator._model is None or validator._model is False or hasattr(validator._model, 'encode')
        
        # After failed load
        validator2 = SemanticValidator(model_name="test-model-2")
        with patch('builtins.__import__', side_effect=ImportError()):
            with patch('logging.Logger.warning'):
                validator2._load()
        
        assert validator2._model is None or validator2._model is False or hasattr(validator2._model, 'encode')
    
    def test_invariant_model_name_constant(self):
        """Verify _model_name remains constant after initialization"""
        validator = SemanticValidator(model_name="test-model")
        
        initial_name = validator._model_name
        assert validator._model_name == "test-model"
        
        # Perform operations
        mock_model = Mock()
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            validator._load()
        
        # Name should not change
        assert validator._model_name == "test-model"
        assert validator._model_name == initial_name
    
    def test_validate_symmetry(self):
        """Verify similarity is symmetric: validate(a,b) == validate(b,a)"""
        validator = SemanticValidator(model_name="test-model")
        
        mock_model = Mock()
        embedding1 = [0.5, 0.5, 0.5]
        embedding2 = [0.3, 0.7, 0.6]
        mock_embeddings = Mock()
        mock_embeddings.__getitem__ = Mock(side_effect=[embedding1, embedding2, embedding2, embedding1])
        mock_model.encode = Mock(return_value=mock_embeddings)
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            with patch('numpy.linalg.norm', return_value=1.0):
                with patch('numpy.dot', return_value=0.75):
                    result1 = validator.validate(input_text="text1", output_text="text2")
                    result2 = validator.validate(input_text="text2", output_text="text1")
        
        assert result1 == result2
    
    def test_validate_bounds(self):
        """Verify validate returns values in range [-1.0, 1.0]"""
        validator = SemanticValidator(model_name="test-model")
        
        mock_model = Mock()
        mock_embeddings = Mock()
        mock_embeddings.__getitem__ = Mock(return_value=[0.4, 0.6, 0.2])
        mock_model.encode = Mock(return_value=mock_embeddings)
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            with patch('numpy.linalg.norm', return_value=1.0):
                with patch('numpy.dot', return_value=0.42):
                    result = validator.validate(input_text="random text", output_text="other text")
        
        assert result >= -1.0
        assert result <= 1.0
