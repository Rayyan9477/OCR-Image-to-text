import pytest
from src.core.model_manager import ModelManager
from unittest.mock import patch, MagicMock

@pytest.fixture
def model_manager():
    with patch('src.core.model_manager._check_available_modules') as mock_check:
        mock_check.return_value = {
            'paddleocr_available': False,
            'easyocr_available': False,
            'tesseract_available': False
        }
        return ModelManager()

def test_model_manager_initialization(model_manager):
    """Test model manager initialization"""
    assert model_manager is not None
    assert hasattr(model_manager, 'OPTIONAL_MODULES')
    assert isinstance(model_manager.OPTIONAL_MODULES, dict)
    assert 'paddleocr_available' in model_manager.OPTIONAL_MODULES
    assert 'easyocr_available' in model_manager.OPTIONAL_MODULES
    assert 'tesseract_available' in model_manager.OPTIONAL_MODULES

def test_get_available_engines(model_manager):
    """Test getting available OCR engines"""
    engines = model_manager.get_available_engines()
    assert isinstance(engines, dict)
    assert 'PaddleOCR' in engines
    assert 'EasyOCR' in engines
    assert 'Tesseract' in engines
    assert all(isinstance(v, bool) for v in engines.values())

def test_get_recommended_engine(model_manager):
    """Test getting recommended OCR engine"""
    engine = model_manager.get_recommended_engine()
    assert isinstance(engine, str)
    assert engine in ['paddle', 'easy', 'tesseract', 'none']

def test_get_engine_description(model_manager):
    """Test getting engine descriptions"""
    # Test known engines
    assert model_manager.get_engine_description('paddle')
    assert model_manager.get_engine_description('easy')
    assert model_manager.get_engine_description('tesseract')
    assert model_manager.get_engine_description('combined')
    
    # Test unknown engine
    assert model_manager.get_engine_description('unknown') == "Unknown engine"

@pytest.mark.skipif(not ModelManager().OPTIONAL_MODULES.get('paddleocr_available', False),
                   reason="PaddleOCR not available")
def test_paddleocr_available():
    """Test PaddleOCR availability"""
    manager = ModelManager()
    assert manager.OPTIONAL_MODULES['paddleocr_available'] == True

@pytest.mark.skipif(not ModelManager().OPTIONAL_MODULES.get('easyocr_available', False),
                   reason="EasyOCR not available")
def test_easyocr_available():
    """Test EasyOCR availability"""
    manager = ModelManager()
    assert manager.OPTIONAL_MODULES['easyocr_available'] == True 