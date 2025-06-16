import pytest
from PIL import Image
from src.core.model_manager import ModelManager
from src.core.ocr_engine import OCREngine
from unittest.mock import patch

@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    img = Image.new('RGB', (200, 100), color='white')
    return img

@pytest.fixture
def model_manager():
    """Create a model manager with mocked dependencies"""
    with patch('src.core.model_manager._check_available_modules') as mock_check:
        mock_check.return_value = {
            'paddleocr_available': False,
            'easyocr_available': False,
            'tesseract_available': False
        }
        return ModelManager()

@pytest.fixture
def ocr_engine():
    """Create an OCR engine with mocked dependencies"""
    with patch('src.core.ocr_engine.OCREngine._check_available_engines') as mock_check:
        mock_check.return_value = {
            'paddle': False,
            'easy': False,
            'tesseract': False,
            'combined': True
        }
        return OCREngine() 