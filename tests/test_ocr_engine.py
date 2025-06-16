import pytest
from PIL import Image
import numpy as np
from src.core.ocr_engine import OCREngine
from unittest.mock import patch, MagicMock

@pytest.fixture
def ocr_engine():
    return OCREngine()

@pytest.fixture
def sample_image():
    # Create a simple test image with text
    img = Image.new('RGB', (200, 100), color='white')
    return img

def test_ocr_engine_initialization(ocr_engine):
    """Test OCR engine initialization"""
    assert ocr_engine is not None
    assert hasattr(ocr_engine, 'available_engines')
    assert isinstance(ocr_engine.available_engines, dict)

@pytest.mark.skipif(not OCREngine().available_engines.get('paddle', False),
                   reason="PaddleOCR not available")
def test_perform_ocr_paddle(ocr_engine, sample_image):
    """Test OCR with PaddleOCR"""
    with patch('paddleocr.PaddleOCR') as mock_paddle:
        mock_paddle.return_value.ocr.return_value = [[
            [[[0, 0], [100, 0], [100, 20], [0, 20]], ('text1', 0.9)],
            [[[0, 30], [100, 30], [100, 50], [0, 50]], ('text2', 0.8)]
        ]]
        result = ocr_engine.perform_ocr(sample_image, engine='paddle')
        assert isinstance(result, str)
        assert 'text1' in result
        assert 'text2' in result

@pytest.mark.skipif(not OCREngine().available_engines.get('easy', False),
                   reason="EasyOCR not available")
def test_perform_ocr_easy(ocr_engine, sample_image):
    """Test OCR with EasyOCR"""
    with patch('easyocr.Reader') as mock_easy:
        mock_easy.return_value.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], 'text1', 0.9),
            ([[0, 30], [100, 30], [100, 50], [0, 50]], 'text2', 0.8)
        ]
        result = ocr_engine.perform_ocr(sample_image, engine='easy')
        assert isinstance(result, str)
        assert 'text1' in result
        assert 'text2' in result

def test_perform_ocr_invalid_engine(ocr_engine, sample_image):
    """Test OCR with invalid engine"""
    result = ocr_engine.perform_ocr(sample_image, engine='invalid_engine')
    assert result.startswith("Error: Unsupported OCR engine")

def test_perform_ocr_preserve_layout(ocr_engine, sample_image):
    """Test OCR with layout preservation"""
    with patch.object(ocr_engine, '_paddle_ocr') as mock_paddle:
        mock_paddle.return_value = "Test text with layout"
        result = ocr_engine.perform_ocr(sample_image, preserve_layout=True)
        assert isinstance(result, str)

def test_perform_ocr_no_layout(ocr_engine, sample_image):
    """Test OCR without layout preservation"""
    with patch.object(ocr_engine, '_paddle_ocr') as mock_paddle:
        mock_paddle.return_value = "Test text without layout"
        result = ocr_engine.perform_ocr(sample_image, preserve_layout=False)
        assert isinstance(result, str)

def test_combined_ocr(ocr_engine, sample_image):
    """Test combined OCR functionality"""
    with patch.object(ocr_engine, '_paddle_ocr') as mock_paddle, \
         patch.object(ocr_engine, '_easy_ocr') as mock_easy:
        mock_paddle.return_value = "Paddle text"
        mock_easy.return_value = "Easy text"
        result = ocr_engine.perform_ocr(sample_image, engine='combined')
        assert isinstance(result, str)
        assert "Paddle text" in result
        assert "Easy text" in result

def test_format_paddle_result(ocr_engine):
    """Test PaddleOCR result formatting"""
    mock_result = [[
        [[[0, 0], [100, 0], [100, 20], [0, 20]], ('text1', 0.9)],
        [[[0, 30], [100, 30], [100, 50], [0, 50]], ('text2', 0.8)]
    ]]
    formatted = ocr_engine._format_paddle_result(mock_result)
    assert isinstance(formatted, str)
    assert 'text1' in formatted
    assert 'text2' in formatted

def test_format_easy_result(ocr_engine):
    """Test EasyOCR result formatting"""
    mock_result = [
        ([[0, 0], [100, 0], [100, 20], [0, 20]], 'text1', 0.9),
        ([[0, 30], [100, 30], [100, 50], [0, 50]], 'text2', 0.8)
    ]
    formatted = ocr_engine._format_easy_result(mock_result)
    assert isinstance(formatted, str)
    assert 'text1' in formatted
    assert 'text2' in formatted 