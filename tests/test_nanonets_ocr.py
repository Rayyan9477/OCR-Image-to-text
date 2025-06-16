import os
import pytest
from PIL import Image
import numpy as np
from src.ocr.nanonets_ocr import NanonetsOCR
from src.ocr.preprocessor import ImagePreprocessor

@pytest.fixture
def ocr_engine():
    """Create OCR engine fixture"""
    return NanonetsOCR()

@pytest.fixture
def preprocessor():
    """Create preprocessor fixture"""
    return ImagePreprocessor()

@pytest.fixture
def test_image():
    """Create test image fixture"""
    # Create a simple test image with text
    img = Image.new('RGB', (100, 100), color='white')
    return img

@pytest.fixture
def test_image_with_text():
    """Create test image with text fixture"""
    # Create a test image with some text
    img = Image.new('RGB', (200, 100), color='white')
    return img

def test_ocr_initialization(ocr_engine):
    """Test OCR engine initialization"""
    assert ocr_engine is not None
    assert ocr_engine.processor is not None
    assert ocr_engine.model is not None
    assert ocr_engine.device in ['cuda', 'cpu']

def test_preprocessor_initialization(preprocessor):
    """Test preprocessor initialization"""
    assert preprocessor is not None

def test_image_enhancement(preprocessor, test_image):
    """Test image enhancement"""
    enhanced = preprocessor.enhance_image(test_image)
    assert enhanced is not None
    assert isinstance(enhanced, Image.Image)
    assert enhanced.mode == 'RGB'

def test_image_denoising(preprocessor, test_image):
    """Test image denoising"""
    denoised = preprocessor.denoise_image(test_image)
    assert denoised is not None
    assert isinstance(denoised, Image.Image)
    assert denoised.mode == 'RGB'

def test_skew_correction(preprocessor, test_image):
    """Test skew correction"""
    corrected = preprocessor.correct_skew(test_image)
    assert corrected is not None
    assert isinstance(corrected, Image.Image)
    assert corrected.mode == 'RGB'

def test_full_preprocessing(preprocessor, test_image):
    """Test full preprocessing pipeline"""
    processed = preprocessor.preprocess(
        test_image,
        enhance=True,
        denoise=True,
        correct_skew=True
    )
    assert processed is not None
    assert isinstance(processed, Image.Image)
    assert processed.mode == 'RGB'

def test_preprocessing_options(preprocessor, test_image):
    """Test preprocessing with different options"""
    # Test with all options disabled
    processed = preprocessor.preprocess(
        test_image,
        enhance=False,
        denoise=False,
        correct_skew=False
    )
    assert processed is not None
    assert isinstance(processed, Image.Image)
    
    # Test with only enhancement
    processed = preprocessor.preprocess(
        test_image,
        enhance=True,
        denoise=False,
        correct_skew=False
    )
    assert processed is not None
    assert isinstance(processed, Image.Image)

def test_ocr_extraction(ocr_engine, test_image):
    """Test OCR text extraction"""
    result = ocr_engine.extract_text(test_image)
    assert isinstance(result, dict)
    assert 'text' in result
    assert 'confidence' in result
    assert isinstance(result['text'], str)
    assert isinstance(result['confidence'], float)

def test_ocr_with_preserve_layout(ocr_engine, test_image):
    """Test OCR with layout preservation"""
    result = ocr_engine.extract_text(test_image, preserve_layout=True)
    assert isinstance(result, dict)
    assert 'text' in result
    assert 'layout' in result
    assert result['layout'] is None or result['layout'] == "Not implemented"

def test_batch_processing(ocr_engine):
    """Test batch processing"""
    # Create multiple test images
    images = [
        Image.new('RGB', (100, 100), color='white')
        for _ in range(3)
    ]
    
    results = ocr_engine.batch_process(images)
    assert isinstance(results, list)
    assert len(results) == len(images)
    for result in results:
        assert isinstance(result, dict)
        assert 'text' in result
        assert 'confidence' in result
        assert isinstance(result['text'], str)
        assert isinstance(result['confidence'], float)

def test_error_handling(ocr_engine):
    """Test error handling"""
    # Test with invalid image
    result = ocr_engine.extract_text(None)
    assert isinstance(result, dict)
    assert 'error' in result
    assert result['text'] == ''
    assert result['confidence'] == 0.0
    
    # Test with invalid image type
    result = ocr_engine.extract_text("not an image")
    assert isinstance(result, dict)
    assert 'error' in result
    assert result['text'] == ''
    assert result['confidence'] == 0.0

def test_preprocessing_error_handling(preprocessor):
    """Test preprocessing error handling"""
    # Test with invalid image
    result = preprocessor.preprocess(None)
    assert result is None
    
    # Test with invalid image type
    with pytest.raises(Exception):
        preprocessor.preprocess("not an image")
    
    # Test with invalid numpy array
    with pytest.raises(Exception):
        preprocessor.preprocess(np.array([1, 2, 3]))

def test_image_conversion(preprocessor):
    """Test image conversion between formats"""
    # Test numpy array to PIL Image
    np_image = np.zeros((100, 100, 3), dtype=np.uint8)
    pil_image = preprocessor.preprocess(np_image)
    assert isinstance(pil_image, Image.Image)
    
    # Test PIL Image to numpy array and back
    pil_image = Image.new('RGB', (100, 100), color='white')
    np_image = np.array(pil_image)
    result = preprocessor.preprocess(np_image)
    assert isinstance(result, Image.Image) 