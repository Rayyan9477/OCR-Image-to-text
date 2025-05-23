#!/usr/bin/env python3
"""
Simple OCR Module - A lightweight alternative OCR module with better error handling
"""

import os
import sys
import time
import importlib.util
import logging
import warnings
import numpy as np
from PIL import Image
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize module availability
OPTIONAL_MODULES = {
    'paddleocr_available': False,
    'easyocr_available': False,
    'tesseract_available': False
}

# Thread-safe model cache
_model_lock = Lock()
_model_cache = {}

def check_dependencies():
    """Check which OCR engines are available"""
    # Check for PaddleOCR
    try:
        if importlib.util.find_spec('paddleocr'):
            OPTIONAL_MODULES['paddleocr_available'] = True
            logger.info("PaddleOCR module is available")
    except Exception as e:
        logger.warning(f"PaddleOCR check error: {e}")
    
    # Check for EasyOCR
    try:
        if importlib.util.find_spec('easyocr'):
            OPTIONAL_MODULES['easyocr_available'] = True
            logger.info("EasyOCR module is available")
    except Exception as e:
        logger.warning(f"EasyOCR check error: {e}")
    
    # Check for Tesseract
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        OPTIONAL_MODULES['tesseract_available'] = True
        logger.info("Tesseract is available")
    except Exception as e:
        logger.warning(f"Tesseract check error: {e}")
    
    return OPTIONAL_MODULES

def get_paddle_ocr():
    """Initialize PaddleOCR instance with error handling"""
    if not OPTIONAL_MODULES['paddleocr_available']:
        return None
    
    try:
        if 'paddle_ocr' not in _model_cache:
            from paddleocr import PaddleOCR
            logger.info("Initializing PaddleOCR model")
            with _model_lock:
                _model_cache['paddle_ocr'] = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
        return _model_cache['paddle_ocr']
    except Exception as e:
        logger.error(f"Error initializing PaddleOCR: {e}")
        return None

def get_easy_ocr():
    """Initialize EasyOCR instance with error handling"""
    if not OPTIONAL_MODULES['easyocr_available']:
        return None
    
    try:
        if 'easy_ocr' not in _model_cache:
            import easyocr
            logger.info("Initializing EasyOCR model")
            with _model_lock:
                _model_cache['easy_ocr'] = easyocr.Reader(['en'], gpu=False)
        
        return _model_cache['easy_ocr']
    except Exception as e:
        logger.error(f"Error initializing EasyOCR: {e}")
        return None

def paddle_ocr(image, preserve_layout=True):
    """Perform OCR using PaddleOCR with robust error handling"""
    if not OPTIONAL_MODULES['paddleocr_available']:
        logger.warning("PaddleOCR not available")
        return ""
    
    try:
        # Initialize PaddleOCR directly
        from paddleocr import PaddleOCR
        
        # Try getting from cache first
        paddle = get_paddle_ocr()
        
        # If that fails, initialize directly
        if paddle is None:
            logger.info("Initializing PaddleOCR directly")
            try:
                paddle = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            except Exception as e:
                logger.error(f"Error initializing PaddleOCR directly: {e}")
                return ""
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            # Convert to RGB if it's not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_np = np.array(image)
        else:
            img_np = image
        
        # Perform OCR
        result = paddle.ocr(img_np, cls=True)
        
        if not result or not result[0]:
            return ""
        
        # Format output
        if not preserve_layout:
            # Simple concatenation
            return " ".join([line[1][0] for line in result[0] if line[1][0]])
        else:
            # Sort by vertical position
            sorted_results = sorted(result[0], key=lambda x: (x[0][0][1] + x[0][3][1]) / 2)
            
            lines = []
            line_y = 0
            current_line = []
            
            # Group by Y position
            for box in sorted_results:
                y_mid = (box[0][0][1] + box[0][3][1]) / 2
                
                if not current_line or abs(y_mid - line_y) < 20:
                    current_line.append(box)
                    line_y = (line_y + y_mid) / 2
                else:
                    # Sort line by X position
                    current_line.sort(key=lambda x: x[0][0][0])
                    lines.append(current_line)
                    current_line = [box]
                    line_y = y_mid
            
            # Add last line
            if current_line:
                current_line.sort(key=lambda x: x[0][0][0])
                lines.append(current_line)
            
            # Format text with newlines between lines and spaces between words
            return "\n".join([" ".join([text[1][0] for text in line]) for line in lines])
    except Exception as e:
        logger.error(f"Error in paddle_ocr: {e}")
        return ""

def easyocr_ocr(image, preserve_layout=True):
    """Perform OCR using EasyOCR with robust error handling"""
    if not OPTIONAL_MODULES['easyocr_available']:
        logger.warning("EasyOCR not available")
        return ""
    
    try:
        # Initialize EasyOCR directly
        import easyocr
        
        # Try getting from cache first
        reader = get_easy_ocr()
        
        # If that fails, initialize directly
        if reader is None:
            logger.info("Initializing EasyOCR directly")
            try:
                reader = easyocr.Reader(['en'], gpu=False)
            except Exception as e:
                logger.error(f"Error initializing EasyOCR directly: {e}")
                return ""
        
        # Convert to numpy array if it's a PIL Image
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
        
        # Perform OCR
        result = reader.readtext(img_np)
        
        if not result:
            return ""
        
        # Format output
        if not preserve_layout:
            # Simple concatenation
            return " ".join([text[1] for text in result])
        else:
            # Sort by vertical position
            sorted_results = sorted(result, key=lambda x: (x[0][0][1] + x[0][2][1]) / 2)
            
            lines = []
            line_y = 0
            current_line = []
            
            # Group by Y position
            for box, text, conf in sorted_results:
                y_mid = (box[0][1] + box[2][1]) / 2
                
                if not current_line or abs(y_mid - line_y) < 20:
                    current_line.append((box, text, conf))
                    line_y = (line_y + y_mid) / 2
                else:
                    # Sort line by X position
                    current_line.sort(key=lambda x: x[0][0][0])
                    lines.append(current_line)
                    current_line = [(box, text, conf)]
                    line_y = y_mid
            
            # Add last line
            if current_line:
                current_line.sort(key=lambda x: x[0][0][0])
                lines.append(current_line)
            
            # Format text with newlines between lines and spaces between words
            return "\n".join([" ".join([word[1] for word in line]) for line in lines])
    except Exception as e:
        logger.error(f"Error in easyocr_ocr: {e}")
        return ""

def pytesseract_ocr(image, preserve_layout=True):
    """Perform OCR using Tesseract with robust error handling"""
    if not OPTIONAL_MODULES['tesseract_available']:
        logger.warning("Tesseract not available")
        return ""
    
    try:
        import pytesseract
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Perform OCR
        if preserve_layout:
            config = "--psm 6"  # Assume a single uniform block of text
        else:
            config = "--psm 3"  # Fully automatic page segmentation
        
        text = pytesseract.image_to_string(image, config=config)
        return text
    except Exception as e:
        logger.error(f"Error in pytesseract_ocr: {e}")
        return ""

def perform_ocr(image, engine="auto", preserve_layout=True):
    """
    Perform OCR on an image using the specified engine
    
    Args:
        image: PIL Image or numpy array
        engine: OCR engine to use ('paddle', 'easy', 'auto', or 'tesseract')
        preserve_layout: Whether to preserve text layout
        
    Returns:
        Extracted text, or error message if OCR fails
    """
    # Check dependencies first
    check_dependencies()
    
    # Ensure we have a valid image
    if image is None:
        return "Error: No image provided"
    
    # Determine which engine to use
    if engine == "auto" or engine == "combined":
        # Try to use the best available engine
        if OPTIONAL_MODULES['paddleocr_available']:
            result = paddle_ocr(image, preserve_layout)
            if result:
                return result
        
        if OPTIONAL_MODULES['easyocr_available']:
            result = easyocr_ocr(image, preserve_layout)
            if result:
                return result
        
        if OPTIONAL_MODULES['tesseract_available']:
            result = pytesseract_ocr(image, preserve_layout)
            if result:
                return result
        
        return "Error: No OCR engines available or all engines failed"
    
    elif engine == "paddle":
        if OPTIONAL_MODULES['paddleocr_available']:
            result = paddle_ocr(image, preserve_layout)
            return result if result else "PaddleOCR failed to extract text"
        else:
            return "Error: PaddleOCR is not installed"
    
    elif engine == "easy":
        if OPTIONAL_MODULES['easyocr_available']:
            result = easyocr_ocr(image, preserve_layout)
            return result if result else "EasyOCR failed to extract text"
        else:
            return "Error: EasyOCR is not installed"
    
    elif engine == "tesseract":
        if OPTIONAL_MODULES['tesseract_available']:
            result = pytesseract_ocr(image, preserve_layout)
            return result if result else "Tesseract OCR failed to extract text"
        else:
            return "Error: Tesseract is not installed"
    
    else:
        return f"Error: Unknown OCR engine '{engine}'"

if __name__ == "__main__":
    # Check which OCR engines are available
    available = check_dependencies()
    for engine, is_available in available.items():
        print(f"{engine}: {'✓' if is_available else '✗'}")
    
    print("\nTo test OCR, run: python test_ocr.py")
