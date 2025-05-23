#!/usr/bin/env python3
"""
OCR Manager - Integrates multiple OCR engines with robust fallback mechanisms
"""

import os
import time
import importlib
import logging
import warnings
import numpy as np
from PIL import Image
import cv2
import re
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Track available OCR engines
AVAILABLE_OCR_ENGINES = {
    'paddle': False,
    'easy': False,
    'tesseract': False
}

# Thread-safe model cache
_model_lock = Lock()
_model_cache = {}

def check_ocr_engines():
    """Check which OCR engines are available"""
    available_engines = []
    missing_engines = []
    installation_instructions = []
    
    # Check PaddleOCR
    paddle_spec = importlib.util.find_spec('paddleocr')
    if paddle_spec:
        try:
            from paddleocr import PaddleOCR
            AVAILABLE_OCR_ENGINES['paddle'] = True
            available_engines.append("PaddleOCR")
            logger.info("PaddleOCR is available")
        except Exception as e:
            AVAILABLE_OCR_ENGINES['paddle'] = False
            missing_engines.append("PaddleOCR")
            installation_instructions.append("pip install paddlepaddle paddleocr")
            logger.warning(f"PaddleOCR import failed: {e}")
    else:
        missing_engines.append("PaddleOCR")
        installation_instructions.append("pip install paddlepaddle paddleocr")
    
    # Check EasyOCR
    easy_spec = importlib.util.find_spec('easyocr')
    if easy_spec:
        try:
            import easyocr
            AVAILABLE_OCR_ENGINES['easy'] = True
            available_engines.append("EasyOCR")
            logger.info("EasyOCR is available")
        except Exception as e:
            AVAILABLE_OCR_ENGINES['easy'] = False
            missing_engines.append("EasyOCR")
            installation_instructions.append("pip install easyocr")
            logger.warning(f"EasyOCR import failed: {e}")
    else:
        missing_engines.append("EasyOCR")
        installation_instructions.append("pip install easyocr")
    
    # Check Tesseract
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        AVAILABLE_OCR_ENGINES['tesseract'] = True
        available_engines.append("Tesseract")
        logger.info("Tesseract is available")
    except Exception as e:
        AVAILABLE_OCR_ENGINES['tesseract'] = False
        missing_engines.append("Tesseract")
        if os.name == 'nt':  # Windows
            installation_instructions.append(
                "1. Download Tesseract installer from https://github.com/UB-Mannheim/tesseract/wiki\n"
                "2. Install and add to PATH\n"
                "3. pip install pytesseract"
            )
        elif os.name == 'posix':  # Linux/macOS
            if os.path.exists('/usr/bin/apt'):  # Debian/Ubuntu
                installation_instructions.append("sudo apt-get install tesseract-ocr && pip install pytesseract")
            elif os.path.exists('/usr/bin/brew'):  # macOS with Homebrew
                installation_instructions.append("brew install tesseract && pip install pytesseract")
            else:
                installation_instructions.append(
                    "Install tesseract-ocr using your package manager and then: pip install pytesseract"
                )
        logger.warning(f"Tesseract not available: {e}")
    
    return available_engines, missing_engines, installation_instructions

def get_paddle_ocr():
    """Initialize PaddleOCR with robust error handling"""
    if not AVAILABLE_OCR_ENGINES['paddle']:
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
        AVAILABLE_OCR_ENGINES['paddle'] = False
        return None

def get_easy_ocr():
    """Initialize EasyOCR with robust error handling"""
    if not AVAILABLE_OCR_ENGINES['easy']:
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
        AVAILABLE_OCR_ENGINES['easy'] = False
        return None

def preprocess_image(image, enhance=True, denoise=True, adaptive_threshold=True):
    """
    Apply various preprocessing techniques to improve OCR quality
    """
    try:
        # Convert PIL Image to OpenCV format
        if isinstance(image, Image.Image):
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            image_cv = image
            
        # Apply grayscale
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply image enhancement if requested
        if enhance:
            # Convert back to PIL Image for enhancement
            pil_img = Image.fromarray(gray)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(1.5)
            
            # Convert back to OpenCV
            gray = np.array(pil_img)
        
        # Apply adaptive thresholding if requested
        if adaptive_threshold:
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            # Simple thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply denoising if requested
        if denoise:
            processed = cv2.medianBlur(thresh, 3)
        else:
            processed = thresh
        
        # Convert back to PIL Image for compatibility
        result = Image.fromarray(processed)
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        # Return original image if preprocessing fails
        if isinstance(image, Image.Image):
            result = image
        else:
            result = Image.fromarray(image_cv)
    
    return result

def detect_tables(image, min_lines=3):
    """Detect if the image likely contains tables based on line detection"""
    try:
        # Convert PIL Image to OpenCV format if needed
        if isinstance(image, Image.Image):
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_cv = image
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Count the number of detected lines
        h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        h_count = 0 if h_lines is None else len(h_lines)
        v_count = 0 if v_lines is None else len(v_lines)
        
        # If we have a significant number of both horizontal and vertical lines, it's likely a table
        return h_count >= min_lines and v_count >= min_lines
        
    except Exception as e:
        logger.error(f"Error detecting tables: {e}")
        return False

def paddle_ocr(image, preserve_layout=True):
    """Perform OCR using PaddleOCR with robust error handling"""
    start_time = time.time()
    
    try:
        # Check if PaddleOCR is available
        try:
            from paddleocr import PaddleOCR
            has_paddle = True
        except ImportError:
            logger.warning("PaddleOCR not available")
            return ""
            
        # Get PaddleOCR instance
        paddle = get_paddle_ocr()
        
        # If the model manager failed to provide a model, try direct initialization
        if paddle is None:
            try:
                logger.info("Initializing PaddleOCR directly...")
                paddle = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            except Exception as e:
                logger.error(f"Error initializing PaddleOCR directly: {e}")
                return ""
        
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
        
        # Perform OCR
        result = paddle.ocr(img_np, cls=True)
        
        if not result or not result[0]:
            return ""
        
        # Format the results
        if preserve_layout:
            # Sort by vertical position for layout preservation
            sorted_results = sorted(result[0], key=lambda x: (x[0][0][1] + x[0][3][1]) / 2)
            
            # Group text by lines based on y-coordinate
            lines = []
            line_y = 0
            current_line = []
            
            for box in sorted_results:
                y_mid = (box[0][0][1] + box[0][3][1]) / 2
                
                if not current_line or abs(y_mid - line_y) < 20:  # Threshold for same line
                    current_line.append(box)
                    line_y = (line_y + y_mid) / 2  # Update average line y-position
                else:
                    # Sort the line horizontally
                    current_line.sort(key=lambda x: x[0][0][0])  # Sort by left x-coordinate
                    lines.append(current_line)
                    current_line = [box]
                    line_y = y_mid
            
            # Add the last line
            if current_line:
                current_line.sort(key=lambda x: x[0][0][0])
                lines.append(current_line)
            
            # Format with newlines between lines and spaces within lines
            text = "\n".join([" ".join([item[1][0] for item in line]) for line in lines])
            
        else:
            # Simple concatenation for non-layout preserving mode
            text = " ".join([item[1][0] for item in result[0]])
        
        logger.info(f"PaddleOCR processing time: {time.time() - start_time:.2f}s")
        return text
        
    except Exception as e:
        logger.error(f"Error in paddle_ocr: {e}")
        AVAILABLE_OCR_ENGINES['paddle'] = False
        return ""

def easyocr_ocr(image, preserve_layout=True):
    """Perform OCR using EasyOCR with robust error handling"""
    start_time = time.time()
    
    try:
        # Check if EasyOCR is available
        try:
            import easyocr
            has_easy_ocr = True
        except ImportError:
            logger.warning("EasyOCR not available")
            return ""
            
        # Get EasyOCR reader
        reader = get_easy_ocr()
        
        # If the model manager failed to provide a reader, try direct initialization
        if reader is None:
            try:
                logger.info("Initializing EasyOCR directly...")
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
        
        # Format the results
        if preserve_layout:
            # Sort by vertical position for layout preservation
            sorted_results = sorted(result, key=lambda x: (x[0][0][1] + x[0][2][1]) / 2)
            
            # Group text by lines based on y-coordinate
            lines = []
            line_y = 0
            current_line = []
            
            for detection in sorted_results:
                box = detection[0]  # Bounding box
                text = detection[1]  # Text
                
                y_mid = (box[0][1] + box[2][1]) / 2
                
                if not current_line or abs(y_mid - line_y) < 20:  # Threshold for same line
                    current_line.append(detection)
                    line_y = (line_y + y_mid) / 2  # Update average line y-position
                else:
                    # Sort the line horizontally
                    current_line.sort(key=lambda x: x[0][0][0])  # Sort by left x-coordinate
                    lines.append(current_line)
                    current_line = [detection]
                    line_y = y_mid
            
            # Add the last line
            if current_line:
                current_line.sort(key=lambda x: x[0][0][0])
                lines.append(current_line)
            
            # Format with newlines between lines and spaces within lines
            text = "\n".join([" ".join([item[1] for item in line]) for line in lines])
            
        else:
            # Simple concatenation for non-layout preserving mode
            text = " ".join([item[1] for item in result])
        
        logger.info(f"EasyOCR processing time: {time.time() - start_time:.2f}s")
        return text
        
    except Exception as e:
        logger.error(f"Error in easyocr_ocr: {e}")
        AVAILABLE_OCR_ENGINES['easy'] = False
        return ""

def tesseract_ocr(image, preserve_layout=True):
    """Perform OCR using Tesseract with robust error handling"""
    if not AVAILABLE_OCR_ENGINES['tesseract']:
        logger.warning("Tesseract not available")
        return ""
    
    try:
        import pytesseract
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # OCR configuration
        if preserve_layout:
            config = r'--psm 6'  # Assume a single uniform block of text
        else:
            config = r'--psm 3'  # Fully automatic page segmentation
        
        # Perform OCR
        text = pytesseract.image_to_string(image, config=config)
        
        return text
    except Exception as e:
        logger.error(f"Error in tesseract_ocr: {e}")
        AVAILABLE_OCR_ENGINES['tesseract'] = False
        return ""

def perform_ocr(image, engine="auto", preserve_layout=True, enhance_image=True):
    """
    Perform OCR with fallback mechanisms
    
    Args:
        image: PIL Image or numpy array
        engine: OCR engine to use ('paddle', 'easy', 'tesseract', 'auto', 'combined')
        preserve_layout: Whether to preserve text layout
        enhance_image: Whether to preprocess the image
        
    Returns:
        Extracted text (or error message if all methods fail)
    """
    # Check available engines
    check_ocr_engines()
    
    # Enhance image if requested
    if enhance_image:
        try:
            from PIL import ImageEnhance
            processed_image = preprocess_image(image)
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            processed_image = image
    else:
        processed_image = image
    
    # Strategy based on requested engine
    if engine == "combined" or engine == "auto":
        # Try multiple engines and use best result
        results = []
        
        # Try PaddleOCR first
        if AVAILABLE_OCR_ENGINES['paddle']:
            paddle_text = paddle_ocr(processed_image, preserve_layout)
            if paddle_text:
                results.append(("paddle", paddle_text))
        
        # Try EasyOCR
        if AVAILABLE_OCR_ENGINES['easy']:
            easy_text = easyocr_ocr(processed_image, preserve_layout)
            if easy_text:
                results.append(("easy", easy_text))
        
        # Try Tesseract as last resort
        if AVAILABLE_OCR_ENGINES['tesseract']:
            tesseract_text = tesseract_ocr(processed_image, preserve_layout)
            if tesseract_text:
                results.append(("tesseract", tesseract_text))
        
        # Pick the best result (currently using length as heuristic)
        if results:
            # Sort by text length (longer is usually better)
            results.sort(key=lambda x: len(x[1]), reverse=True)
            logger.info(f"Used {results[0][0]} engine for best OCR result")
            return results[0][1]
        else:
            return "Error: All OCR engines failed"
            
    elif engine == "paddle":
        if AVAILABLE_OCR_ENGINES['paddle']:
            result = paddle_ocr(processed_image, preserve_layout)
            if result:
                return result
                
            # If paddle fails, try fallback
            if AVAILABLE_OCR_ENGINES['easy']:
                logger.info("PaddleOCR failed, trying EasyOCR as fallback")
                result = easyocr_ocr(processed_image, preserve_layout)
                if result:
                    return result
            
            # Last resort: Tesseract
            if AVAILABLE_OCR_ENGINES['tesseract']:
                logger.info("Trying Tesseract as last resort")
                result = tesseract_ocr(processed_image, preserve_layout)
                if result:
                    return result
                    
            return "Error: PaddleOCR failed and no fallback engines are available"
        else:
            # PaddleOCR not available, try others
            if AVAILABLE_OCR_ENGINES['easy']:
                logger.info("PaddleOCR not available, using EasyOCR")
                result = easyocr_ocr(processed_image, preserve_layout)
                if result:
                    return result
            
            if AVAILABLE_OCR_ENGINES['tesseract']:
                logger.info("Using Tesseract as fallback")
                result = tesseract_ocr(processed_image, preserve_layout)
                if result:
                    return result
                    
            return "Error: PaddleOCR not available and fallbacks failed"
            
    elif engine == "easy":
        if AVAILABLE_OCR_ENGINES['easy']:
            result = easyocr_ocr(processed_image, preserve_layout)
            if result:
                return result
                
            # If EasyOCR fails, try fallback
            if AVAILABLE_OCR_ENGINES['paddle']:
                logger.info("EasyOCR failed, trying PaddleOCR as fallback")
                result = paddle_ocr(processed_image, preserve_layout)
                if result:
                    return result
            
            # Last resort: Tesseract
            if AVAILABLE_OCR_ENGINES['tesseract']:
                logger.info("Trying Tesseract as last resort")
                result = tesseract_ocr(processed_image, preserve_layout)
                if result:
                    return result
                    
            return "Error: EasyOCR failed and no fallback engines are available"
        else:
            # EasyOCR not available, try others
            if AVAILABLE_OCR_ENGINES['paddle']:
                logger.info("EasyOCR not available, using PaddleOCR")
                result = paddle_ocr(processed_image, preserve_layout)
                if result:
                    return result
            
            if AVAILABLE_OCR_ENGINES['tesseract']:
                logger.info("Using Tesseract as fallback")
                result = tesseract_ocr(processed_image, preserve_layout)
                if result:
                    return result
                    
            return "Error: EasyOCR not available and fallbacks failed"
            
    elif engine == "tesseract":
        if AVAILABLE_OCR_ENGINES['tesseract']:
            result = tesseract_ocr(processed_image, preserve_layout)
            if result:
                return result
            
            # If Tesseract fails, try others
            if AVAILABLE_OCR_ENGINES['paddle']:
                logger.info("Tesseract failed, trying PaddleOCR")
                result = paddle_ocr(processed_image, preserve_layout)
                if result:
                    return result
                
            if AVAILABLE_OCR_ENGINES['easy']:
                logger.info("Trying EasyOCR as fallback")
                result = easyocr_ocr(processed_image, preserve_layout)
                if result:
                    return result
                    
            return "Error: All OCR engines failed"
        else:
            # Tesseract not available, try others
            if AVAILABLE_OCR_ENGINES['paddle']:
                logger.info("Tesseract not available, using PaddleOCR")
                result = paddle_ocr(processed_image, preserve_layout)
                if result:
                    return result
                
            if AVAILABLE_OCR_ENGINES['easy']:
                logger.info("Using EasyOCR as fallback")
                result = easyocr_ocr(processed_image, preserve_layout)
                if result:
                    return result
                    
            return "Error: Tesseract not available and fallbacks failed"
    
    else:
        return f"Error: Unknown OCR engine '{engine}'"

if __name__ == "__main__":
    # Check which OCR engines are available
    available, missing, instructions = check_ocr_engines()
    print("Available OCR engines:", available)
    print("Missing OCR engines:", missing)
    
    if missing:
        print("\nInstallation instructions:")
        for instruction in instructions:
            print(f"- {instruction}")
