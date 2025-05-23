# pytesseract OCR fallback implementation
import pytesseract
import numpy as np
from PIL import Image
import logging
import cv2

logger = logging.getLogger(__name__)

def tesseract_available():
    """Check if pytesseract is properly installed and available"""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception as e:
        logger.error(f"Tesseract OCR not available: {e}")
        return False

def perform_tesseract_ocr(image, preserve_layout=True):
    """
    Use pytesseract as a fallback OCR method when other engines fail
    
    Args:
        image: PIL Image or numpy array
        preserve_layout: Whether to preserve layout with page segmentation mode
        
    Returns:
        Extracted text
    """
    try:
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
        
        # Convert to grayscale if needed
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_np
            
        # Apply adaptive threshold to improve readability
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Set up tesseract OCR configuration
        config = ""
        if preserve_layout:
            # Use --psm 6 for block of text with layout preservation
            config = "--psm 6"
        else:
            # Use --psm 11 for sparse text without layout concerns
            config = "--psm 11"
            
        # Add extra OCR Engine configuration for better accuracy
        config += " --oem 3"  # Use LSTM OCR Engine (most accurate)
        
        # Perform OCR
        text = pytesseract.image_to_string(thresh, config=config)
        return text
    except Exception as e:
        logger.error(f"Tesseract OCR error: {e}")
        return "Error performing OCR with Tesseract"