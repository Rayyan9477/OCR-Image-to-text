"""
Optimized OCR Engine for Multiple Backends

This module provides an enhanced OCR engine that can work with available engines
and provides optimal performance through intelligent engine selection.
"""

import os
import sys
import logging
import importlib.util
import time
import io
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2

logger = logging.getLogger(__name__)

class OptimizedImageProcessor:
    """Enhanced image processor with fallback capabilities"""
    
    def __init__(self):
        """Initialize the image processor"""
        self.cv2_available = self._check_cv2()
    
    def _check_cv2(self) -> bool:
        """Check if OpenCV is available"""
        try:
            import cv2
            return True
        except ImportError:
            logger.warning("OpenCV not available, using PIL-only processing")
            return False
    
    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Apply preprocessing steps to enhance OCR accuracy
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            Processed PIL Image
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Make a copy to avoid modifying the original
        img = image.copy()
        
        # Convert to grayscale for better OCR
        if img.mode != 'L':
            img = img.convert('L')
        
        # Enhance contrast
        img = self._enhance_contrast(img)
        
        # Remove noise
        img = self._remove_noise(img)
        
        # Resize if too small (minimum 300px width)
        width, height = img.size
        if width < 300:
            scale = 300 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return img
    
    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        """Enhance image contrast"""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.5)
    
    def _remove_noise(self, image: Image.Image) -> Image.Image:
        """Remove noise from image"""
        # Apply a slight gaussian blur to reduce noise
        return image.filter(ImageFilter.GaussianBlur(0.5))
    
    def assess_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        """
        Assess image quality for OCR
        
        Args:
            image: PIL Image to assess
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            # Convert to grayscale if needed
            if image.mode != 'L':
                gray_image = image.convert('L')
            else:
                gray_image = image
            
            # Convert to numpy array for analysis
            img_array = np.array(gray_image)
            
            # Calculate basic quality metrics
            mean_brightness = np.mean(img_array)
            std_brightness = np.std(img_array)
            
            # Calculate contrast (standard deviation of pixel values)
            contrast = std_brightness / 255.0
            
            # Calculate quality score (0-1)
            # Good images have moderate brightness and good contrast
            brightness_score = 1.0 - abs(mean_brightness - 127) / 127
            contrast_score = min(contrast * 2, 1.0)  # Cap at 1.0
            
            quality_score = (brightness_score + contrast_score) / 2
            
            return {
                'quality_score': quality_score,
                'mean_brightness': mean_brightness,
                'contrast': contrast,
                'width': image.width,
                'height': image.height,
                'total_pixels': image.width * image.height
            }
        except Exception as e:
            logger.warning(f"Error assessing image quality: {e}")
            return {
                'quality_score': 0.5,
                'mean_brightness': 127,
                'contrast': 0.5,
                'width': image.width,
                'height': image.height,
                'total_pixels': image.width * image.height
            }
    
    def detect_tables(self, image: Image.Image) -> bool:
        """
        Simple table detection based on line density
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            True if tables are likely present
        """
        try:
            # Convert to grayscale
            if image.mode != 'L':
                gray_image = image.convert('L')
            else:
                gray_image = image
            
            # Convert to numpy array
            img_array = np.array(gray_image)
            
            # Simple line detection using edge detection
            # Apply threshold to create binary image
            _, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY) if self.cv2_available else (None, None)
            
            if binary is not None:
                # Count horizontal and vertical lines
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
                vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
                
                horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
                vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
                
                h_line_count = np.sum(horizontal_lines > 0)
                v_line_count = np.sum(vertical_lines > 0)
                
                # If we have significant horizontal and vertical lines, likely a table
                total_pixels = img_array.size
                line_density = (h_line_count + v_line_count) / total_pixels
                
                return line_density > 0.01  # Threshold for table detection
            else:
                # Fallback: simple grid pattern detection using PIL
                # Look for regular patterns in the image
                width, height = gray_image.size
                
                # Sample a few horizontal and vertical lines
                mid_height = height // 2
                mid_width = width // 2
                
                # Get horizontal line (middle row)
                h_line = [gray_image.getpixel((x, mid_height)) for x in range(width)]
                # Get vertical line (middle column)  
                v_line = [gray_image.getpixel((mid_width, y)) for y in range(height)]
                
                # Count transitions (dark to light or light to dark)
                h_transitions = sum(1 for i in range(1, len(h_line)) if abs(h_line[i] - h_line[i-1]) > 50)
                v_transitions = sum(1 for i in range(1, len(v_line)) if abs(v_line[i] - v_line[i-1]) > 50)
                
                # If we have many transitions, it might be a table
                return h_transitions > 10 and v_transitions > 10
                
        except Exception as e:
            logger.warning(f"Error detecting tables: {e}")
            return False


class OptimizedOCREngine:
    """Optimized OCR engine with multiple backend support"""
    
    def __init__(self):
        """Initialize the OCR engine"""
        self.available_engines = self._check_engines()
        self.enabled_engines = [engine for engine, available in self.available_engines.items() if available]
        self.image_processor = OptimizedImageProcessor()
        
        # Initialize available engines
        self._initialize_engines()
        
        logger.info(f"Initialized OCR engine with: {self.enabled_engines}")
    
    def _check_engines(self) -> Dict[str, bool]:
        """Check which OCR engines are available"""
        engines = {
            "tesseract": False,
            "easyocr": False,
            "paddleocr": False
        }
        
        # Check Tesseract
        try:
            import pytesseract
            import platform
            
            # Try to auto-detect Tesseract installation on Windows
            if platform.system() == "Windows":
                common_paths = [
                    "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
                    "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
                    f"C:\\Users\\{os.getenv('USERNAME')}\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
                ]
                for path in common_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        logger.info(f"Found Tesseract at: {path}")
                        break
            
            # Test if Tesseract actually works
            try:
                pytesseract.get_tesseract_version()
                engines["tesseract"] = True
                logger.info("Tesseract OCR available")
            except Exception as e:
                logger.warning(f"Tesseract installed but not working: {e}")
        except ImportError:
            logger.warning("Pytesseract not installed")
        
        # Check EasyOCR
        try:
            spec = importlib.util.find_spec('easyocr')
            if spec is not None:
                engines["easyocr"] = True
                logger.info("EasyOCR available")
        except Exception as e:
            logger.warning(f"EasyOCR check failed: {e}")
        
        # Check PaddleOCR (with more robust error handling)
        try:
            spec = importlib.util.find_spec('paddleocr')
            if spec is not None:
                # Try a safe import test
                try:
                    # Don't actually import paddleocr here to avoid dependency issues
                    # Just check if the module exists
                    engines["paddleocr"] = True
                    logger.info("PaddleOCR module available (will test during initialization)")
                except Exception as e:
                    logger.warning(f"PaddleOCR available but may have issues: {e}")
        except Exception as e:
            logger.warning(f"PaddleOCR check failed: {e}")
        
        return engines
    
    def _initialize_engines(self):
        """Initialize the available OCR engines"""
        self.engines = {}
        
        # Initialize Tesseract
        if self.available_engines.get("tesseract", False):
            try:
                import pytesseract
                self.engines["tesseract"] = pytesseract
                logger.info("Tesseract engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Tesseract: {e}")
                self.available_engines["tesseract"] = False
        
        # Initialize EasyOCR
        if self.available_engines.get("easyocr", False):
            try:
                import easyocr
                self.engines["easyocr"] = easyocr.Reader(['en'])
                logger.info("EasyOCR engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                self.available_engines["easyocr"] = False
        
        # Initialize PaddleOCR with careful error handling
        if self.available_engines.get("paddleocr", False):
            try:
                import paddleocr
                self.engines["paddleocr"] = paddleocr.PaddleOCR(use_angle_cls=True, lang='en')
                logger.info("PaddleOCR engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {e}")
                self.available_engines["paddleocr"] = False
        
        # Update enabled engines list
        self.enabled_engines = [engine for engine, available in self.available_engines.items() if available and engine in self.engines]
    
    def perform_ocr(self, image: Union[Image.Image, np.ndarray], engine: str = 'auto', 
                    preserve_layout: bool = True, preprocess: bool = True) -> str:
        """
        Perform OCR on an image using the specified engine
        
        Args:
            image: Input image (PIL Image or numpy array)
            engine: OCR engine to use ('auto', 'tesseract', 'easyocr', 'paddleocr')
            preserve_layout: Whether to preserve text layout
            preprocess: Whether to preprocess the image
            
        Returns:
            Extracted text as string
        """
        if not self.enabled_engines:
            raise RuntimeError("No OCR engines available. Please install at least one OCR engine.")
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Preprocess image if requested
        if preprocess:
            image = self.image_processor.preprocess_image(image)
        
        # Determine which engine to use
        if engine == 'auto':
            # Use the best available engine
            if 'tesseract' in self.enabled_engines:
                engine = 'tesseract'
            elif 'easyocr' in self.enabled_engines:
                engine = 'easyocr'
            elif 'paddleocr' in self.enabled_engines:
                engine = 'paddleocr'
            else:
                raise RuntimeError("No suitable OCR engine found")
        
        if engine not in self.enabled_engines:
            raise ValueError(f"Engine '{engine}' is not available. Available engines: {self.enabled_engines}")
        
        # Perform OCR with the selected engine
        try:
            if engine == 'tesseract':
                return self._tesseract_ocr(image, preserve_layout)
            elif engine == 'easyocr':
                return self._easyocr_ocr(image)
            elif engine == 'paddleocr':
                return self._paddleocr_ocr(image)
            else:
                raise ValueError(f"Unknown engine: {engine}")
        except Exception as e:
            logger.error(f"OCR failed with {engine}: {e}")
            # Try fallback engines
            return self._fallback_ocr(image, engine, preserve_layout, preprocess)
    
    def _tesseract_ocr(self, image: Image.Image, preserve_layout: bool = True) -> str:
        """Perform OCR using Tesseract"""
        config = '--psm 6'  # Uniform block of text
        if preserve_layout:
            config = '--psm 6 -c preserve_interword_spaces=1'
        
        return self.engines["tesseract"].image_to_string(image, config=config).strip()
    
    def _easyocr_ocr(self, image: Image.Image) -> str:
        """Perform OCR using EasyOCR"""
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Perform OCR
        results = self.engines["easyocr"].readtext(img_array)
        
        # Extract text from results
        text_lines = []
        for (bbox, text, confidence) in results:
            if confidence > 0.1:  # Filter low confidence results
                text_lines.append(text)
        
        return '\\n'.join(text_lines)
    
    def _paddleocr_ocr(self, image: Image.Image) -> str:
        """Perform OCR using PaddleOCR"""
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Perform OCR
        results = self.engines["paddleocr"].ocr(img_array, cls=True)
        
        # Extract text from results
        text_lines = []
        for line in results:
            if line:
                for word_info in line:
                    if len(word_info) >= 2:
                        text = word_info[1][0] if isinstance(word_info[1], (list, tuple)) else word_info[1]
                        confidence = word_info[1][1] if isinstance(word_info[1], (list, tuple)) and len(word_info[1]) > 1 else 1.0
                        if confidence > 0.1:  # Filter low confidence results
                            text_lines.append(text)
        
        return '\\n'.join(text_lines)
    
    def _fallback_ocr(self, image: Image.Image, failed_engine: str, preserve_layout: bool, preprocess: bool) -> str:
        """Try fallback engines if the primary engine fails"""
        fallback_order = ['tesseract', 'easyocr', 'paddleocr']
        fallback_order = [eng for eng in fallback_order if eng != failed_engine and eng in self.enabled_engines]
        
        for engine in fallback_order:
            try:
                logger.info(f"Trying fallback engine: {engine}")
                if engine == 'tesseract':
                    return self._tesseract_ocr(image, preserve_layout)
                elif engine == 'easyocr':
                    return self._easyocr_ocr(image)
                elif engine == 'paddleocr':
                    return self._paddleocr_ocr(image)
            except Exception as e:
                logger.warning(f"Fallback engine {engine} also failed: {e}")
                continue
        
        raise RuntimeError("All OCR engines failed")
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about available engines"""
        return {
            'available_engines': self.available_engines,
            'enabled_engines': self.enabled_engines,
            'total_available': len(self.enabled_engines)
        }


# Compatibility functions for the main app
class ImageProcessor:
    """Compatibility wrapper for the main app"""
    
    def __init__(self, settings=None):
        self.processor = OptimizedImageProcessor()
    
    def assess_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        return self.processor.assess_image_quality(image)
    
    def detect_tables(self, image: Image.Image) -> bool:
        return self.processor.detect_tables(image)
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        return self.processor.preprocess_image(image)


class OCREngine:
    """Compatibility wrapper for the main app"""
    
    def __init__(self, settings=None):
        self.engine = OptimizedOCREngine()
        self.enabled_engines = self.engine.enabled_engines
    
    def perform_ocr(self, image: Union[Image.Image, np.ndarray], engine: str = 'auto', 
                    preserve_layout: bool = True, preprocess: bool = True) -> str:
        return self.engine.perform_ocr(image, engine, preserve_layout, preprocess)
    
    def get_engine_info(self) -> Dict[str, Any]:
        return self.engine.get_engine_info()


if __name__ == "__main__":
    # Test the optimized OCR engine
    engine = OptimizedOCREngine()
    processor = OptimizedImageProcessor()
    
    print("OCR Engine Status:")
    info = engine.get_engine_info()
    print(f"Available engines: {info['available_engines']}")
    print(f"Enabled engines: {info['enabled_engines']}")
    print(f"Total available: {info['total_available']}")
