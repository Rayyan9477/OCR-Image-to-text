"""
Image Processing Module

Provides image preprocessing functions to improve OCR accuracy.
"""

import logging
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Optional, Union
import cv2

from ..config.settings import Settings

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Image processing utilities for OCR"""
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize image processor
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings or Settings()
    
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
        
        # Apply enabled preprocessing steps
        if self.settings.get("ocr.preprocessing.enhance_contrast", True):
            img = self._enhance_contrast(img)
        
        if self.settings.get("ocr.preprocessing.remove_noise", True):
            img = self._remove_noise(img)
        
        if self.settings.get("ocr.preprocessing.correct_skew", True):
            img = self._correct_skew(img)
        
        logger.debug("Applied image preprocessing")
        return img
    
    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        """Enhance image contrast"""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(2.0)  # Increase contrast
    
    def _remove_noise(self, image: Image.Image) -> Image.Image:
        """Remove noise from image"""
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply median filter to reduce noise
        return image.filter(ImageFilter.MedianFilter(size=3))
    
    def _correct_skew(self, image: Image.Image) -> Image.Image:
        """Correct image skew"""
        try:
            # Convert PIL image to OpenCV format
            img_cv = np.array(image)
            
            # Convert to grayscale if needed
            if len(img_cv.shape) == 3:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            
            # Threshold image
            thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            # Find contours
            coords = np.column_stack(np.where(thresh > 0))
            
            # Find rotated rectangle
            if len(coords) > 20:  # Only if we have enough points
                angle = cv2.minAreaRect(coords)[-1]
                
                # Adjust angle
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                
                # Only correct if angle is significant
                if abs(angle) > 0.5:
                    (h, w) = img_cv.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(img_cv, M, (w, h), flags=cv2.INTER_CUBIC, 
                                           borderMode=cv2.BORDER_REPLICATE)
                    
                    # Convert back to PIL
                    return Image.fromarray(rotated)
            
            return image  # Return original if no rotation needed
        except Exception as e:
            logger.warning(f"Error in skew correction: {e}")
            return image  # Return original on error
    
    def detect_tables(self, image: Union[Image.Image, np.ndarray]) -> bool:
        """
        Detect if image contains tables
        
        Args:
            image: Input image
            
        Returns:
            True if tables are detected, False otherwise
        """
        try:
            # Convert to OpenCV format if needed
            if isinstance(image, Image.Image):
                img_cv = np.array(image)
                if len(img_cv.shape) == 3:
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            else:
                img_cv = image
                if len(img_cv.shape) == 3:
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Threshold and find edges
            thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
            
            # Use Hough Line Transform to detect lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            if lines is not None:
                # Count horizontal and vertical lines
                h_lines = 0
                v_lines = 0
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate angle
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                    
                    # Classify as horizontal or vertical
                    if angle < 10 or angle > 170:  # Near horizontal
                        h_lines += 1
                    elif 80 < angle < 100:  # Near vertical
                        v_lines += 1
                
                # If we have several horizontal and vertical lines, likely a table
                return h_lines > 3 and v_lines > 3
                
            return False
            
        except Exception as e:
            logger.warning(f"Error in table detection: {e}")
            return False
    
    def assess_image_quality(self, image: Image.Image) -> dict:
        """
        Assess image quality metrics relevant to OCR
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            # Convert to grayscale if needed
            if image.mode != 'L':
                gray = image.convert('L')
            else:
                gray = image
            
            # Get image as numpy array
            img_array = np.array(gray)
            
            # Calculate contrast
            contrast = img_array.std()
            
            # Calculate brightness
            brightness = img_array.mean()
            
            # Calculate sharpness (using variance of Laplacian)
            img_cv = np.array(gray)
            laplacian = cv2.Laplacian(img_cv, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Calculate noise level (using difference between image and blurred version)
            blurred = cv2.GaussianBlur(img_cv, (5, 5), 0)
            noise_level = np.mean(np.abs(img_cv - blurred))
            
            # Resolution adequacy (based on image size)
            width, height = image.size
            min_dimension = min(width, height)
            res_score = min(1.0, min_dimension / 1000)  # Score of 1.0 for images 1000px or larger
            
            # Overall quality score
            overall_score = (
                self._normalize(contrast, 40, 80) * 0.3 +
                self._normalize(brightness, 100, 200) * 0.2 +
                self._normalize(sharpness, 50, 200) * 0.3 +
                (1.0 - self._normalize(noise_level, 0, 10)) * 0.1 +
                res_score * 0.1
            )
            
            return {
                "contrast": float(contrast),
                "brightness": float(brightness),
                "sharpness": float(sharpness),
                "noise_level": float(noise_level),
                "resolution": {"width": width, "height": height},
                "quality_score": float(overall_score)
            }
            
        except Exception as e:
            logger.warning(f"Error in quality assessment: {e}")
            return {
                "error": str(e),
                "quality_score": 0.5  # Default middle score on error
            }
    
    def _normalize(self, value, min_val, max_val):
        """Normalize value to 0-1 range"""
        if value < min_val:
            return value / min_val
        elif value > max_val:
            return 1.0
        else:
            return (value - min_val) / (max_val - min_val)
