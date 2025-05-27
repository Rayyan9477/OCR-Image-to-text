#!/usr/bin/env python
"""
Enhanced Image Processor for Multi-Engine OCR
Handles image preprocessing, quality assessment, and optimization
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging
from typing import Dict, Any, Tuple, Optional, Union
import os

logger = logging.getLogger(__name__)

class EnhancedImageProcessor:
    """Enhanced image processor with quality assessment and optimization"""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    def assess_image_quality(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Assess image quality for OCR processing"""
        try:
            # Convert to numpy array if PIL Image
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image.copy()
            
            # Convert to grayscale for analysis
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Calculate various quality metrics
            height, width = gray.shape
            
            # 1. Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 100.0, 1.0)
            
            # 2. Contrast (standard deviation)
            contrast_score = min(np.std(gray) / 127.0, 1.0)
            
            # 3. Brightness (mean intensity)
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Optimal around 0.5
            
            # 4. Resolution score
            pixel_count = height * width
            resolution_score = min(pixel_count / (1920 * 1080), 1.0)  # Normalize to Full HD
            
            # 5. Noise assessment
            noise_level = self._estimate_noise(gray)
            noise_score = max(0.0, 1.0 - noise_level)
            
            # Calculate overall quality score
            weights = {
                'sharpness': 0.3,
                'contrast': 0.25,
                'brightness': 0.2,
                'resolution': 0.15,
                'noise': 0.1
            }
            
            quality_score = (
                weights['sharpness'] * sharpness_score +
                weights['contrast'] * contrast_score +
                weights['brightness'] * brightness_score +
                weights['resolution'] * resolution_score +
                weights['noise'] * noise_score
            )
            
            return {
                'quality_score': round(quality_score, 3),
                'sharpness': round(sharpness_score, 3),
                'contrast': round(contrast_score, 3),
                'brightness': round(brightness_score, 3),
                'resolution': round(resolution_score, 3),
                'noise_level': round(noise_level, 3),
                'dimensions': (width, height),
                'recommendations': self._get_recommendations(
                    sharpness_score, contrast_score, brightness_score, noise_level
                )
            }
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {
                'quality_score': 0.5,
                'error': str(e),
                'recommendations': ["Image quality assessment failed"]
            }
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate noise level in the image"""
        try:
            # Use Laplacian to detect edges, then measure noise in non-edge areas
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            edges = np.abs(laplacian) > np.std(laplacian) * 0.5
            
            # Calculate noise in non-edge areas
            non_edge_pixels = image[~edges]
            if len(non_edge_pixels) > 0:
                noise_level = np.std(non_edge_pixels) / 255.0
            else:
                noise_level = 0.0
            
            return min(noise_level, 1.0)
        except:
            return 0.5
    
    def _get_recommendations(self, sharpness: float, contrast: float, 
                           brightness: float, noise: float) -> list:
        """Get recommendations for image improvement"""
        recommendations = []
        
        if sharpness < 0.5:
            recommendations.append("Image appears blurry - consider using a sharper image")
        if contrast < 0.4:
            recommendations.append("Low contrast detected - enhance contrast for better OCR")
        if brightness < 0.3:
            recommendations.append("Image is too dark - increase brightness")
        elif brightness > 0.8:
            recommendations.append("Image is too bright - reduce brightness")
        if noise > 0.6:
            recommendations.append("High noise detected - apply noise reduction")
        
        if not recommendations:
            recommendations.append("Image quality is good for OCR processing")
        
        return recommendations
    
    def detect_tables(self, image: Union[np.ndarray, Image.Image]) -> bool:
        """Detect if image contains table structures"""
        try:
            # Convert to numpy array if PIL Image
            if isinstance(image, Image.Image):
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_array
            else:
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Detect horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Detect vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            
            # Count line intersections
            combined = cv2.add(horizontal_lines, vertical_lines)
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If we find multiple rectangular structures, likely a table
            rectangles = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Filter small noise
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) == 4:  # Rectangle
                        rectangles += 1
            
            # If we have multiple rectangles and both horizontal and vertical lines
            horizontal_count = len([c for c in cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if cv2.contourArea(c) > 500])
            vertical_count = len([c for c in cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if cv2.contourArea(c) > 500])
            
            return rectangles >= 2 and horizontal_count >= 2 and vertical_count >= 2
            
        except Exception as e:
            logger.warning(f"Table detection failed: {e}")
            return False
    
    def enhance_for_ocr(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Enhance image specifically for OCR processing"""
        try:
            # Convert to PIL Image if numpy array
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    pil_image = Image.fromarray(image)
            else:
                pil_image = image.copy()
            
            # Convert to grayscale for processing
            if pil_image.mode != 'L':
                gray_image = pil_image.convert('L')
            else:
                gray_image = pil_image.copy()
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(gray_image)
            enhanced = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.3)
            
            # Apply slight noise reduction
            enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
            
            # Convert back to RGB for consistent output
            enhanced_rgb = enhanced.convert('RGB')
            
            return enhanced_rgb
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            # Return original image if enhancement fails
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    return Image.fromarray(image)
            return image
    
    def preprocess_for_engine(self, image: Union[np.ndarray, Image.Image], 
                            engine: str = 'multi') -> np.ndarray:
        """Preprocess image specifically for different OCR engines"""
        try:
            # Convert to numpy array
            if isinstance(image, Image.Image):
                img_array = np.array(image)
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_array = image.copy()
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_array.copy()
            
            # Engine-specific preprocessing
            if engine == 'easyocr':
                # EasyOCR works well with adaptive thresholding
                processed = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
            elif engine == 'paddleocr':
                # PaddleOCR benefits from noise reduction
                processed = cv2.bilateralFilter(gray, 9, 75, 75)
                _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif engine == 'trocr':
                # TrOCR works better with original grayscale
                processed = gray
            else:
                # Multi-engine: balanced preprocessing
                processed = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                # Apply slight noise reduction
                kernel = np.ones((1, 1), np.uint8)
                processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            
            return processed
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            # Return original image if preprocessing fails
            if isinstance(image, Image.Image):
                return np.array(image)
            return image
    
    def validate_image(self, image_data: Any) -> Tuple[bool, str]:
        """Validate if the image data is suitable for OCR"""
        try:
            if image_data is None:
                return False, "No image data provided"
            
            # Try to load as PIL Image
            if isinstance(image_data, Image.Image):
                pil_image = image_data
            elif isinstance(image_data, np.ndarray):
                if len(image_data.shape) < 2:
                    return False, "Invalid image dimensions"
                pil_image = Image.fromarray(image_data)
            else:
                # Try to open as file
                try:
                    pil_image = Image.open(image_data)
                except:
                    return False, "Cannot open image file"
            
            # Check dimensions
            width, height = pil_image.size
            if width < 50 or height < 50:
                return False, "Image too small (minimum 50x50 pixels)"
            
            if width * height > 50000000:  # ~50MP limit
                return False, "Image too large (maximum ~50MP)"
            
            # Check format
            if hasattr(pil_image, 'format'):
                format_ext = f".{pil_image.format.lower()}"
                if format_ext not in self.supported_formats:
                    return False, f"Unsupported format: {pil_image.format}"
            
            return True, "Image is valid for OCR processing"
            
        except Exception as e:
            return False, f"Image validation failed: {str(e)}"

# Global instance
enhanced_processor = None

def get_image_processor():
    """Get or create the global image processor instance"""
    global enhanced_processor
    if enhanced_processor is None:
        enhanced_processor = EnhancedImageProcessor()
    return enhanced_processor
