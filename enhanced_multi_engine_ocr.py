#!/usr/bin/env python
"""
Enhanced Multi-Engine OCR System
Optimized version with better error handling and fallbacks
"""

import os
import cv2
import numpy as np
from PIL import Image
import concurrent.futures
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import time
import threading
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """OCR result with confidence and metadata"""
    text: str
    confidence: float
    engine: str
    processing_time: float
    bbox: Optional[List] = None
    status: str = "success"
    error: Optional[str] = None
    raw_results: Optional[List] = None  # Store raw results for layout analysis

class EnhancedMultiEngineOCR:
    """Enhanced multi-engine OCR system with better fallbacks"""
    
    def __init__(self):
        self.engines = {}
        self.initialize_engines()
    
    def initialize_engines(self):
        """Initialize all available OCR engines"""
        
        # Initialize EasyOCR
        try:
            import easyocr
            self.engines['easyocr'] = easyocr.Reader(['en'], gpu=False)
            logger.info("✅ EasyOCR initialized successfully")
        except Exception as e:
            logger.warning(f"❌ EasyOCR initialization failed: {e}")
        
        # Initialize PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self.engines['paddleocr'] = PaddleOCR(
                use_angle_cls=True, 
                lang='en',
                use_gpu=False,
                show_log=False
            )
            logger.info("✅ PaddleOCR initialized successfully")
        except Exception as e:
            logger.warning(f"❌ PaddleOCR initialization failed: {e}")
        
        # Initialize Tesseract (fallback)
        try:
            import pytesseract
            # Test if tesseract is available
            pytesseract.get_tesseract_version()
            self.engines['tesseract'] = True
            logger.info("✅ Tesseract initialized successfully")
        except Exception as e:
            logger.warning(f"❌ Tesseract initialization failed: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Basic image preprocessing"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            return enhanced
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image
    
    def detect_handwriting(self, image: np.ndarray) -> bool:
        """Simple handwriting detection"""
        try:
            # Basic heuristic: analyze edge density and irregularity
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
            
            # High edge density might indicate handwriting
            return edge_density > 0.05
        except:
            return False
    
    def process_with_easyocr(self, image: np.ndarray) -> OCRResult:
        """Process image with EasyOCR"""
        start_time = time.time()
        try:
            if 'easyocr' not in self.engines:
                return OCRResult("", 0.0, "easyocr", 0.0, status="error", error="Engine not available")
            
            results = self.engines['easyocr'].readtext(image)
            
            if not results:
                return OCRResult("", 0.0, "easyocr", time.time() - start_time, status="success")
            
            # Combine all text
            text_parts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.1:  # Filter low confidence results
                    text_parts.append(text)
                    confidences.append(confidence)
            
            combined_text = " ".join(text_parts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return OCRResult(
                combined_text, 
                avg_confidence, 
                "easyocr", 
                time.time() - start_time,
                status="success",
                raw_results=results  # Store raw results for layout analysis
            )
            
        except Exception as e:
            logger.error(f"EasyOCR processing failed: {e}")
            return OCRResult("", 0.0, "easyocr", time.time() - start_time, status="error", error=str(e))
    
    def process_with_paddleocr(self, image: np.ndarray) -> OCRResult:
        """Process image with PaddleOCR"""
        start_time = time.time()
        try:
            if 'paddleocr' not in self.engines:
                return OCRResult("", 0.0, "paddleocr", 0.0, status="error", error="Engine not available")
            
            results = self.engines['paddleocr'].ocr(image, cls=True)
            
            if not results or not results[0]:
                return OCRResult("", 0.0, "paddleocr", time.time() - start_time, status="success")
            
            # Extract text and confidence
            text_parts = []
            confidences = []
            
            for line in results[0]:
                if line:
                    bbox, (text, confidence) = line
                    if confidence > 0.1:  # Filter low confidence results
                        text_parts.append(text)
                        confidences.append(confidence)
            
            combined_text = " ".join(text_parts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return OCRResult(
                combined_text, 
                avg_confidence, 
                "paddleocr", 
                time.time() - start_time,
                status="success",
                raw_results=results  # Store raw results for layout analysis
            )
            
        except Exception as e:
            logger.error(f"PaddleOCR processing failed: {e}")
            return OCRResult("", 0.0, "paddleocr", time.time() - start_time, status="error", error=str(e))
    
    def process_with_tesseract(self, image: np.ndarray) -> OCRResult:
        """Process image with Tesseract (fallback)"""
        start_time = time.time()
        try:
            if 'tesseract' not in self.engines:
                return OCRResult("", 0.0, "tesseract", 0.0, status="error", error="Engine not available")
            
            import pytesseract
            
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Extract text
            text = pytesseract.image_to_string(pil_image).strip()
            
            # Get confidence data
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0
            
            return OCRResult(
                text, 
                avg_confidence, 
                "tesseract", 
                time.time() - start_time,
                status="success",
                raw_results=data  # Store raw results for layout analysis
            )
            
        except Exception as e:
            logger.error(f"Tesseract processing failed: {e}")
            return OCRResult("", 0.0, "tesseract", time.time() - start_time, status="error", error=str(e))
    
    def combine_results(self, results: List[OCRResult]) -> str:
        """Combine results from multiple engines intelligently"""
        try:
            valid_results = [r for r in results if r.status == "success" and r.text.strip()]
            
            if not valid_results:
                return ""
            
            if len(valid_results) == 1:
                return valid_results[0].text
            
            # Use the result with highest confidence
            best_result = max(valid_results, key=lambda x: x.confidence)
            return best_result.text
            
        except Exception as e:
            logger.error(f"Result combination failed: {e}")
            return valid_results[0].text if valid_results else ""
    
    def get_image_quality_assessment(self, image: np.ndarray) -> Dict[str, float]:
        """Assess image quality for OCR"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Calculate various quality metrics
            height, width = gray.shape
            
            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = min(laplacian_var / 1000.0, 1.0)  # Normalize
            
            # Contrast (standard deviation)
            contrast = gray.std() / 128.0
            
            # Brightness (mean)
            brightness = gray.mean() / 255.0
            
            # Resolution score
            resolution_score = min((width * height) / (1000 * 1000), 1.0)
            
            # Overall score
            overall_score = (sharpness * 0.3 + contrast * 0.3 + brightness * 0.2 + resolution_score * 0.2)
            
            return {
                'sharpness': sharpness,
                'contrast': contrast,
                'brightness': brightness,
                'resolution': resolution_score,
                'overall_score': overall_score
            }
        except Exception as e:
            logger.error(f"Image quality assessment failed: {e}")
            return {'overall_score': 0.5}
    
    def extract_text(self, image: Union[np.ndarray, Image.Image], timeout: int = 30) -> Dict[str, Any]:
        """Extract text using all available engines with comprehensive results"""
        try:
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                image_np = np.array(image)
                if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_np = image.copy()
            
            # Preprocess image
            processed_image = self.preprocess_image(image_np)
            
            # Assess image quality
            image_quality = self.get_image_quality_assessment(image_np)
            
            # Detect if handwritten
            is_handwritten = self.detect_handwriting(image_np)
            
            # Prepare engines to use
            engines_to_use = []
            if 'easyocr' in self.engines:
                engines_to_use.append(('easyocr', self.process_with_easyocr))
            if 'paddleocr' in self.engines:
                engines_to_use.append(('paddleocr', self.process_with_paddleocr))
            if 'tesseract' in self.engines:
                engines_to_use.append(('tesseract', self.process_with_tesseract))
            
            if not engines_to_use:
                return {
                    'text': 'No OCR engines available',
                    'confidence': 0.0,
                    'engines_used': [],
                    'processing_time': 0.0,
                    'is_handwritten': is_handwritten,
                    'image_quality': image_quality,
                    'results': {},
                    'best_result': {
                        'text': 'No OCR engines available',
                        'confidence': 0.0,
                        'engine': 'none'
                    }
                }
            
            # Process with multiple engines simultaneously
            results = []
            results_dict = {}
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(engines_to_use)) as executor:
                # Submit all engines
                future_to_engine = {}
                for engine_name, engine_func in engines_to_use:
                    future = executor.submit(engine_func, processed_image)
                    future_to_engine[future] = engine_name
                
                # Collect results with timeout
                for future in concurrent.futures.as_completed(future_to_engine, timeout=timeout):
                    try:
                        result = future.result()
                        results.append(result)
                        engine_name = future_to_engine[future]
                        results_dict[engine_name] = {
                            'text': result.text,
                            'confidence': result.confidence,
                            'processing_time': result.processing_time,
                            'status': result.status,
                            'error': result.error
                        }
                    except Exception as e:
                        engine_name = future_to_engine[future]
                        logger.error(f"Engine {engine_name} failed: {e}")
                        results_dict[engine_name] = {
                            'text': '',
                            'confidence': 0.0,
                            'processing_time': 0.0,
                            'status': 'error',
                            'error': str(e)
                        }
            
            total_time = time.time() - start_time
            
            # Find best result
            valid_results = [r for r in results if r.status == "success" and r.text.strip()]
            if valid_results:
                best_result = max(valid_results, key=lambda x: x.confidence)
                best_result_dict = {
                    'text': best_result.text,
                    'confidence': best_result.confidence,
                    'engine': best_result.engine
                }
                combined_text = best_result.text
            else:
                best_result_dict = {
                    'text': 'No text detected',
                    'confidence': 0.0,
                    'engine': 'none'
                }
                combined_text = 'No text detected'
            
            # Calculate overall confidence
            confidences = [r.confidence for r in results if r.status == "success" and r.confidence > 0]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'engines_used': [r.engine for r in results if r.status == "success" and r.text.strip()],
                'processing_time': total_time,
                'is_handwritten': is_handwritten,
                'image_quality': image_quality,
                'results': results_dict,
                'best_result': best_result_dict,
                'individual_results': [
                    {
                        'engine': r.engine,
                        'text': r.text,
                        'confidence': r.confidence,
                        'time': r.processing_time,
                        'status': r.status,
                        'error': r.error
                    } for r in results
                ]
            }
            
        except Exception as e:
            logger.error(f"Multi-engine OCR failed: {e}")
            error_result = {
                'text': f'OCR processing failed: {str(e)}',
                'confidence': 0.0,
                'engines_used': [],
                'processing_time': 0.0,
                'is_handwritten': False,
                'image_quality': {'overall_score': 0.0},
                'results': {},
                'best_result': {
                    'text': f'OCR processing failed: {str(e)}',
                    'confidence': 0.0,
                    'engine': 'error'
                }
            }
            return error_result

# Global instance
enhanced_multi_ocr = None

def get_enhanced_multi_ocr():
    """Get or create the global enhanced multi-OCR instance"""
    global enhanced_multi_ocr
    if enhanced_multi_ocr is None:
        enhanced_multi_ocr = EnhancedMultiEngineOCR()
    return enhanced_multi_ocr

def extract_text_enhanced_multi_engine(image: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
    """Convenience function to extract text with enhanced multi-engine approach"""
    ocr_system = get_enhanced_multi_ocr()
    return ocr_system.extract_text(image)

if __name__ == "__main__":
    # Test the enhanced multi-engine system
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            result = extract_text_enhanced_multi_engine(image)
            
            print(f"Best Result: {result['best_result']['text']}")
            print(f"Confidence: {result['best_result']['confidence']:.2f}")
            print(f"Engine: {result['best_result']['engine']}")
            print(f"Processing Time: {result['processing_time']:.2f}s")
            print(f"Image Quality: {result['image_quality']['overall_score']:.2f}")
        else:
            print(f"Image file not found: {image_path}")
    else:
        print("Usage: python enhanced_multi_engine_ocr.py <image_path>")
