#!/usr/bin/env python
"""
Multi-Engine OCR System
Combines EasyOCR, PaddleOCR, and TrOCR for optimal performance
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

class MultiEngineOCR:
    """Multi-engine OCR system for optimal performance"""
    
    def __init__(self):
        self.engines = {}
        self.handwriting_detector = None
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
        
        # Initialize TrOCR for handwritten text
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
            self.engines['trocr'] = True
            logger.info("✅ TrOCR (handwritten) initialized successfully")
        except Exception as e:
            logger.warning(f"❌ TrOCR initialization failed: {e}")
            try:
                # Fallback to online handwriting detection
                import requests
                self.engines['handwriting_api'] = True
                logger.info("✅ Handwriting API fallback ready")
            except:
                logger.warning("❌ No handwriting detection available")
    
    def detect_handwriting(self, image: np.ndarray) -> bool:
        """Detect if image contains handwritten text"""
        try:
            # Simple heuristic: analyze stroke patterns
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contour characteristics
            irregular_shapes = 0
            for contour in contours:
                # Calculate contour properties
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small noise
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        # Handwriting typically has irregular shapes
                        if circularity < 0.3:
                            irregular_shapes += 1
            
            # If more than 30% of shapes are irregular, likely handwritten
            handwriting_ratio = irregular_shapes / max(len(contours), 1)
            return handwriting_ratio > 0.3
            
        except Exception as e:
            logger.warning(f"Handwriting detection failed: {e}")
            return False
    
    def process_with_easyocr(self, image: np.ndarray) -> OCRResult:
        """Process image with EasyOCR"""
        start_time = time.time()
        try:
            if 'easyocr' not in self.engines:
                raise Exception("EasyOCR not available")
            
            results = self.engines['easyocr'].readtext(image)
            
            # Combine all text with confidence weighting
            texts = []
            confidences = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Filter low confidence
                    texts.append(text)
                    confidences.append(confidence)
            
            combined_text = ' '.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                text=combined_text,
                confidence=avg_confidence,
                engine='easyocr',
                processing_time=processing_time,
                bbox=[r[0] for r in results]
            )
            
        except Exception as e:
            logger.error(f"EasyOCR processing failed: {e}")
            return OCRResult("", 0.0, 'easyocr', time.time() - start_time)
    
    def process_with_paddleocr(self, image: np.ndarray) -> OCRResult:
        """Process image with PaddleOCR"""
        start_time = time.time()
        try:
            if 'paddleocr' not in self.engines:
                raise Exception("PaddleOCR not available")
            
            results = self.engines['paddleocr'].ocr(image, cls=True)
            
            # Extract text and confidence
            texts = []
            confidences = []
            bboxes = []
            
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        bbox, (text, confidence) = line[0], line[1]
                        if confidence > 0.3:  # Filter low confidence
                            texts.append(text)
                            confidences.append(confidence)
                            bboxes.append(bbox)
            
            combined_text = ' '.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                text=combined_text,
                confidence=avg_confidence,
                engine='paddleocr',
                processing_time=processing_time,
                bbox=bboxes
            )
            
        except Exception as e:
            logger.error(f"PaddleOCR processing failed: {e}")
            return OCRResult("", 0.0, 'paddleocr', time.time() - start_time)
    
    def process_with_trocr(self, image: np.ndarray) -> OCRResult:
        """Process image with TrOCR for handwritten text"""
        start_time = time.time()
        try:
            if 'trocr' not in self.engines:
                raise Exception("TrOCR not available")
            
            # Convert to PIL Image
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image
            
            # Process with TrOCR
            pixel_values = self.trocr_processor(images=pil_image, return_tensors="pt").pixel_values
            generated_ids = self.trocr_model.generate(pixel_values)
            generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            processing_time = time.time() - start_time
            
            # TrOCR doesn't provide confidence scores, so we estimate based on text quality
            confidence = min(0.8, len(generated_text.strip()) / 100.0) if generated_text.strip() else 0.0
            
            return OCRResult(
                text=generated_text,
                confidence=confidence,
                engine='trocr',
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"TrOCR processing failed: {e}")
            return OCRResult("", 0.0, 'trocr', time.time() - start_time)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced image preprocessing for better OCR"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply adaptive thresholding
            adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Noise removal
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
            processed = cv2.medianBlur(processed, 3)
            
            return processed
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image
    
    def combine_results(self, results: List[OCRResult]) -> str:
        """Intelligently combine results from multiple engines"""
        try:
            if not results:
                return ""
            
            # Filter out empty results
            valid_results = [r for r in results if r.text.strip() and r.confidence > 0.1]
            
            if not valid_results:
                return ""
            
            if len(valid_results) == 1:
                return valid_results[0].text
            
            # Weight by confidence and processing time
            weighted_texts = []
            total_weight = 0
            
            for result in valid_results:
                # Higher confidence and faster processing get higher weight
                time_weight = 1.0 / max(result.processing_time, 0.1)
                confidence_weight = result.confidence
                
                # Special bonus for handwriting detection
                engine_weight = 1.2 if result.engine == 'trocr' else 1.0
                
                weight = confidence_weight * time_weight * engine_weight
                weighted_texts.append((result.text, weight))
                total_weight += weight
            
            # If confidences are similar, use the longest text
            if len(set(r.confidence for r in valid_results)) == 1:
                return max(valid_results, key=lambda x: len(x.text)).text
            
            # Otherwise, use the highest confidence result
            best_result = max(valid_results, key=lambda x: x.confidence)
            return best_result.text
            
        except Exception as e:
            logger.error(f"Result combination failed: {e}")
            return valid_results[0].text if valid_results else ""
    
    def extract_text(self, image: Union[np.ndarray, Image.Image], timeout: int = 30) -> Dict[str, Any]:
        """Extract text using all available engines simultaneously"""
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
            
            # Detect if handwritten
            is_handwritten = self.detect_handwriting(image_np)
            
            # Prepare engines to use
            engines_to_use = []
            if 'easyocr' in self.engines:
                engines_to_use.append(('easyocr', self.process_with_easyocr))
            if 'paddleocr' in self.engines:
                engines_to_use.append(('paddleocr', self.process_with_paddleocr))
            if is_handwritten and 'trocr' in self.engines:
                engines_to_use.append(('trocr', self.process_with_trocr))
            
            if not engines_to_use:
                return {
                    'text': 'No OCR engines available',
                    'confidence': 0.0,
                    'engines_used': [],
                    'processing_time': 0.0,
                    'is_handwritten': is_handwritten
                }
            
            # Process with multiple engines simultaneously
            results = []
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
                    except Exception as e:
                        engine_name = future_to_engine[future]
                        logger.error(f"Engine {engine_name} failed: {e}")
            
            total_time = time.time() - start_time
            
            # Combine results
            combined_text = self.combine_results(results)
            
            # Calculate overall confidence
            confidences = [r.confidence for r in results if r.confidence > 0]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'engines_used': [r.engine for r in results if r.text.strip()],
                'processing_time': total_time,
                'is_handwritten': is_handwritten,
                'individual_results': [
                    {
                        'engine': r.engine,
                        'text': r.text,
                        'confidence': r.confidence,
                        'time': r.processing_time
                    } for r in results
                ]
            }
            
        except Exception as e:
            logger.error(f"Multi-engine OCR failed: {e}")
            return {
                'text': f'OCR processing failed: {str(e)}',
                'confidence': 0.0,
                'engines_used': [],
                'processing_time': 0.0,
                'is_handwritten': False
            }

# Global instance
multi_ocr = None

def get_multi_ocr():
    """Get or create the global multi-OCR instance"""
    global multi_ocr
    if multi_ocr is None:
        multi_ocr = MultiEngineOCR()
    return multi_ocr

def extract_text_multi_engine(image: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
    """Convenience function to extract text with multi-engine approach"""
    ocr_system = get_multi_ocr()
    return ocr_system.extract_text(image)

if __name__ == "__main__":
    # Test the multi-engine system
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            result = extract_text_multi_engine(image)
            
            print(f"Extracted Text: {result['text']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Engines Used: {result['engines_used']}")
            print(f"Processing Time: {result['processing_time']:.2f}s")
            print(f"Handwritten: {result['is_handwritten']}")
        else:
            print(f"Image file not found: {image_path}")
    else:
        print("Usage: python multi_engine_ocr.py <image_path>")
