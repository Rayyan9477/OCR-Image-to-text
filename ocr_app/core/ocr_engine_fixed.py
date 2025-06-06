"""
Core OCR Engine Module

This module provides the main OCR functionality with support for multiple OCR engines.
"""

import os
import logging
import importlib
import time
import io
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import numpy as np
from PIL import Image

from ..config.settings import Settings
from .image_processor import ImageProcessor
from ..utils.performance import PerformanceOptimizer, CacheManager, ProgressTracker

logger = logging.getLogger(__name__)

class OCREngine:
    """Main OCR engine that coordinates multiple OCR backends"""
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the OCR engine with configuration
        
        Args:
            settings: Configuration settings (optional, will create default if None)
        """
        self.settings = settings or Settings()
        self.available_engines = self._check_engines()
        self.image_processor = ImageProcessor(self.settings)
        
        # Initialize performance components
        self.performance_optimizer = PerformanceOptimizer(self.settings)
        self.cache_manager = CacheManager()
        
        self._initialize_engines()
    
    def _check_engines(self) -> Dict[str, bool]:
        """Check which OCR engines are available on the system"""
        engines = {
            "tesseract": False,
            "easyocr": False,
            "paddleocr": False
        }
        
        # Check tesseract
        try:
            import pytesseract
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
            importlib.util.find_spec('easyocr')
            engines["easyocr"] = True
            logger.info("EasyOCR available")
        except ImportError:
            logger.warning("EasyOCR not installed")
        
        # Check PaddleOCR
        try:
            importlib.util.find_spec('paddleocr')
            engines["paddleocr"] = True
            logger.info("PaddleOCR available")
        except ImportError:
            logger.warning("PaddleOCR not installed")
            
        return engines
    
    def _initialize_engines(self):
        """Initialize the enabled OCR engines"""
        self.engines = {}
        
        # Only initialize engines that are both available and enabled in settings
        if self.available_engines.get("tesseract", False) and self.settings.get("ocr.engines.tesseract.enabled", True):
            try:
                self.engines["tesseract"] = TesseractEngine(self.settings)
                logger.info("Initialized Tesseract engine")
            except Exception as e:
                logger.error(f"Failed to initialize Tesseract engine: {e}")
        
        if self.available_engines.get("easyocr", False) and self.settings.get("ocr.engines.easyocr.enabled", True):
            try:
                self.engines["easyocr"] = EasyOCREngine(self.settings)
                logger.info("Initialized EasyOCR engine")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR engine: {e}")
        
        if self.available_engines.get("paddleocr", False) and self.settings.get("ocr.engines.paddleocr.enabled", True):
            try:
                self.engines["paddleocr"] = PaddleOCREngine(self.settings)
                logger.info("Initialized PaddleOCR engine")
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR engine: {e}")
    
    def perform_ocr(self, 
                   image: Union[str, Path, Image.Image, np.ndarray], 
                   engine: str = "auto", 
                   preserve_layout: bool = True,
                   preprocess: bool = True) -> str:
        """
        Perform OCR on an image
        
        Args:
            image: Image to process (path, PIL Image, or numpy array)
            engine: OCR engine to use ('auto', 'tesseract', 'easyocr', 'paddleocr', 'combined')
            preserve_layout: Whether to preserve text layout in output
            preprocess: Whether to apply image preprocessing
            
        Returns:
            Extracted text as string
        """
        # Convert image to PIL Image if it's a path or numpy array
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            from PIL import Image
            image = Image.fromarray(image)
        
        # Preprocess image if enabled
        if preprocess and self.settings.get("ocr.preprocessing.enabled", True):
            image = self.image_processor.preprocess_image(image)
        
        # Determine which engine to use
        if engine == "auto":
            engine = self._select_best_engine()
        
        if engine == "combined" and len(self.engines) > 1:
            return self._combined_ocr(image, preserve_layout)
        
        # Use the selected engine
        if engine in self.engines:
            try:
                result = self.engines[engine].extract_text(image, preserve_layout)
                if result:
                    return result
            except Exception as e:
                logger.error(f"Error with {engine} OCR: {e}")
                # Fall through to fallback
        else:
            logger.warning(f"Engine {engine} not available, using fallback")
        
        # Try fallback engines if primary engine fails
        return self._fallback_ocr(image, preserve_layout, excluded_engine=engine)
    
    def perform_batch_ocr(self, images: List[Image.Image], engine: str = "auto", 
                         preserve_layout: bool = True, preprocess: bool = False,
                         use_cache: bool = True, show_progress: bool = True) -> List[str]:
        """
        Perform OCR on multiple images with performance optimizations
        
        Args:
            images: List of PIL Images to process
            engine: OCR engine to use
            preserve_layout: Whether to preserve text layout
            preprocess: Whether to apply image preprocessing
            use_cache: Whether to use caching
            show_progress: Whether to show progress indicator
            
        Returns:
            List of extracted text strings
        """
        if not images:
            return []
            
        start_time = time.time()
        results = []
        
        # Initialize progress tracker
        if show_progress:
            progress = ProgressTracker(len(images), "Processing images")
        
        # Use parallel processing for large batches
        if len(images) > 1 and engine != "combined":
            try:
                results = self.performance_optimizer.parallel_ocr_processing(
                    images, self, engine=engine, preserve_layout=preserve_layout, 
                    preprocess=preprocess
                )
                
                if show_progress:
                    progress.current_item = len(images)
                    progress._print_progress()
                    
            except Exception as e:
                logger.warning(f"Parallel processing failed, falling back to sequential: {e}")
                results = self._sequential_batch_ocr(images, engine, preserve_layout, 
                                                   preprocess, use_cache, show_progress)
        else:
            # Sequential processing
            results = self._sequential_batch_ocr(images, engine, preserve_layout, 
                                               preprocess, use_cache, show_progress)
        
        processing_time = time.time() - start_time
        logger.info(f"Batch OCR completed: {len(images)} images in {processing_time:.2f}s")
        logger.info(f"Average time per image: {processing_time/len(images):.2f}s")
        
        # Memory cleanup
        self.performance_optimizer.memory_cleanup()
        
        return results
    
    def _sequential_batch_ocr(self, images: List[Image.Image], engine: str,
                             preserve_layout: bool, preprocess: bool, use_cache: bool,
                             show_progress: bool) -> List[str]:
        """Sequential batch OCR processing with caching"""
        results = []
        
        if show_progress:
            progress = ProgressTracker(len(images), "Processing images")
        
        for i, image in enumerate(images):
            try:
                # Generate cache key if using cache
                cache_key = None
                if use_cache:
                    # Create a simple cache key based on image properties and settings
                    import hashlib
                    img_data = f"{image.size}_{engine}_{preserve_layout}_{preprocess}"
                    cache_key = hashlib.md5(img_data.encode()).hexdigest()
                    
                    # Check cache first
                    cached_result = self.cache_manager.get_cached_result(cache_key)
                    if cached_result is not None:
                        results.append(cached_result)
                        if show_progress:
                            progress.update()
                        continue
                
                # Optimize image for processing
                optimized_image = self.performance_optimizer.optimize_image_for_ocr(image)
                
                # Apply preprocessing if requested
                if preprocess:
                    optimized_image = self.performance_optimizer.preprocess_image_fast(optimized_image)
                
                # Perform OCR
                text = self.perform_ocr(optimized_image, engine, preserve_layout, preprocess=False)
                results.append(text)
                
                # Cache result if using cache
                if use_cache and cache_key:
                    self.cache_manager.save_to_cache(cache_key, text)
                
                if show_progress:
                    progress.update()
                    
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                results.append(f"Error: {str(e)}")
                if show_progress:
                    progress.update()
        
        return results
    
    def process_pdf_optimized(self, pdf_path: str, engine: str = "auto",
                            preserve_layout: bool = True, preprocess: bool = False,
                            show_progress: bool = True) -> List[str]:
        """
        Process PDF with performance optimizations
        
        Args:
            pdf_path: Path to PDF file
            engine: OCR engine to use
            preserve_layout: Whether to preserve text layout
            preprocess: Whether to apply image preprocessing
            show_progress: Whether to show progress
            
        Returns:
            List of extracted text strings (one per page)
        """
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            images = []
            
            # Convert all pages to images first
            if show_progress:
                progress = ProgressTracker(len(doc), "Converting PDF pages")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convert page to image with optimal DPI
                dpi = 200  # Good balance between quality and processing speed
                mat = fitz.Matrix(dpi/72, dpi/72)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                image = Image.open(io.BytesIO(img_data))
                images.append(image)
                
                if show_progress:
                    progress.update()
            
            doc.close()
            
            # Process all images in batch
            return self.perform_batch_ocr(images, engine, preserve_layout, 
                                        preprocess, show_progress=show_progress)
            
        except ImportError:
            logger.error("PyMuPDF not installed. Install with: pip install PyMuPDF")
            return ["Error: PyMuPDF not installed"]
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return [f"Error: {str(e)}"]
    
    def _select_best_engine(self) -> str:
        """Select the best available engine based on availability and settings"""
        default_engine = self.settings.get("ocr.default_engine", "auto")
        
        # If default_engine is not "auto" and is available, use it
        if default_engine != "auto" and default_engine in self.engines:
            return default_engine
        
        # Otherwise, choose based on preference order: paddleocr > easyocr > tesseract
        if "paddleocr" in self.engines:
            return "paddleocr"
        elif "easyocr" in self.engines:
            return "easyocr"
        elif "tesseract" in self.engines:
            return "tesseract"
        
        # If no engines are available, log error and return tesseract as default
        logger.error("No OCR engines available")
        return "tesseract"  # This will likely fail but at least we tried
    
    def _combined_ocr(self, image: Image.Image, preserve_layout: bool) -> str:
        """
        Run OCR using multiple engines and combine results by selecting the best
        
        Args:
            image: PIL Image to process
            preserve_layout: Whether to preserve text layout
            
        Returns:
            Best OCR result
        """
        results = {}
        scores = {}
        
        # Run each available engine
        for name, engine in self.engines.items():
            try:
                result = engine.extract_text(image, preserve_layout)
                results[name] = result
                scores[name] = self._score_result(result)
            except Exception as e:
                logger.error(f"Error with {name} OCR: {e}")
        
        if not results:
            return "Error: No OCR engine produced results"
        
        # Return the result with the highest score
        best_engine = max(scores, key=scores.get)
        logger.info(f"Combined OCR selected {best_engine} result with score {scores[best_engine]}")
        return results[best_engine]
    
    def _fallback_ocr(self, image: Image.Image, preserve_layout: bool, excluded_engine: str = None) -> str:
        """
        Try fallback OCR engines when the primary engine fails
        
        Args:
            image: PIL Image to process
            preserve_layout: Whether to preserve text layout
            excluded_engine: Engine to exclude from fallbacks
            
        Returns:
            OCR result from fallback engine, or error message
        """
        for name, engine in self.engines.items():
            # Skip the excluded engine
            if name == excluded_engine:
                continue
                
            try:
                result = engine.extract_text(image, preserve_layout)
                if result:
                    logger.info(f"Fallback to {name} OCR successful")
                    return result
            except Exception as e:
                logger.error(f"Fallback {name} OCR failed: {e}")
        
        return "Error: All OCR engines failed"
    
    def _score_result(self, text: str) -> float:
        """
        Score OCR result quality based on heuristics
        
        Args:
            text: OCR result text
            
        Returns:
            Quality score (0-1)
        """
        if not text:
            return 0.0
            
        # Simple scoring based on text length and character ratio
        score = min(1.0, len(text) / 100)  # Basic length score
        
        # Penalize results with too many special characters
        text_len = len(text)
        if text_len > 0:
            alpha_ratio = sum(c.isalnum() or c.isspace() for c in text) / text_len
            score *= alpha_ratio
            
        return score
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return self.performance_optimizer.get_system_stats()
    
    def clear_cache(self):
        """Clear OCR result cache"""
        self.cache_manager.clear_cache()
    
    @property
    def enabled_engines(self) -> List[str]:
        """Get list of enabled OCR engines"""
        return list(self.engines.keys())


class BaseOCREngine:
    """Base class for OCR engines"""
    
    def __init__(self, settings: Settings):
        """Initialize base OCR engine with settings"""
        self.settings = settings
    
    def extract_text(self, image: Image.Image, preserve_layout: bool = True) -> str:
        """
        Extract text from image
        
        Args:
            image: PIL Image to process
            preserve_layout: Whether to preserve text layout
            
        Returns:
            Extracted text
        """
        raise NotImplementedError("Subclasses must implement extract_text")


class TesseractEngine(BaseOCREngine):
    """Tesseract OCR engine implementation"""
    
    def __init__(self, settings: Settings):
        """Initialize Tesseract engine"""
        super().__init__(settings)
        import pytesseract
        self.pytesseract = pytesseract
        
        # Set custom path from settings if provided
        custom_path = self.settings.get("ocr.engines.tesseract.cmd_path")
        if custom_path:
            self.pytesseract.pytesseract.tesseract_cmd = custom_path
    
    def extract_text(self, image: Image.Image, preserve_layout: bool = True) -> str:
        """Extract text using Tesseract OCR"""
        config = '--psm 1' if preserve_layout else '--psm 6'
        try:
            return self.pytesseract.image_to_string(image, config=config)
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return ""


class EasyOCREngine(BaseOCREngine):
    """EasyOCR engine implementation"""
    
    def __init__(self, settings: Settings):
        """Initialize EasyOCR engine"""
        super().__init__(settings)
        self.reader = None  # Lazy initialization to save memory
    
    def _get_reader(self):
        """Lazy initialization of EasyOCR reader"""
        if self.reader is None:
            import easyocr
            use_gpu = self.settings.get("ocr.engines.easyocr.gpu", False)
            self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        return self.reader
    
    def extract_text(self, image: Image.Image, preserve_layout: bool = True) -> str:
        """Extract text using EasyOCR"""
        try:
            reader = self._get_reader()
            result = reader.readtext(np.array(image))
            
            if preserve_layout:
                return self._format_with_layout(result)
            else:
                # Simple concatenation of detected text
                return ' '.join([item[1] for item in result])
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return ""
    
    def _format_with_layout(self, result) -> str:
        """Format EasyOCR results preserving approximate layout"""
        if not result:
            return ""
            
        # Sort by vertical position (top to bottom)
        result.sort(key=lambda x: x[0][0][1])  # Sort by y-coordinate of top-left point
        
        lines = []
        current_line = []
        last_y = -1
        line_height_threshold = 20  # Adjust based on image resolution
        
        for box, text, _ in result:
            top_y = min(p[1] for p in box)
            
            # Check if this is a new line
            if last_y >= 0 and abs(top_y - last_y) > line_height_threshold:
                # Sort words in current line by x-coordinate (left to right)
                current_line.sort(key=lambda x: x[0])
                lines.append(' '.join(word[1] for word in current_line))
                current_line = []
            
            # Add word to current line
            current_line.append(((box[0][0], top_y), text))
            last_y = top_y
        
        # Add the last line if it exists
        if current_line:
            current_line.sort(key=lambda x: x[0])
            lines.append(' '.join(word[1] for word in current_line))
        
        return '\n'.join(lines)


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR engine implementation"""
    
    def __init__(self, settings: Settings):
        """Initialize PaddleOCR engine"""
        super().__init__(settings)
        self.ocr = None  # Lazy initialization to save memory
    
    def _get_ocr(self):
        """Lazy initialization of PaddleOCR"""
        if self.ocr is None:
            try:
                from paddleocr import PaddleOCR
                use_gpu = self.settings.get("ocr.engines.paddleocr.gpu", False)
                self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)
            except ImportError:
                logger.error("PaddleOCR not installed")
                raise
        return self.ocr
    
    def extract_text(self, image: Image.Image, preserve_layout: bool = True) -> str:
        """Extract text using PaddleOCR"""
        try:
            ocr = self._get_ocr()
            result = ocr.ocr(np.array(image), cls=True)
            
            if not result or not result[0]:
                return ""
                
            if preserve_layout:
                return self._format_with_layout(result[0])
            else:
                # Simple concatenation of detected text
                return ' '.join([item[1][0] for item in result[0]])
        except Exception as e:
            logger.error(f"PaddleOCR error: {e}")
            return ""
    
    def _format_with_layout(self, result) -> str:
        """Format PaddleOCR results preserving approximate layout"""
        if not result:
            return ""
            
        # Sort by vertical position (top to bottom)
        sorted_result = sorted(result, key=lambda x: min(p[1] for p in x[0]))
        
        lines = []
        current_line = []
        last_y = -1
        line_height_threshold = 20  # Adjust based on image resolution
        
        for box, (text, _) in sorted_result:
            top_y = min(p[1] for p in box)
            
            # Check if this is a new line
            if last_y >= 0 and abs(top_y - last_y) > line_height_threshold:
                # Sort words in current line by x-coordinate (left to right)
                current_line.sort(key=lambda x: x[0])
                lines.append(' '.join(word[1] for word in current_line))
                current_line = []
            
            # Add word to current line
            current_line.append(((box[0][0], top_y), text))
            last_y = top_y
        
        # Add the last line if it exists
        if current_line:
            current_line.sort(key=lambda x: x[0])
            lines.append(' '.join(word[1] for word in current_line))
        
        return '\n'.join(lines)
