"""
Performance Enhancements for OCR Application
"""

import os
import logging
import gc
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
from typing import List, Dict, Any, Optional, Union
import time
import psutil
import threading
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Performance optimization utilities for OCR application"""
    
    def __init__(self, settings):
        self.settings = settings
        self.cpu_count = mp.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.gpu_available = torch.cuda.is_available()
        
        # Configure based on system resources
        self.max_workers = min(self.cpu_count, 8)  # Cap at 8 to avoid memory issues
        self.batch_size = self._calculate_optimal_batch_size()
        
        logger.info(f"Performance Optimizer initialized:")
        logger.info(f"  CPU cores: {self.cpu_count}")
        logger.info(f"  Memory: {self.memory_gb:.1f} GB")
        logger.info(f"  GPU available: {self.gpu_available}")
        logger.info(f"  Max workers: {self.max_workers}")
        logger.info(f"  Batch size: {self.batch_size}")
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory"""
        if self.memory_gb < 4:
            return 1
        elif self.memory_gb < 8:
            return 2
        elif self.memory_gb < 16:
            return 4
        else:
            return 8
    
    @lru_cache(maxsize=128)
    def get_optimal_image_size(self, width: int, height: int) -> tuple:
        """Get optimal image size for processing"""
        max_dimension = 2048
        
        if max(width, height) <= max_dimension:
            return width, height
        
        # Maintain aspect ratio while resizing
        if width > height:
            new_width = max_dimension
            new_height = int(height * max_dimension / width)
        else:
            new_height = max_dimension
            new_width = int(width * max_dimension / height)
        
        return new_width, new_height
    
    def optimize_image_for_ocr(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Optimize image for OCR processing"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Get optimal size
        width, height = image.size
        opt_width, opt_height = self.get_optimal_image_size(width, height)
        
        # Resize if needed
        if (opt_width, opt_height) != (width, height):
            image = image.resize((opt_width, opt_height), Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def preprocess_image_fast(self, image: Image.Image) -> Image.Image:
        """Fast image preprocessing for OCR"""
        # Convert to numpy array for faster processing
        img_array = np.array(image)
        
        # Fast grayscale conversion if needed
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Fast noise reduction
        img_array = cv2.medianBlur(img_array, 3)
        
        # Adaptive thresholding for better text detection
        img_array = cv2.adaptiveThreshold(
            img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return Image.fromarray(img_array)
    
    def batch_process_images(self, images: List[Image.Image], process_func, **kwargs) -> List[Any]:
        """Process multiple images in batches with multiprocessing"""
        results = []
        
        # Split into batches
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            
            if len(batch) == 1:
                # Single image, process directly
                result = process_func(batch[0], **kwargs)
                results.append(result)
            else:
                # Multiple images, use threading
                with ThreadPoolExecutor(max_workers=min(len(batch), self.max_workers)) as executor:
                    batch_results = list(executor.map(
                        lambda img: process_func(img, **kwargs), batch
                    ))
                    results.extend(batch_results)
            
            # Memory cleanup after each batch
            gc.collect()
        
        return results
    
    def parallel_ocr_processing(self, images: List[Image.Image], ocr_engine, **kwargs) -> List[str]:
        """Process multiple images with OCR in parallel"""
        start_time = time.time()
        
        def process_single_image(image):
            try:
                # Optimize image first
                optimized_image = self.optimize_image_for_ocr(image)
                
                # Apply preprocessing if requested
                if kwargs.get('preprocess', False):
                    optimized_image = self.preprocess_image_fast(optimized_image)
                
                # Perform OCR
                return ocr_engine.perform_ocr(optimized_image, **kwargs)
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                return f"Error: {str(e)}"
        
        # Process in batches
        results = self.batch_process_images(images, process_single_image)
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(images)} images in {processing_time:.2f} seconds")
        logger.info(f"Average time per image: {processing_time/len(images):.2f} seconds")
        
        return results
    
    def memory_cleanup(self):
        """Perform memory cleanup"""
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system performance statistics"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        stats = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'memory_used_gb': memory.used / (1024**3),
        }
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_reserved() / (1024**3)
            stats['gpu_memory_gb'] = gpu_memory
        
        return stats


class CacheManager:
    """Cache manager for OCR results and processed images"""
    
    def __init__(self, cache_dir: str = "cache", max_cache_size_gb: float = 1.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size = max_cache_size_gb * 1024**3  # Convert to bytes
        self._cache_lock = threading.Lock()
    
    def get_cache_key(self, image_path: str, settings: Dict[str, Any]) -> str:
        """Generate cache key for image and settings"""
        import hashlib
        
        # Include file modification time and settings in hash
        mtime = os.path.getmtime(image_path)
        cache_data = f"{image_path}_{mtime}_{str(sorted(settings.items()))}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def get_cached_result(self, cache_key: str) -> Optional[str]:
        """Get cached OCR result"""
        cache_file = self.cache_dir / f"{cache_key}.txt"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
        
        return None
    
    def save_to_cache(self, cache_key: str, result: str):
        """Save OCR result to cache"""
        with self._cache_lock:
            cache_file = self.cache_dir / f"{cache_key}.txt"
            
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(result)
                
                # Clean up cache if it's too large
                self._cleanup_cache()
                
            except Exception as e:
                logger.warning(f"Error saving to cache: {e}")
    
    def _cleanup_cache(self):
        """Clean up cache if it exceeds size limit"""
        try:
            total_size = sum(f.stat().st_size for f in self.cache_dir.glob('*.txt'))
            
            if total_size > self.max_cache_size:
                # Remove oldest files first
                files = [(f, f.stat().st_mtime) for f in self.cache_dir.glob('*.txt')]
                files.sort(key=lambda x: x[1])  # Sort by modification time
                
                removed_size = 0
                target_size = self.max_cache_size * 0.8  # Remove to 80% of limit
                
                for file_path, _ in files:
                    if total_size - removed_size <= target_size:
                        break
                    
                    removed_size += file_path.stat().st_size
                    file_path.unlink()
                
                logger.info(f"Cache cleanup: removed {removed_size / 1024**2:.1f} MB")
                
        except Exception as e:
            logger.warning(f"Error during cache cleanup: {e}")
    
    def clear_cache(self):
        """Clear all cached results"""
        try:
            for cache_file in self.cache_dir.glob('*.txt'):
                cache_file.unlink()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")


class ProgressTracker:
    """Progress tracking for long-running operations"""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.current_item = 0
        self.description = description
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, increment: int = 1):
        """Update progress"""
        self.current_item += increment
        
        # Update every second or at completion
        current_time = time.time()
        if current_time - self.last_update >= 1.0 or self.current_item >= self.total_items:
            self._print_progress()
            self.last_update = current_time
    
    def _print_progress(self):
        """Print current progress"""
        percentage = (self.current_item / self.total_items) * 100
        elapsed_time = time.time() - self.start_time
        
        if self.current_item > 0:
            time_per_item = elapsed_time / self.current_item
            eta = time_per_item * (self.total_items - self.current_item)
            eta_str = f" ETA: {eta:.1f}s"
        else:
            eta_str = ""
        
        print(f"\r{self.description}: {self.current_item}/{self.total_items} "
              f"({percentage:.1f}%){eta_str}", end='', flush=True)
        
        if self.current_item >= self.total_items:
            print()  # New line at completion


def apply_performance_optimizations(settings):
    """Apply system-wide performance optimizations"""
    
    # Set optimal number of threads for various libraries
    cpu_count = mp.cpu_count()
    
    # OpenCV optimizations
    cv2.setNumThreads(min(cpu_count, 4))
    
    # PyTorch optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Set thread counts for CPU-bound operations
    os.environ['OMP_NUM_THREADS'] = str(min(cpu_count, 4))
    os.environ['MKL_NUM_THREADS'] = str(min(cpu_count, 4))
    
    logger.info("Performance optimizations applied")


# Initialize performance components
def initialize_performance_components(settings):
    """Initialize all performance enhancement components"""
    apply_performance_optimizations(settings)
    
    optimizer = PerformanceOptimizer(settings)
    cache_manager = CacheManager()
    
    return optimizer, cache_manager
