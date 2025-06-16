import os
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import numpy as np
import logging
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NanonetsOCR:
    def __init__(self):
        """Initialize the Nanonets OCR model"""
        try:
            # Set device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Load model and processor
            model_name = "nanonets/Nanonets-OCR-s"
            logger.info(f"Loading model: {model_name}")
            
            self.processor = self._load_processor(model_name)
            self.model = self._load_model(model_name)
            
            # Move model to device
            self.model.to(self.device)
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Nanonets OCR: {str(e)}")
            raise

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_processor(model_name):
        """Cache and load the processor"""
        return AutoProcessor.from_pretrained(model_name)

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_model(model_name):
        """Cache and load the model"""
        return AutoModelForVision2Seq.from_pretrained(model_name)

    def preprocess_image(self, image):
        """Preprocess image for OCR"""
        try:
            if image is None:
                raise ValueError("Input image is None")
                
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
                
            if not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Validate image size
            max_size = 4096  # Maximum dimension
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.warning(f"Image resized from {image.size} to {new_size}")
                
            return image
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            raise

    def extract_text(self, image):
        """
        Extract text from image using Nanonets OCR
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            dict: Contains extracted text and metadata
        """
        try:
            # Preprocess image
            image = self.preprocess_image(image)
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values=inputs.pixel_values,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode text
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            return {
                "text": generated_text,
                "metadata": {
                    "model": "nanonets/Nanonets-OCR-s",
                    "device": self.device,
                    "image_size": image.size
                }
            }
            
        except Exception as e:
            logger.error(f"Error in text extraction: {str(e)}")
            return {
                "text": "",
                "error": str(e),
                "metadata": {
                    "error_type": type(e).__name__,
                    "model": "nanonets/Nanonets-OCR-s"
                }
            }

    def batch_process(self, images):
        """
        Process multiple images in batch
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            list: List of extraction results
        """
        results = []
        for idx, image in enumerate(images):
            try:
                result = self.extract_text(image)
                result["metadata"]["batch_index"] = idx
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing image {idx} in batch: {str(e)}")
                results.append({
                    "text": "",
                    "error": str(e),
                    "metadata": {
                        "error_type": type(e).__name__,
                        "batch_index": idx,
                        "model": "nanonets/Nanonets-OCR-s"
                    }
                })
        return results 