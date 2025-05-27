"""
Model Manager Module

Handles loading, caching, and management of ML models for the OCR application.
"""

import os
import pickle
import json
import logging
import importlib
from threading import Lock
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import warnings

from ..config.settings import Settings

# Configure logging
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set environment variables for TF-Keras compatibility 
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'

class ModelManager:
    """
    Manages ML models for the OCR application with efficient loading and caching
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the model manager
        
        Args:
            settings: Application settings
        """
        self.settings = settings or Settings()
        self.models = {}
        self.model_locks = {}
        self.check_modules()
    
    def check_modules(self) -> Dict[str, bool]:
        """
        Check which optional modules are available
        
        Returns:
            Dictionary of module availability
        """
        self.module_status = {
            'tf_keras_installed': False,
            'transformers_available': False,
            'sentence_transformers_available': False,
            'pytorch_available': False
        }
        
        # Check for TensorFlow/Keras
        try:
            # Try importing tf_keras first
            try:
                import tf_keras
                logger.info(f"tf-keras available, version: {tf_keras.__version__}")
                self.module_status['tf_keras_installed'] = True
            except ImportError:
                # Fall back to regular keras
                import keras
                keras_version = keras.__version__
                if keras_version.startswith('3.'):
                    logger.info("Detected Keras 3 - using limited transformer features")
                    # Set environment variable to prevent iterator_model_ops error
                    os.environ['TF_KERAS'] = '1'
                    try:
                        import tensorflow as tf
                        logger.info(f"TensorFlow version: {tf.__version__}")
                        self.module_status['tf_keras_installed'] = True
                    except ImportError:
                        logger.warning("TensorFlow import failed with Keras 3")
                else:
                    # Non-Keras 3 version
                    self.module_status['tf_keras_installed'] = True
        except ImportError:
            logger.warning("Keras not available - QA features will be limited")
        
        # Check for PyTorch
        try:
            import torch
            logger.info(f"PyTorch available, version: {torch.__version__}")
            self.module_status['pytorch_available'] = True
        except ImportError:
            logger.warning("PyTorch not available - some features will be limited")
        
        # Check for Transformers
        try:
            import transformers
            logger.info(f"Transformers available, version: {transformers.__version__}")
            self.module_status['transformers_available'] = True
        except ImportError:
            logger.warning("Transformers not available - QA features will be limited")
        
        # Check for SentenceTransformers
        try:
            import sentence_transformers
            logger.info(f"SentenceTransformers available, version: {sentence_transformers.__version__}")
            self.module_status['sentence_transformers_available'] = True
        except ImportError:
            logger.warning("SentenceTransformers not available - QA features will be limited")
            
        return self.module_status
    
    def get_qa_model(self):
        """
        Get Question Answering model for RAG functionality
        
        Returns:
            QA pipeline or None if not available
        """
        model_name = self.settings.get("models.qa_model", "distilbert-base-cased-distilled-squad")
        
        # Return cached model if available
        if "qa_model" in self.models:
            return self.models["qa_model"]
        
        # Create lock if doesn't exist
        if "qa_model" not in self.model_locks:
            self.model_locks["qa_model"] = Lock()
        
        # Load model with lock to prevent concurrent loading
        with self.model_locks["qa_model"]:
            # Check again in case another thread loaded while we were waiting
            if "qa_model" in self.models:
                return self.models["qa_model"]
            
            # Check if transformers is available
            if not self.module_status['transformers_available']:
                logger.error("Transformers library not available for QA model")
                return None
            
            try:
                from transformers import pipeline
                logger.info(f"Loading QA model: {model_name}")
                
                # Get cache dir from settings
                cache_dir = self.settings.models_path / "qa_model"
                os.makedirs(cache_dir, exist_ok=True)
                
                # Load the model
                qa_pipeline = pipeline("question-answering", model=model_name, cache_dir=str(cache_dir))
                self.models["qa_model"] = qa_pipeline
                return qa_pipeline
                
            except Exception as e:
                logger.error(f"Error loading QA model: {e}")
                return None
    
    def get_sentence_transformer(self):
        """
        Get SentenceTransformer model for text embedding
        
        Returns:
            SentenceTransformer model or None if not available
        """
        model_name = self.settings.get("models.sentence_transformer", "all-MiniLM-L6-v2")
        
        # Return cached model if available
        if "sentence_transformer" in self.models:
            return self.models["sentence_transformer"]
        
        # Create lock if doesn't exist
        if "sentence_transformer" not in self.model_locks:
            self.model_locks["sentence_transformer"] = Lock()
        
        # Load model with lock to prevent concurrent loading
        with self.model_locks["sentence_transformer"]:
            # Check again in case another thread loaded while we were waiting
            if "sentence_transformer" in self.models:
                return self.models["sentence_transformer"]
            
            # Check if sentence_transformers is available
            if not self.module_status['sentence_transformers_available']:
                logger.error("SentenceTransformers library not available")
                return None
            
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model: {model_name}")
                
                # Get cache dir from settings
                cache_dir = self.settings.models_path / "sentence_transformer"
                os.makedirs(cache_dir, exist_ok=True)
                
                # Load the model
                model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
                self.models["sentence_transformer"] = model
                return model
                
            except Exception as e:
                logger.error(f"Error loading SentenceTransformer model: {e}")
                return None
    
    def save_embeddings(self, data: Dict[str, Any], file_path: str) -> bool:
        """
        Save text embeddings to disk
        
        Args:
            data: Dictionary with embedding data
            file_path: Path to save the embeddings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved embeddings to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            return False
    
    def load_embeddings(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load text embeddings from disk
        
        Args:
            file_path: Path to load the embeddings from
            
        Returns:
            Dictionary with embedding data or None if failed
        """
        try:
            if not os.path.exists(file_path):
                return None
                
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded embeddings from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return None
    
    def get_module_status(self) -> Dict[str, bool]:
        """
        Get status of optional modules
        
        Returns:
            Dictionary of module availability
        """
        return self.module_status
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model to free memory
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            True if successful, False otherwise
        """
        if model_name in self.models:
            try:
                del self.models[model_name]
                import gc
                gc.collect()
                logger.info(f"Unloaded model: {model_name}")
                return True
            except Exception as e:
                logger.error(f"Error unloading model {model_name}: {e}")
                return False
        return False
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about loaded models
        
        Returns:
            Dictionary with model information
        """
        info = {}
        
        for model_name in self.models:
            model = self.models[model_name]
            
            if model_name == "qa_model":
                info[model_name] = {
                    "type": "QuestionAnswering",
                    "loaded": True,
                    "model_name": self.settings.get("models.qa_model", "Unknown")
                }
            elif model_name == "sentence_transformer":
                info[model_name] = {
                    "type": "SentenceTransformer",
                    "loaded": True,
                    "model_name": self.settings.get("models.sentence_transformer", "Unknown")
                }
            else:
                info[model_name] = {
                    "type": "Unknown",
                    "loaded": True
                }
        
        return info
