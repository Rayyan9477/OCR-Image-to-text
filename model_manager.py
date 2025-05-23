# model_manager.py
import os
import json
import torch
import importlib
import pickle
import logging
from threading import Lock
import warnings
import sys
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set environment variables for TF-Keras compatibility 
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Optional modules tracking
OPTIONAL_MODULES = {
    'tf_keras_installed': False,
    'transformers_available': False,
    'sentence_transformers_available': False,
    'paddleocr_available': False,
    'easyocr_available': False
}

# Check if tensorflow/keras is available but don't try to install packages
try:
    # Use importlib to check for module availability first
    tf_keras_spec = importlib.util.find_spec('tf_keras')
    if tf_keras_spec:
        # Only try to import if the module is actually available
        tf_keras = importlib.import_module('tf_keras')
        logger.info(f"tf-keras available, version: {tf_keras.__version__}")
        OPTIONAL_MODULES['tf_keras_installed'] = True
    else:
        # Fall back to regular keras
        import keras
        keras_version = keras.__version__
        if keras_version.startswith('3.'):
            logger.info("Detected Keras 3 - will use limited transformer features")
            # Set environment variable to prevent iterator_model_ops error
            os.environ['TF_KERAS'] = '1'
            try:
                import tensorflow as tf
                logger.info(f"TensorFlow version: {tf.__version__}")
                OPTIONAL_MODULES['tf_keras_installed'] = True
            except ImportError:
                logger.warning("TensorFlow import failed with Keras 3")
                OPTIONAL_MODULES['tf_keras_installed'] = False
        else:
            # Non-Keras 3 version, should work fine
            OPTIONAL_MODULES['tf_keras_installed'] = True
except ImportError:
    logger.warning("Keras not available - QA features will be limited")

# Global model cache
_model_cache = {}
_model_lock = Lock()

# Default OCR config
_default_ocr_config = {
    "preprocessing": {
        "default_contrast": 1.5,
        "denoise": True,
        "adaptive_threshold": True
    },
    "engines": {
        "paddle": {
            "use_angle_cls": True,
            "lang": "en"
        },
        "easy": {
            "langs": ["en"],
            "gpu": False
        }
    }
}

# Load or create config file
def _load_or_create_config():
    """Load or create OCR config file"""
    config_path = os.path.join(os.path.dirname(__file__), "models", "ocr_config.json")
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info("OCR config loaded")
            return config
        else:
            # Create directory if needed
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Create default config
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(_default_ocr_config, f, indent=2)
            logger.info("Default OCR config created")
            return _default_ocr_config
    except Exception as e:
        logger.error(f"Error loading/creating OCR config: {e}")
        return _default_ocr_config

# Initialize OCR config
_ocr_config = _load_or_create_config()

def get_ocr_config():
    """Get current OCR configuration"""
    return _ocr_config

def update_ocr_config(new_config):
    """Update OCR configuration"""
    global _ocr_config
    
    # Merge new config with existing
    for key, value in new_config.items():
        if key in _ocr_config:
            if isinstance(_ocr_config[key], dict) and isinstance(value, dict):
                _ocr_config[key].update(value)
            else:
                _ocr_config[key] = value
        else:
            _ocr_config[key] = value
    
    # Save to file
    config_path = os.path.join(os.path.dirname(__file__), "models", "ocr_config.json")
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(_ocr_config, f, indent=2)
        logger.info("OCR config updated")
    except Exception as e:
        logger.error(f"Error saving OCR config: {e}")
    
    return _ocr_config

def initialize_models():
    """Initialize and cache models needed for OCR and QA"""
    # Check for PaddleOCR
    try:
        importlib.util.find_spec("paddleocr")
        OPTIONAL_MODULES['paddleocr_available'] = True
        logger.info("PaddleOCR module is available")
    except ImportError:
        OPTIONAL_MODULES['paddleocr_available'] = False
        logger.warning("PaddleOCR not available. Consider installing with: pip install paddleocr")
    
    # Check for EasyOCR
    try:
        importlib.util.find_spec("easyocr")
        OPTIONAL_MODULES['easyocr_available'] = True
        logger.info("EasyOCR module is available")
    except ImportError:
        OPTIONAL_MODULES['easyocr_available'] = False
        logger.warning("EasyOCR not available. Consider installing with: pip install easyocr")
    
    # Check for transformers and sentence transformers (for QA functionality)
    try:
        importlib.util.find_spec("transformers")
        OPTIONAL_MODULES['transformers_available'] = True
        
        try:
            importlib.util.find_spec("sentence_transformers")
            OPTIONAL_MODULES['sentence_transformers_available'] = True
            logger.info("Sentence Transformers module is available for advanced search")
            
            # Initialize sentence transformer model if needed for advanced search
            if 'sentence_transformer' not in _model_cache:
                try:
                    from sentence_transformers import SentenceTransformer
                    model_path = os.path.join(os.path.dirname(__file__), "models", "sentence_transformer")
                    
                    # Use local model if available, otherwise download from HuggingFace
                    if os.path.exists(model_path):
                        with _model_lock:
                            _model_cache['sentence_transformer'] = SentenceTransformer(model_path)
                    else:
                        logger.info("Downloading sentence transformer model (this may take a while)...")
                        with _model_lock:
                            _model_cache['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    logger.info("Sentence transformer model initialized")
                except Exception as e:
                    logger.error(f"Error loading sentence transformer model: {e}")
        except ImportError:
            OPTIONAL_MODULES['sentence_transformers_available'] = False
            logger.warning("Sentence Transformers not available. Some search features will be limited.")
    
    except ImportError:
        OPTIONAL_MODULES['transformers_available'] = False
        logger.warning("HuggingFace Transformers not available. QA features will be limited.")
    
    # Initialize QA model if transformers is available
    if OPTIONAL_MODULES['transformers_available'] and OPTIONAL_MODULES['tf_keras_installed']:
        if 'qa_model' not in _model_cache:
            try:
                from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
                
                model_path = os.path.join(os.path.dirname(__file__), "models", "qa_model")
                
                # Use local model if available, otherwise use default from HuggingFace
                if os.path.exists(model_path):
                    logger.info("Loading QA model from local path")
                    try:
                        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        with _model_lock:
                            _model_cache['qa_model'] = pipeline("question-answering", model=model, tokenizer=tokenizer)
                    except Exception as local_err:
                        logger.error(f"Failed to load local QA model: {local_err}")
                        logger.info("Falling back to default model")
                        with _model_lock:
                            _model_cache['qa_model'] = pipeline("question-answering")
                else:
                    logger.info("Loading default QA model from HuggingFace")
                    with _model_lock:
                        _model_cache['qa_model'] = pipeline("question-answering")
                
                logger.info("QA model initialized")
            
            except Exception as e:
                logger.error(f"Failed to initialize QA model: {e}")
                _model_cache['qa_model'] = None
    
    logger.info("Model initialization completed")
    return OPTIONAL_MODULES

def get_paddle_ocr():
    """Get or initialize PaddleOCR model"""
    if not OPTIONAL_MODULES['paddleocr_available']:
        logger.error("PaddleOCR is not available")
        return None
    
    try:
        if 'paddle_ocr' not in _model_cache:
            from paddleocr import PaddleOCR
            
            # Get config settings
            config = get_ocr_config()
            paddle_config = config.get("engines", {}).get("paddle", {})
            
            # Create PaddleOCR instance with appropriate settings
            use_angle_cls = paddle_config.get("use_angle_cls", True)
            lang = paddle_config.get("lang", "en")
            
            logger.info(f"Initializing PaddleOCR (lang={lang}, use_angle_cls={use_angle_cls})")
            
            with _model_lock:
                _model_cache['paddle_ocr'] = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, show_log=False)
            
            logger.info("PaddleOCR initialized")
        
        return _model_cache['paddle_ocr']
    
    except Exception as e:
        logger.error(f"Error initializing PaddleOCR: {e}")
        return None

def get_easy_ocr():
    """Get or initialize EasyOCR model"""
    if not OPTIONAL_MODULES['easyocr_available']:
        logger.error("EasyOCR is not available")
        return None
    
    try:
        if 'easy_ocr' not in _model_cache:
            import easyocr
            
            # Get config settings
            config = get_ocr_config()
            easy_config = config.get("engines", {}).get("easy", {})
            
            # Create Reader with appropriate settings
            languages = easy_config.get("langs", ["en"])
            gpu = easy_config.get("gpu", False)
            
            logger.info(f"Initializing EasyOCR (langs={languages}, gpu={gpu})")
            
            with _model_lock:
                _model_cache['easy_ocr'] = easyocr.Reader(languages, gpu=gpu)
            
            logger.info("EasyOCR initialized")
        
        return _model_cache['easy_ocr']
    
    except Exception as e:
        logger.error(f"Error initializing EasyOCR: {e}")
        return None

def get_qa_model():
    """Get QA model if available"""
    if 'qa_model' in _model_cache:
        return _model_cache['qa_model']
    else:
        # Try initializing if not already tried
        if OPTIONAL_MODULES['transformers_available'] and OPTIONAL_MODULES['tf_keras_installed']:
            initialize_models()
            return _model_cache.get('qa_model')
    return None

def get_sentence_transformer():
    """Get sentence transformer model if available"""
    if 'sentence_transformer' in _model_cache:
        return _model_cache['sentence_transformer']
    else:
        # Try initializing if not already tried
        if OPTIONAL_MODULES['sentence_transformers_available']:
            initialize_models()
            return _model_cache.get('sentence_transformer')
    return None