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
    # Try importing tf_keras first
    try:
        import tf_keras
        logger.info(f"tf-keras available, version: {tf_keras.__version__}")
        OPTIONAL_MODULES['tf_keras_installed'] = True
    except ImportError:
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

# Rest of your model_manager.py code follows...
