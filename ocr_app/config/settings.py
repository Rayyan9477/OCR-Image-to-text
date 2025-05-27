"""
Configuration settings for the OCR application
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

class Settings:
    """
    Configuration settings manager for the OCR application
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize settings
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = Path(config_path) if config_path else self._get_default_config_path()
        self.models_path = self._get_models_path()
        self._config = self._load_config()
    
    def _get_default_config_path(self) -> Path:
        """Get the default configuration file path"""
        config_dir = Path(__file__).parent
        return config_dir / "config.json"
    
    def _get_models_path(self) -> Path:
        """Get the models directory path"""
        # Use the models directory in the project root
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / "models"
        os.makedirs(models_dir, exist_ok=True)
        return models_dir
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
            else:
                logger.info("Using default configuration")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "ocr": {
                "default_engine": "auto",
                "preprocessing": {
                    "enabled": True,
                    "enhance_contrast": True,
                    "remove_noise": True,
                    "correct_skew": True
                },
                "engines": {
                    "tesseract": {
                        "enabled": True,
                        "cmd_path": None
                    },
                    "easyocr": {
                        "enabled": True,
                        "gpu": False
                    },
                    "paddleocr": {
                        "enabled": True,
                        "use_gpu": False,
                        "use_angle_cls": True
                    }
                }
            },
            "models": {
                "qa_model": "distilbert-base-cased-distilled-squad",
                "sentence_transformer": "all-MiniLM-L6-v2"
            },
            "ui": {
                "theme": "light",
                "max_file_size": "200MB"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'ocr.preprocessing.enabled')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'ocr.preprocessing.enabled')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save(self) -> bool:
        """
        Save configuration to file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(self.config_path.parent, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return self._config.copy()

__all__ = ["Settings"]
