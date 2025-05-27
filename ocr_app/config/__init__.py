"""
Configuration Settings for OCR Application
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class Settings:
    """Configuration management for the OCR application"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or os.path.join(os.path.dirname(__file__), "config.json")
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        default_config = {
            "ocr": {
                "engines": {
                    "tesseract": {
                        "enabled": True,
                        "cmd_path": "tesseract"
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
                },
                "default_engine": "auto",
                "preserve_layout": True,
                "preprocessing": {
                    "enabled": True,
                    "enhance_contrast": True,
                    "remove_noise": True,
                    "correct_skew": True
                }
            },
            "models": {
                "download_path": "./models",
                "cache_enabled": True,
                "qa_model": "distilbert-base-cased-distilled-squad",
                "sentence_transformer": "all-MiniLM-L6-v2"
            },
            "ui": {
                "theme": "default",
                "max_file_size": "200MB",
                "supported_formats": ["jpg", "jpeg", "png", "pdf", "bmp", "tiff"]
            },
            "logging": {
                "level": "INFO",
                "file": "ocr_app.log"
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return self._merge_configs(default_config, config)
            except Exception as e:
                logger.warning(f"Could not load config file: {e}. Using defaults.")
                return default_config
        else:
            # Create default config file
            self.save_config(default_config)
            return default_config
    
    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Recursively merge user config with defaults"""
        merged = default.copy()
        for key, value in user.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    def save_config(self, config: Dict[str, Any] = None):
        """Save configuration to file"""
        config_to_save = config or self.config
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save config file: {e}")
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'ocr.engines.tesseract.enabled')"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
        self.save_config()
    
    @property
    def ocr_engines(self) -> List[str]:
        """Get list of enabled OCR engines"""
        engines = []
        for engine, config in self.config["ocr"]["engines"].items():
            if config.get("enabled", False):
                engines.append(engine)
        return engines
    
    @property
    def models_path(self) -> Path:
        """Get models directory path"""
        path = Path(self.config["models"]["download_path"])
        path.mkdir(parents=True, exist_ok=True)
        return path
