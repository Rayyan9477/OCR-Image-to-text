import os
from typing import Dict, Any

class ModelManager:
    def __init__(self):
        self.OPTIONAL_MODULES = {
            'paddleocr_available': False,
            'easyocr_available': False,
            'tesseract_available': False
        }
        self._check_available_modules()
        
    def _check_available_modules(self):
        """Check which OCR modules are available in the environment"""
        try:
            import paddleocr
            self.OPTIONAL_MODULES['paddleocr_available'] = True
        except ImportError:
            pass
            
        try:
            import easyocr
            self.OPTIONAL_MODULES['easyocr_available'] = True
        except ImportError:
            pass
            
        try:
            import pytesseract
            self.OPTIONAL_MODULES['tesseract_available'] = True
        except ImportError:
            pass
            
    def get_available_engines(self) -> Dict[str, bool]:
        """Get dictionary of available OCR engines"""
        return {
            'PaddleOCR': self.OPTIONAL_MODULES['paddleocr_available'],
            'EasyOCR': self.OPTIONAL_MODULES['easyocr_available'],
            'Tesseract': self.OPTIONAL_MODULES['tesseract_available']
        }
        
    def get_recommended_engine(self) -> str:
        """Get the recommended OCR engine based on availability"""
        if self.OPTIONAL_MODULES['paddleocr_available']:
            return 'paddle'
        elif self.OPTIONAL_MODULES['easyocr_available']:
            return 'easy'
        elif self.OPTIONAL_MODULES['tesseract_available']:
            return 'tesseract'
        return 'none'
        
    def get_engine_description(self, engine: str) -> str:
        """Get description of OCR engine capabilities"""
        descriptions = {
            'paddle': "Fast and accurate, optimized for Asian languages but works well for English.",
            'easy': "Good general-purpose OCR with support for 80+ languages.",
            'tesseract': "Open-source OCR engine with good accuracy for printed text.",
            'combined': "Uses multiple engines and selects the best result (recommended but slower)."
        }
        return descriptions.get(engine, "Unknown engine") 