from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from PIL import Image
from multi_engine_ocr import OCRResult

class OCRReader(ABC):
    """
    Base class for OCR readers. Each reader should implement the read method.
    """
    @abstractmethod
    def read(self, image: Union[np.ndarray, Image.Image]) -> OCRResult:
        """Read text from image and return OCRResult"""
        pass
