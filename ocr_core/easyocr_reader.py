from .base_reader import OCRReader
import numpy as np
import time
from multi_engine_ocr import OCRResult
import easyocr

class EasyOCRReader(OCRReader):
    """
    OCR reader using EasyOCR
    """
    def __init__(self, langs: list = ['en'], gpu: bool = False):
        self.reader = easyocr.Reader(langs, gpu=gpu)

    def read(self, image: np.ndarray) -> OCRResult:
        start_time = time.time()
        results = self.reader.readtext(image)
        texts, confidences, bboxes = [], [], []
        for bbox, text, conf in results:
            if conf > 0.3:
                texts.append(text)
                confidences.append(conf)
                bboxes.append(bbox)
        combined_text = ' '.join(texts)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        processing_time = time.time() - start_time
        return OCRResult(
            text=combined_text,
            confidence=avg_conf,
            engine='easyocr',
            processing_time=processing_time,
            bbox=bboxes
        )
