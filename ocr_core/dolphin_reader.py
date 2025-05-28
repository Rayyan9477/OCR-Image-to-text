from .base_reader import OCRReader
import numpy as np
from PIL import Image
import time
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from multi_engine_ocr import OCRResult

class DolphinOCRReader(OCRReader):
    """
    OCR reader using ByteDance/Dolphin model for high-accuracy text recognition
    """
    def __init__(self, model_name: str = 'ByteDance/Dolphin', device: str = None):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.device = device
        if device:
            self.model.to(device)

    def read(self, image: Union[np.ndarray, Image.Image]) -> OCRResult:
        start_time = time.time()
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            from cv2 import cvtColor, COLOR_BGR2RGB
            image = Image.fromarray(cvtColor(image, COLOR_BGR2RGB))

        # Generate tokens
        pixel_values = self.processor(images=image, return_tensors='pt').pixel_values
        if self.device:
            pixel_values = pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        processing_time = time.time() - start_time
        # Estimate confidence based on text length
        confidence = min(1.0, len(generated_text.strip()) / 100.0) if generated_text.strip() else 0.0
        return OCRResult(
            text=generated_text,
            confidence=confidence,
            engine='dolphin',
            processing_time=processing_time
        )
