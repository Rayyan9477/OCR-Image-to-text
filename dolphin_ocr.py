# filepath: c:/Users/rayyan.a/Downloads/Repo/OCR-Image-to-text/dolphin_ocr.py
"""
Dolphin OCR integration using HuggingFace VisionEncoderDecoderModel
"""
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import numpy as np

# Thread-safe model cache
_model = None
_processor = None
_tokenizer = None


def get_dolphin_model():
    global _model, _processor, _tokenizer
    if _model is None or _processor is None or _tokenizer is None:
        # Load Dolphin model and processor
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _model = VisionEncoderDecoderModel.from_pretrained('ByteDance/Dolphin')
        _processor = ViTImageProcessor.from_pretrained('ByteDance/Dolphin')
        _tokenizer = AutoTokenizer.from_pretrained('ByteDance/Dolphin')
        _model.to(device)
    return _model, _processor, _tokenizer


def dolphin_ocr(image: Image.Image) -> str:
    """
    Perform OCR using the Dolphin model.
    Args:
        image: PIL Image in RGB mode
    Returns:
        Parsed text as string
    """
    model, processor, tokenizer = get_dolphin_model()
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Preprocess image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device)
    # Generate output
    outputs = model.generate(pixel_values, max_length=1024)
    preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return preds[0]
