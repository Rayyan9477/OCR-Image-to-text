from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")

def perform_ocr(image):
    # Convert image to RGB mode
    image = image.convert('RGB')
    
    # Resize image to a maximum size
    max_size = 1000
    ratio = max(image.size) / max_size
    new_size = tuple([int(x / ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

