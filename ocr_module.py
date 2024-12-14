import pytesseract
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import numpy as np

def perform_ocr(image):
    # Convert image to grayscale
    gray_image = ImageOps.grayscale(image)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(gray_image)
    enhanced_image = enhancer.enhance(2.0)
    
    # Convert image to numpy array
    image_array = np.array(enhanced_image)
    
    # Apply thresholding
    threshold_value = 128
    binary_array = (image_array > threshold_value) * 255
    binary_image = Image.fromarray(np.uint8(binary_array))
    
    # Perform OCR using Tesseract with layout analysis
    custom_config = r'--oem 3 --psm 1'
    extracted_text = pytesseract.image_to_string(binary_image, config=custom_config)
    
    return extracted_text.strip()