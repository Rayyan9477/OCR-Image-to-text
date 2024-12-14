import pytesseract
from PIL import Image, ImageFilter, ImageOps

def perform_ocr(image):
    # Convert image to grayscale
    gray_image = ImageOps.grayscale(image)
    
    # Apply a median filter to reduce noise
    denoised_image = gray_image.filter(ImageFilter.MedianFilter(size=3))
    
    # Apply thresholding
    threshold_image = denoised_image.point(lambda p: p > 128 and 255)
    
    # Perform OCR using Tesseract
    extracted_text = pytesseract.image_to_string(threshold_image)
    
    return extracted_text.strip()