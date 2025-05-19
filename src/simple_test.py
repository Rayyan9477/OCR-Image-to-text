# Simple OCR test script
import os
import sys
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_test_image():
    """Create a simple test image with text"""
    print("Creating test image...")
    # Create a white image
    width, height = 800, 300
    image = Image.new('RGB', (width, height), color='white')
    
    # Get a drawing context
    draw = ImageDraw.Draw(image)
    
    # Draw some sample text
    text = "OCR Test Image - The quick brown fox jumps over the lazy dog"
    try:
        # Try to use a system font
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        # Fallback to default
        font = ImageFont.load_default()
    
    # Draw text
    draw.text((50, 120), text, fill='black', font=font)
    
    # Save the image
    test_path = os.path.join(os.path.dirname(__file__), "simple_test_image.jpg")
    image.save(test_path)
    print(f"Test image saved to: {test_path}")
    return test_path, image

def test_ocr():
    """Test the OCR functionality"""
    # Set environment variables
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    
    # Add current directory to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Create test image
    test_path, image = create_test_image()
    
    # First test with pytesseract directly
    print("Testing with pytesseract directly first...")
    try:
        import pytesseract
        pytesseract_text = pytesseract.image_to_string(image)
        print(f"Pytesseract direct result ({len(pytesseract_text)} chars):")
        print("-" * 40)
        print(pytesseract_text)
        print("-" * 40)
        print("Pytesseract direct test successful!\n")
    except Exception as e:
        print(f"Pytesseract direct test error: {str(e)}\n")
    
    try:
        # Try to import our OCR module
        from ocr_module import perform_ocr
        from model_manager import initialize_models
        
        print("Initializing OCR models...")
        initialize_models()
        
        # Try OCR with each engine
        engines = ["paddle", "easy", "combined"]
        for engine in engines:
            try:
                print(f"\nTesting {engine.upper()} OCR engine:")
                start_time = time.time()
                text = perform_ocr(image, engine=engine)
                elapsed_time = time.time() - start_time
                
                print(f"OCR completed in {elapsed_time:.2f} seconds")
                print(f"Text extracted ({len(text)} characters):")
                print("-" * 40)
                print(text)
                print("-" * 40)
            except Exception as e:
                print(f"Error with {engine} OCR engine: {str(e)}")
    except Exception as e:
        print(f"Error in OCR test: {str(e)}")
        # Try pytesseract directly as fallback
        try:
            import pytesseract
            print("\nFalling back to pytesseract directly:")
            text = pytesseract.image_to_string(image)
            print(f"Text extracted ({len(text)} characters):")
            print("-" * 40)
            print(text)
            print("-" * 40)
        except Exception as e2:
            print(f"Pytesseract fallback also failed: {str(e2)}")

if __name__ == "__main__":
    test_ocr()
