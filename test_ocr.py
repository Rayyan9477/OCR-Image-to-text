#!/usr/bin/env python3
"""
Simple OCR Test Script

This script provides a simplified test for the OCR functionality,
using a minimal set of dependencies and handling missing packages gracefully.
"""

import os
import sys
import time
import platform
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Set environment variables to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'

def create_test_image():
    """Create a simple test image with text for OCR testing"""
    try:
        width, height = 800, 400
        background_color = (255, 255, 255)
        text_color = (0, 0, 0)
        
        # Create a blank image
        image = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(image)
        
        # Try to get a font
        try:
            # Try to use a common font
            if platform.system() == "Windows":
                font_path = "C:\\Windows\\Fonts\\Arial.ttf"
            elif platform.system() == "Darwin":  # macOS
                font_path = "/System/Library/Fonts/Helvetica.ttc"
            else:  # Linux and others
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 32)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Add text to the image
        text1 = "OCR Test Image"
        text2 = "This is a sample image for testing OCR functionality."
        text3 = "It contains different text sizes and formatting."
        text4 = "1234567890 !@#$%^&*()"
        
        draw.text((width/2-150, 50), text1, fill=text_color, font=font)
        draw.text((50, 150), text2, fill=text_color, font=font)
        draw.text((50, 200), text3, fill=text_color, font=font)
        draw.text((50, 250), text4, fill=text_color, font=font)
        
        # Draw some lines and a rectangle
        draw.line([(50, 300), (750, 300)], fill=(0, 0, 0), width=3)
        draw.rectangle([(50, 320), (750, 380)], outline=(0, 0, 0), width=2)
        draw.text((400-100, 340), "Text in a box", fill=text_color, font=font)
        
        # Save the image
        image.save('simple_test_image.jpg', quality=95)
        print(f"Created test image: simple_test_image.jpg")
        return Image.open('simple_test_image.jpg')
    except Exception as e:
        print(f"Error creating test image: {str(e)}")
        return None

def check_ocr_engine(name):
    """Check if an OCR engine is available"""
    import importlib.util
    
    if name == "paddle":
        return importlib.util.find_spec("paddleocr") is not None
    elif name == "easy":
        return importlib.util.find_spec("easyocr") is not None
    elif name == "tesseract":
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except:
            return False
    return False

def test_ocr():
    """Test the OCR functionality with a simple image"""
    print("OCR System Test")
    print("-" * 40)
    
    # Get or create test image
    if os.path.exists('simple_test_image.jpg'):
        print("Using existing test image: simple_test_image.jpg")
        image = Image.open('simple_test_image.jpg')
    else:
        print("Test image not found, creating a new one...")
        image = create_test_image()
        if image is None:
            print("Failed to create test image. Exiting.")
            return

    # First try the fixed OCR module
    print("Attempting to use fixed OCR module...")
    try:
        from ocr_module_fixed import perform_ocr
        ocr_function = perform_ocr
        print("Using fixed OCR module")
    except ImportError:
        # Try the main OCR module
        try:
            from ocr_module import perform_ocr
            ocr_function = perform_ocr
            print("Using main OCR module")
        except ImportError:
            # Fall back to lite version
            try:
                from ocr_module_lite import LiteOCR
                lite_ocr = LiteOCR()
                ocr_function = lite_ocr.perform_ocr
                print("Using lightweight OCR module")
            except ImportError:
                # Fall back to fallback version
                try:
                    from ocr_fallback import FallbackOCR
                    fallback_ocr = FallbackOCR()
                    ocr_function = fallback_ocr.extract_text
                    print("Using fallback OCR module")
                except ImportError:
                    print("No OCR module available!")
                    return
        
        # Check which engines are available
        available_engines = []
        if check_ocr_engine("paddle"):
            available_engines.append("paddle")
        if check_ocr_engine("easy"):
            available_engines.append("easy")
        if len(available_engines) >= 2:
            available_engines.append("combined")
        if not available_engines:
            if check_ocr_engine("tesseract"):
                available_engines.append("tesseract")
            else:
                print("No OCR engines available! Please install at least one OCR engine.")
                return
        
        # Test each available OCR engine
        for engine in available_engines:
            print(f"\nTesting OCR using engine: {engine}")
            print("-" * 40)
            
            try:
                start_time = time.time()
                if engine == "tesseract":
                    # Special handling for tesseract-only fallback
                    import pytesseract
                    result = pytesseract.image_to_string(image)
                else:
                    result = ocr_function(image, engine=engine, preserve_layout=True)
                elapsed = time.time() - start_time
                
                print(f"OCR completed in {elapsed:.2f} seconds")
                print("Result:")
                print("-" * 40)
                print(result[:500] + "..." if len(result) > 500 else result)
                print("-" * 40)
            except Exception as e:
                print(f"Error with {engine} OCR: {str(e)}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ocr()