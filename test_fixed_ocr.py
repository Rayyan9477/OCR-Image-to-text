#!/usr/bin/env python3
"""
OCR Test Script - Tests the fixed OCR module with the test images
"""

import os
import sys
import time
from PIL import Image

def test_ocr_on_image(image_path):
    """Test OCR on a single image"""
    print(f"Testing OCR on image: {image_path}")
    
    try:
        # Load image
        image = Image.open(image_path)
        print(f"Image loaded: {image.size[0]}x{image.size[1]}")
        
        # Import the fixed OCR module
        from ocr_module_fixed import perform_ocr, check_dependencies
        
        # Check available engines
        deps = check_dependencies()
        print("\nAvailable OCR engines:")
        for engine, available in deps.items():
            print(f"  {engine}: {'Available' if available else 'Not available'}")
        
        # Test available engines
        engines_to_test = []
        if deps['paddleocr_available']:
            engines_to_test.append("paddle")
        if deps['easyocr_available']:
            engines_to_test.append("easy")
        if deps['tesseract_available']:
            engines_to_test.append("tesseract")
        if len(engines_to_test) >= 2:
            engines_to_test.append("auto")
        
        for engine in engines_to_test:
            print(f"\n=== Testing {engine} OCR engine ===")
            
            start_time = time.time()
            result = perform_ocr(image, engine=engine, preserve_layout=True)
            elapsed = time.time() - start_time
            
            print(f"OCR completed in {elapsed:.2f} seconds")
            print("Result:")
            print("-" * 40)
            print(result[:500] + "..." if len(result) > 500 else result)
            print("-" * 40)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point"""
    print("OCR Test Script")
    print("--------------")
    
    # Check for test images
    test_images = []
    
    if os.path.exists("simple_test_image.jpg"):
        test_images.append("simple_test_image.jpg")
    
    if os.path.exists("complex_test_image.jpg"):
        test_images.append("complex_test_image.jpg")
    
    if not test_images:
        print("Error: No test images found!")
        print("Run create_test_image.py first to create test images.")
        return 1
    
    # Test each image
    for image_path in test_images:
        test_ocr_on_image(image_path)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
