#!/usr/bin/env python3
"""
Simple OCR Test Script

This script provides a simplified test for the OCR functionality,
using a minimal set of dependencies and handling missing packages gracefully.
"""

import os
import sys
import time
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import pytesseract

# Set environment variables to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Create a test image with text if no image path is provided
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
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        # Fallback to default
        font = ImageFont.load_default()
    
    # Draw text
    draw.text((50, 100), text, fill='black', font=font)
    
    # Save the image
    test_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
    image.save(test_path)
    print(f"Test image saved to: {test_path}")
    return test_path

def test_ocr_direct(image_path):
    """Test OCR directly using pytesseract without extra dependencies"""
    print(f"Testing OCR on: {image_path}")
    
    try:
        # Load image
        image = Image.open(image_path)
        img_np = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Process with Tesseract
        start_time = time.time()
        text = pytesseract.image_to_string(thresh)
        elapsed_time = time.time() - start_time
        
        print("\n=== OCR Results ===")
        print(f"Time: {elapsed_time:.2f} seconds")
        print(f"Extracted {len(text)} characters")
        print("\nSample text (first 500 characters):")
        print("=" * 40)
        print(text[:500])
        print("=" * 40)
        
        return True
    except Exception as e:
        print(f"Error during OCR test: {str(e)}")
        return False

def try_advanced_ocr(image_path):
    """Try to use the advanced OCR module if available"""
    try:
        # Try to import our OCR module
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from ocr_module import perform_ocr, detect_tables, detect_image_quality
        from model_manager import initialize_models
        
        print("Successfully imported OCR modules")
        print("Initializing models...")
        initialize_models()
        
        print(f"Testing OCR on: {image_path}")
        image = Image.open(image_path)
        
        # Check image quality
        quality_issues = detect_image_quality(image)
        if quality_issues:
            print(f"Image quality issues: {', '.join(quality_issues)}")
        
        # Check for tables
        has_tables = detect_tables(image)
        print(f"Tables detected: {'Yes' if has_tables else 'No'}")
        
        # Try each OCR engine
        engines = ["paddle", "easy", "combined"]
        for engine in engines:
            try:
                print(f"\nTesting {engine.upper()} engine:")
                start_time = time.time()
                text = perform_ocr(image, engine=engine, preserve_layout=True)
                elapsed_time = time.time() - start_time
                
                print(f"Time: {elapsed_time:.2f} seconds")
                print(f"Extracted {len(text)} characters")
                print("\nSample text (first 500 characters):")
                print("=" * 40)
                print(text[:500])
                print("=" * 40)
            except Exception as e:
                print(f"Error with {engine} engine: {str(e)}")
        
        return True
    except ImportError as e:
        print(f"Advanced OCR module not available: {str(e)}")
        return False
    except Exception as e:
        print(f"Error in advanced OCR test: {str(e)}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_ocr.py <image_path>")
        return 1
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return 1
    
    # First try the advanced OCR
    print("\n=== Testing Advanced OCR Module ===")
    advanced_success = try_advanced_ocr(image_path)
    
    if not advanced_success:
        # Fall back to direct pytesseract
        print("\n=== Falling back to Direct Tesseract OCR ===")
        test_ocr_direct(image_path)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
