#!/usr/bin/env python3
"""
Simple test of OCR functionality
"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ocr_app.core.ocr_engine import OCREngine
from ocr_app.config.settings import Settings

def create_test_image():
    """Create a simple test image with text"""
    # Create white background
    image = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(image)
    
    # Add text
    text = "Hello OCR Test!"
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    draw.text((50, 80), text, fill='black', font=font)
    
    # Save the test image
    image.save('test_image.png')
    print("Test image created: test_image.png")
    return image

def test_ocr():
    """Test OCR functionality"""
    print("Testing OCR functionality...")
    
    # Create test image
    test_image = create_test_image()
    
    # Initialize OCR
    settings = Settings()
    ocr_engine = OCREngine(settings)
    
    print(f"Available engines: {ocr_engine.enabled_engines}")
    
    # Test OCR with auto engine selection
    print("\nPerforming OCR...")
    text_result = ocr_engine.perform_ocr(test_image, engine="auto")
    
    print(f"OCR Result: '{text_result}'")
    
    # Test batch processing with single image
    print("\nTesting batch processing...")
    batch_results = ocr_engine.perform_batch_ocr([test_image], engine="auto")
    
    print(f"Batch Result: '{batch_results[0]}'")
    
    # Test performance stats
    stats = ocr_engine.get_performance_stats()
    print(f"\nPerformance Stats:")
    print(f"  CPU: {stats['cpu_percent']:.1f}%")
    print(f"  Memory: {stats['memory_percent']:.1f}%")
    
    print("\n✅ OCR test completed successfully!")

if __name__ == "__main__":
    test_ocr()
