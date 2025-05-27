#!/usr/bin/env python
"""
Test script for Multi-Engine OCR System
"""

import os
import sys
import time
from PIL import Image
import numpy as np

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from enhanced_multi_engine_ocr import extract_text_enhanced_multi_engine

def test_multi_engine_ocr():
    """Test the multi-engine OCR system with available test images"""
    
    test_images = [
        "test_image.png",
        "simple_test_image.jpg", 
        "complex_test_image.jpg"
    ]
    
    print("🔍 Testing Multi-Engine OCR System")
    print("=" * 50)
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n📷 Processing: {image_path}")
            print("-" * 30)
            
            try:
                # Load image
                image = Image.open(image_path)
                  # Extract text using multi-engine system
                start_time = time.time()
                result = extract_text_enhanced_multi_engine(image)
                processing_time = time.time() - start_time
                
                print(f"⏱️  Total Processing Time: {processing_time:.2f}s")
                print(f"🏆 Best Result: {result['best_result']['engine']}")
                print(f"📊 Confidence: {result['best_result']['confidence']:.2f}")
                print(f"📝 Text Preview: {result['best_result']['text'][:100]}...")
                
                # Show individual engine results
                print("\n🔧 Engine Results:")
                for engine_name, engine_result in result['results'].items():
                    if engine_result['status'] == 'success':
                        print(f"  ✅ {engine_name}: {engine_result['processing_time']:.2f}s - Confidence: {engine_result['confidence']:.2f}")
                    else:
                        print(f"  ❌ {engine_name}: {engine_result['error']}")
                
                print(f"\n📈 Image Quality Score: {result['image_quality']['overall_score']:.2f}")
                
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
        else:
            print(f"⚠️  Image not found: {image_path}")
    
    print("\n" + "=" * 50)
    print("✅ Multi-Engine OCR Test Complete!")

if __name__ == "__main__":
    test_multi_engine_ocr()
