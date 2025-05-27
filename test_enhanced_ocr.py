#!/usr/bin/env python
"""
Simple Test for Enhanced Multi-Engine OCR System
"""

import os
import sys
import time
from PIL import Image

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_enhanced_ocr():
    """Test the enhanced multi-engine OCR system"""
    
    print("🔍 Testing Enhanced Multi-Engine OCR System")
    print("=" * 60)
    
    # Import the enhanced system
    try:
        from enhanced_multi_engine_ocr import extract_text_enhanced_multi_engine
        print("✅ Enhanced multi-engine OCR imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import enhanced OCR: {e}")
        return
    
    test_images = [
        "test_image.png",
        "simple_test_image.jpg", 
        "complex_test_image.jpg"
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n📷 Processing: {image_path}")
            print("-" * 40)
            
            try:
                # Load image
                image = Image.open(image_path)
                
                # Extract text using enhanced multi-engine system
                start_time = time.time()
                result = extract_text_enhanced_multi_engine(image)
                processing_time = time.time() - start_time
                
                print(f"⏱️  Total Processing Time: {processing_time:.2f}s")
                print(f"🏆 Best Engine: {result['best_result']['engine']}")
                print(f"📊 Best Confidence: {result['best_result']['confidence']:.2f}")
                print(f"📝 Best Text: {result['best_result']['text'][:80]}...")
                
                # Show individual engine results
                print("\n🔧 Engine Results:")
                for engine_name, engine_result in result['results'].items():
                    if engine_result['status'] == 'success':
                        print(f"  ✅ {engine_name}: {engine_result['processing_time']:.2f}s - Confidence: {engine_result['confidence']:.2f}")
                        if engine_result['text']:
                            print(f"      Text: {engine_result['text'][:60]}...")
                    else:
                        print(f"  ❌ {engine_name}: {engine_result['error']}")
                
                print(f"\n📈 Image Quality Score: {result['image_quality']['overall_score']:.2f}")
                print(f"✋ Handwriting Detected: {result['is_handwritten']}")
                
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
        else:
            print(f"⚠️  Image not found: {image_path}")
    
    print("\n" + "=" * 60)
    print("✅ Enhanced Multi-Engine OCR Test Complete!")

if __name__ == "__main__":
    test_enhanced_ocr()
