#!/usr/bin/env python
"""
Test Precision Layout OCR System
Demonstrates the enhanced layout preservation capabilities
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import time

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from precision_layout_ocr import extract_text_with_precision_layout
    PRECISION_AVAILABLE = True
    print("✅ Precision Layout OCR system loaded successfully")
except ImportError as e:
    PRECISION_AVAILABLE = False
    print(f"❌ Precision Layout OCR not available: {e}")

try:
    from enhanced_multi_engine_ocr import extract_text_enhanced_multi_engine
    ENHANCED_OCR_AVAILABLE = True
    print("✅ Enhanced Multi-Engine OCR system loaded successfully")
except ImportError as e:
    ENHANCED_OCR_AVAILABLE = False
    print(f"❌ Enhanced Multi-Engine OCR not available: {e}")

def create_test_document():
    """Create a test document with complex layout"""
    # Create image with white background
    width, height = 800, 600
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to load a font, fallback to default if not available
        title_font = ImageFont.truetype("arial.ttf", 24)
        header_font = ImageFont.truetype("arial.ttf", 18)
        body_font = ImageFont.truetype("arial.ttf", 14)
    except:
        # Fallback to default font
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    
    # Document title
    title_text = "PRECISION LAYOUT TEST DOCUMENT"
    draw.text((50, 30), title_text, fill='black', font=title_font)
    
    # Two-column layout
    col1_x = 50
    col2_x = 420
    current_y = 80
    
    # Column 1 content
    draw.text((col1_x, current_y), "Features:", fill='black', font=header_font)
    current_y += 30
    
    features = [
        "• Multi-engine OCR integration",
        "• Precision layout preservation",
        "• Column detection",
        "• Text structure analysis",
        "• Format conversion"
    ]
    
    for feature in features:
        draw.text((col1_x, current_y), feature, fill='black', font=body_font)
        current_y += 25
    
    # Column 2 content
    current_y = 110  # Reset for second column
    draw.text((col2_x, current_y), "Capabilities:", fill='black', font=header_font)
    current_y += 30
    
    capabilities = [
        "1. EasyOCR engine",
        "2. PaddleOCR engine", 
        "3. Tesseract fallback",
        "4. HTML output",
        "5. Markdown conversion"
    ]
    
    for capability in capabilities:
        draw.text((col2_x, current_y), capability, fill='black', font=body_font)
        current_y += 25
    
    # Footer section
    footer_y = 400
    draw.text((50, footer_y), "Performance Metrics", fill='black', font=header_font)
    footer_y += 30
    
    metrics_text = """
Processing Speed: 2-12 seconds per image
Accuracy Rate: 95-98% with PaddleOCR
Layout Preservation: Ultra-precise positioning
Export Formats: TXT, HTML, MD, JSON
    """.strip()
    
    for line in metrics_text.split('\n'):
        draw.text((50, footer_y), line, fill='black', font=body_font)
        footer_y += 20
    
    return image

def test_precision_layout():
    """Test the precision layout system"""
    if not PRECISION_AVAILABLE:
        print("❌ Cannot test precision layout - system not available")
        return
    
    print("\n🔬 Testing Precision Layout OCR System")
    print("=" * 50)
    
    # Create test document
    print("📄 Creating test document with complex layout...")
    test_image = create_test_document()
    
    # Save test image
    test_image_path = "precision_test_document.png"
    test_image.save(test_image_path)
    print(f"✅ Test document saved as: {test_image_path}")
    
    # Convert to OpenCV format
    image_np = np.array(test_image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Test precision layout extraction
    print("\n🎯 Testing precision layout extraction...")
    start_time = time.time()
    
    try:
        result = extract_text_with_precision_layout(image_np)
        processing_time = time.time() - start_time
        
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        
        if 'error' in result:
            print(f"❌ Error in precision layout: {result['error']}")
            return
        
        # Display results
        print("\n📊 Analysis Results:")
        print("-" * 30)
        
        layout_analysis = result.get('layout_analysis', {})
        if layout_analysis:
            print(f"Text Elements: {layout_analysis.get('total_elements', 0)}")
            print(f"Line Groups: {layout_analysis.get('line_groups', 0)}")
            
            columns = layout_analysis.get('columns', {})
            print(f"Columns Detected: {columns.get('count', 1)}")
            
            structure = layout_analysis.get('structure', {})
            print(f"Titles Found: {len(structure.get('titles', []))}")
            print(f"Bullet Points: {len(structure.get('bullet_points', []))}")
            print(f"Numbered Lists: {len(structure.get('numbered_lists', []))}")
        
        # Save results
        precision_text = result.get('precision_formatted', '')
        if precision_text:
            with open('precision_layout_output.txt', 'w', encoding='utf-8') as f:
                f.write(precision_text)
            print(f"✅ Precision formatted text saved to: precision_layout_output.txt")
        
        markdown_text = result.get('markdown_layout', '')
        if markdown_text:
            with open('precision_layout_output.md', 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            print(f"✅ Markdown output saved to: precision_layout_output.md")
        
        html_content = result.get('html_layout', '')
        if html_content:
            with open('precision_layout_output.html', 'w', encoding='utf-8') as f:
                f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Precision Layout Test Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Precision Layout OCR Results</h1>
        <h2>Visual Layout Reproduction</h2>
        {html_content}
    </div>
</body>
</html>
                """)
            print(f"✅ HTML layout saved to: precision_layout_output.html")
        
        # Display sample output
        print("\n📝 Sample Precision Formatted Output:")
        print("-" * 40)
        lines = precision_text.split('\n')[:10]  # First 10 lines
        for i, line in enumerate(lines, 1):
            print(f"{i:2d}: {line}")
        if len(precision_text.split('\n')) > 10:
            print("    ... (truncated)")
        
        print("\n✅ Precision layout test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during precision layout test: {e}")
        import traceback
        traceback.print_exc()

def compare_with_basic_ocr():
    """Compare precision layout with basic OCR"""
    if not ENHANCED_OCR_AVAILABLE or not PRECISION_AVAILABLE:
        print("❌ Cannot compare - required systems not available")
        return
    
    print("\n🔍 Comparing Precision Layout vs Basic OCR")
    print("=" * 50)
    
    # Load test image
    test_image_path = "precision_test_document.png"
    if not os.path.exists(test_image_path):
        print("❌ Test image not found - run precision layout test first")
        return
    
    image = cv2.imread(test_image_path)
    
    # Basic OCR
    print("🔧 Running basic multi-engine OCR...")
    start_time = time.time()
    basic_result = extract_text_enhanced_multi_engine(image)
    basic_time = time.time() - start_time
    
    # Precision OCR
    print("🎯 Running precision layout OCR...")
    start_time = time.time()
    precision_result = extract_text_with_precision_layout(image)
    precision_time = time.time() - start_time
    
    # Compare results
    print("\n📊 Comparison Results:")
    print("-" * 30)
    print(f"Basic OCR Time:     {basic_time:.2f}s")
    print(f"Precision OCR Time: {precision_time:.2f}s")
    print(f"Time Difference:    {abs(precision_time - basic_time):.2f}s")
    
    basic_text = basic_result.get('text', '')
    precision_text = precision_result.get('precision_formatted', '')
    
    print(f"\nBasic OCR Length:     {len(basic_text)} characters")
    print(f"Precision OCR Length: {len(precision_text)} characters")
    
    # Save comparison
    with open('ocr_comparison.txt', 'w', encoding='utf-8') as f:
        f.write("OCR COMPARISON RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write("BASIC MULTI-ENGINE OCR:\n")
        f.write("-" * 25 + "\n")
        f.write(basic_text)
        f.write("\n\n")
        f.write("PRECISION LAYOUT OCR:\n")
        f.write("-" * 25 + "\n")
        f.write(precision_text)
    
    print("✅ Comparison saved to: ocr_comparison.txt")

def main():
    """Main test function"""
    print("🚀 Starting Precision Layout OCR Test Suite")
    print("=" * 60)
    
    # System status
    print("\n🔋 System Status:")
    print(f"Enhanced Multi-Engine OCR: {'✅ Available' if ENHANCED_OCR_AVAILABLE else '❌ Not Available'}")
    print(f"Precision Layout OCR:      {'✅ Available' if PRECISION_AVAILABLE else '❌ Not Available'}")
    
    if not ENHANCED_OCR_AVAILABLE and not PRECISION_AVAILABLE:
        print("\n❌ No OCR systems available - exiting")
        return
    
    # Run tests
    test_precision_layout()
    compare_with_basic_ocr()
    
    print("\n🎉 Test suite completed!")
    print("\nGenerated files:")
    files = [
        "precision_test_document.png",
        "precision_layout_output.txt", 
        "precision_layout_output.md",
        "precision_layout_output.html",
        "ocr_comparison.txt"
    ]
    
    for filename in files:
        if os.path.exists(filename):
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename} (not created)")

if __name__ == "__main__":
    main()
