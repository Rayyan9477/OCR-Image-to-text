"""
Streamlit application entry point test program

This script runs a simple version of the OCR web interface using Streamlit.
"""

import sys
import os
import logging
from pathlib import Path
from PIL import Image
import time
import argparse

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import OCR components
from ocr_app.core.ocr_engine import OCREngine
from ocr_app.core.image_processor import ImageProcessor
from ocr_app.config.settings import Settings
from ocr_app.rag.rag_processor import RAGProcessor
from ocr_app.models.model_manager import ModelManager

def main():
    """Run a simple test of the OCR application"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test OCR functionality")
    parser.add_argument("--image", "-i", type=str, help="Path to image file to process")
    parser.add_argument("--query", "-q", type=str, help="Question to ask about the image")
    parser.add_argument("--engine", "-e", type=str, default="auto", 
                       choices=["auto", "tesseract", "easyocr", "paddleocr", "combined"],
                       help="OCR engine to use")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize components
    settings = Settings()
    ocr_engine = OCREngine(settings)
    model_manager = ModelManager(settings)
    rag_processor = RAGProcessor(model_manager, settings)
    
    # Print available OCR engines
    print(f"Available OCR engines: {ocr_engine.enabled_engines}")
    
    # If no image specified, use a test image
    if not args.image:
        # Look for test images in the repository
        test_images = []
        test_patterns = ["*test*.jpg", "*test*.png", "*sample*.jpg", "*sample*.png"]
        for pattern in test_patterns:
            test_images.extend(list(Path.cwd().glob(pattern)))
        
        if test_images:
            image_path = test_images[0]
            print(f"Using test image: {image_path}")
        else:
            print("No test image found and no image specified with --image")
            return 1
    else:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Image file not found: {image_path}")
            return 1
    
    # Open the image
    try:
        image = Image.open(image_path)
        print(f"Opened image: {image_path} (Size: {image.width}x{image.height})")
    except Exception as e:
        print(f"Error opening image: {e}")
        return 1
    
    # Process the image
    print(f"Processing image with {args.engine} engine...")
    start_time = time.time()
    
    try:
        text = ocr_engine.perform_ocr(
            image,
            engine=args.engine,
            preserve_layout=True,
            preprocess=True
        )
        elapsed_time = time.time() - start_time
        print(f"OCR completed in {elapsed_time:.2f} seconds")
        
        # Print extracted text
        print("\n--- Extracted Text ---\n")
        print(text)
        print(f"\nExtracted {len(text)} characters")
        
        # Process question if provided
        if args.query:
            print(f"\nProcessing question: {args.query}")
            qa_start_time = time.time()
            answer = rag_processor.process_query(text, args.query)
            qa_elapsed_time = time.time() - qa_start_time
            
            print(f"\n--- Answer (processed in {qa_elapsed_time:.2f} seconds) ---\n")
            print(f"Q: {args.query}")
            print(f"A: {answer['answer']}")
            print(f"Confidence: {answer.get('confidence', 0):.2f}")
        
        return 0
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
