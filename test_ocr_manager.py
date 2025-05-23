#!/usr/bin/env python3
"""
Test script for OCR Manager
"""

import os
import sys
import argparse
from PIL import Image
from ocr_manager import check_ocr_engines, perform_ocr, detect_tables

def main():
    parser = argparse.ArgumentParser(description='Test OCR functionality')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--engine', type=str, default='auto', 
                        choices=['auto', 'paddle', 'easy', 'tesseract', 'combined'],
                        help='OCR engine to use')
    parser.add_argument('--layout', action='store_true', help='Preserve layout')
    parser.add_argument('--enhance', action='store_true', help='Enhance image')
    
    args = parser.parse_args()
    
    # Check available OCR engines
    available_engines, missing_engines, installation_instructions = check_ocr_engines()
    
    print("Available OCR engines:", available_engines)
    print("Missing OCR engines:", missing_engines)
    
    if missing_engines:
        print("\nInstallation instructions:")
        for instruction in installation_instructions:
            print(f"- {instruction}")
    
    # If image path provided, perform OCR
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}")
            return
        
        try:
            image = Image.open(args.image)
            print(f"Loaded image: {args.image} ({image.width}x{image.height})")
            
            # Check for tables
            has_tables = detect_tables(image)
            if has_tables:
                print("Table structures detected in the image")
            
            print(f"Performing OCR using {args.engine} engine...")
            text = perform_ocr(
                image, 
                engine=args.engine, 
                preserve_layout=args.layout,
                enhance_image=args.enhance
            )
            
            print("\nExtracted Text:")
            print("-" * 40)
            print(text)
            print("-" * 40)
            
            # Save the extracted text to a file
            output_file = os.path.splitext(args.image)[0] + "_ocr.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Text saved to: {output_file}")
            
        except Exception as e:
            print(f"Error processing image: {e}")
    else:
        print("No image provided. Use --image to specify an image file to process.")

if __name__ == "__main__":
    main()
