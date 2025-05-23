#!/usr/bin/env python3
"""
Extract Text From Image - Simple OCR Tool

This script provides a quick way to extract text from images or PDFs.
Just drag and drop an image or PDF onto this script or run it with a file path.

Usage:
  python extract_text.py [file_path]

Examples:
  python extract_text.py image.jpg
  python extract_text.py document.pdf
  python extract_text.py  # Will prompt for file selection
"""

import os
import sys
import time
import traceback
from PIL import Image

# Set environment variables for TensorFlow/Keras compatibility
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # Use legacy Keras with TF
os.environ["KERAS_BACKEND"] = "tensorflow"  # Ensure TF backend

def extract_text_from_image(image_path):
    """Extract text from an image file using the available OCR engines"""
    try:
        # Load image
        image = Image.open(image_path)
        print(f"Successfully loaded image: {os.path.basename(image_path)}")
        
        # Try main OCR module first
        print("Performing OCR...")
        try:
            # Try to use the main OCR module
            from ocr_module import perform_ocr
            text = perform_ocr(image, engine="combined", preserve_layout=True)
            print("Used main OCR engine")
        except ImportError:
            try:
                # Try the lite version
                from ocr_module_lite import LiteOCR
                lite_ocr = LiteOCR()
                text = lite_ocr.perform_ocr(image, engine="auto", preserve_layout=True)
                print("Used lightweight OCR engine")
            except ImportError:
                # Fall back to the most basic version
                try:
                    from ocr_fallback import FallbackOCR
                    fallback_ocr = FallbackOCR()
                    text = fallback_ocr.extract_text(image)
                    print("Used fallback OCR engine")
                except ImportError:
                    # Last resort - try pytesseract directly
                    try:
                        import pytesseract
                        text = pytesseract.image_to_string(image)
                        print("Used Tesseract directly")
                    except ImportError:
                        return "Error: No OCR engine available. Please install at least one OCR engine."
        
        return text
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        traceback.print_exc()
        return f"Error: {str(e)}"

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    try:
        # Try to import PyMuPDF
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return "Error: PyMuPDF not installed. Install with 'pip install pymupdf' for PDF support."
        
        # Open PDF
        doc = fitz.open(pdf_path)
        print(f"PDF has {len(doc)} pages")
        
        all_text = ""
        
        # Process each page
        for i, page in enumerate(doc):
            print(f"Processing page {i+1}/{len(doc)}...")
            
            # Convert PDF page to image
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Extract text from the page image
            try:
                # Try to use the main OCR module
                from ocr_module import perform_ocr
                page_text = perform_ocr(img, engine="combined", preserve_layout=True)
            except ImportError:
                try:
                    # Try the lite version
                    from ocr_module_lite import LiteOCR
                    lite_ocr = LiteOCR()
                    page_text = lite_ocr.perform_ocr(img, engine="auto", preserve_layout=True)
                except ImportError:
                    # Fall back to the most basic version
                    try:
                        from ocr_fallback import FallbackOCR
                        fallback_ocr = FallbackOCR()
                        page_text = fallback_ocr.extract_text(img)
                    except ImportError:
                        # Last resort - try pytesseract directly
                        try:
                            import pytesseract
                            page_text = pytesseract.image_to_string(img)
                        except ImportError:
                            return "Error: No OCR engine available. Please install at least one OCR engine."
            
            # Add page separator and text
            all_text += f"\n--- PAGE {i+1} ---\n{page_text}\n"
            
        return all_text
    
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        traceback.print_exc()
        return f"Error: {str(e)}"

def save_text_to_file(text, output_path):
    """Save extracted text to a file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Text saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving text: {str(e)}")
        return False

def select_file_dialog():
    """Show a file selection dialog"""
    try:
        # Try to use tkinter for a file dialog
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        file_path = filedialog.askopenfilename(
            title="Select an image or PDF file",
            filetypes=[
                ("Image & PDF files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.pdf"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            return file_path
        else:
            return None
    
    except:
        print("Could not open file dialog. Please provide a file path as an argument.")
        return None

def main():
    print("OCR Image-to-Text Extractor")
    print("-------------------------")
    
    # Get input file path
    input_path = None
    
    # Check if a file path was provided as an argument
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        if not os.path.exists(input_path):
            print(f"Error: File '{input_path}' not found.")
            return 1
    else:
        # No file path provided, try to show a file selection dialog
        input_path = select_file_dialog()
        
    if not input_path:
        print("No file selected. Exiting.")
        return 1
    
    print(f"Processing file: {input_path}")
    
    # Determine file type based on extension
    _, file_ext = os.path.splitext(input_path)
    file_ext = file_ext.lower()
    
    start_time = time.time()
    
    # Process the file based on its type
    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
        extracted_text = extract_text_from_image(input_path)
    elif file_ext == '.pdf':
        extracted_text = extract_text_from_pdf(input_path)
    else:
        print(f"Unsupported file type: {file_ext}")
        print("Supported formats: JPG, PNG, BMP, TIFF, PDF")
        return 1
    
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    
    # Generate output path
    base_name = os.path.splitext(input_path)[0]
    output_path = f"{base_name}_text.txt"
    
    # Save the extracted text
    save_text_to_file(extracted_text, output_path)
    
    # Display a preview of the text
    print("\nExtracted text preview:")
    print("-" * 40)
    preview = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
    print(preview)
    print("-" * 40)
    
    print(f"\nFull text saved to: {output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
