#!/usr/bin/env python3
"""
OCR System Command-Line Interface

This script provides a command-line interface for the enhanced OCR system
for processing images and PDFs with advanced options.
"""

import os
import argparse
from PIL import Image
import time
import sys
import traceback
import importlib.util

def check_dependencies():
    """Check for required dependencies and provide helpful error messages"""
    missing_deps = []
    
    # Check PIL/Pillow
    try:
        from PIL import Image
        print("✓ PIL/Pillow: Available")
    except ImportError:
        missing_deps.append("PIL/Pillow (pip install Pillow)")
        print("✗ PIL/Pillow: Missing")
    
    # Check PyMuPDF for PDF support
    try:
        import fitz
        print("✓ PyMuPDF: Available (PDF support enabled)")
    except ImportError:
        missing_deps.append("PyMuPDF (pip install pymupdf)")
        print("✗ PyMuPDF: Missing (PDF support disabled)")
    
    # Check for OCR engines
    ocr_engines = []
    
    # Check PaddleOCR
    paddle_spec = importlib.util.find_spec('paddleocr')
    if paddle_spec:
        print("✓ PaddleOCR: Available")
        ocr_engines.append("PaddleOCR")
    else:
        print("✗ PaddleOCR: Missing")
    
    # Check EasyOCR
    easy_spec = importlib.util.find_spec('easyocr')
    if easy_spec:
        print("✓ EasyOCR: Available")
        ocr_engines.append("EasyOCR")
    else:
        print("✗ EasyOCR: Missing")
    
    # Check PyTesseract
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print("✓ PyTesseract: Available")
        ocr_engines.append("PyTesseract")
    except:
        print("✗ PyTesseract: Missing or Tesseract not in PATH")
    
    if not ocr_engines:
        missing_deps.append("At least one OCR engine (pip install pytesseract easyocr paddleocr)")
        print("\n⚠️ No OCR engines available. OCR functionality will be limited.")
    else:
        print(f"\n✓ Available OCR engines: {', '.join(ocr_engines)}")
    
    return missing_deps

def process_image(image_path, engine="combined", preserve_layout=True, output_format="txt"):
    """Process a single image file"""
    print(f"\nProcessing image: {os.path.basename(image_path)}")
    start_time = time.time()
    
    try:
        # Load image
        image = Image.open(image_path)
        
        # Check if we need to run table detection
        has_tables = False
        try:
            from ocr_module import detect_tables
            has_tables = detect_tables(image)
            if has_tables:
                print("📊 Tables detected in the image. Layout preservation enabled.")
                preserve_layout = True
        except ImportError:
            pass
        
        # Perform OCR - try main module first, fall back to lite module
        try:
            from ocr_module import perform_ocr
            print(f"Performing OCR using {engine} engine...")
            extracted_text = perform_ocr(image, engine=engine, preserve_layout=preserve_layout)
        except ImportError:
            # Fall back to lite version
            print(f"Using lightweight OCR module...")
            from ocr_module_lite import LiteOCR
            lite_ocr = LiteOCR()
            extracted_text = lite_ocr.perform_ocr(image, engine="auto" if engine == "combined" else engine, preserve_layout=preserve_layout)
        
        processing_time = time.time() - start_time
        print(f"OCR completed in {processing_time:.2f} seconds")
        
        # Format output based on selected format
        if output_format == "json":
            import json
            result = {
                "filename": os.path.basename(image_path),
                "processing_time": processing_time,
                "engine": engine,
                "preserve_layout": preserve_layout,
                "has_tables": has_tables,
                "text": extracted_text
            }
            return json.dumps(result, indent=2)
        elif output_format == "md":
            return f"# OCR Results: {os.path.basename(image_path)}\n\n" + \
                   f"**Engine:** {engine}  \n" + \
                   f"**Processing Time:** {processing_time:.2f} seconds  \n" + \
                   f"**Layout Preserved:** {'Yes' if preserve_layout else 'No'}  \n\n" + \
                   "## Extracted Text\n\n```\n" + extracted_text + "\n```"
        else:  # txt format
            return extracted_text
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        traceback.print_exc()
        return f"ERROR: {str(e)}"

def process_pdf(pdf_path, engine="combined", preserve_layout=True, output_format="txt"):
    """Process a PDF file"""
    print(f"\nProcessing PDF: {os.path.basename(pdf_path)}")
    start_time = time.time()
    
    try:
        # Try to import PyMuPDF
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return "ERROR: PyMuPDF not installed. Install with 'pip install pymupdf' for PDF support."
        
        # Open PDF
        doc = fitz.open(pdf_path)
        print(f"PDF has {len(doc)} pages")
        
        extracted_text = ""
        page_results = []
        
        # Process each page
        for i, page in enumerate(doc):
            page_start = time.time()
            print(f"Processing page {i+1}/{len(doc)}...")
            
            # Convert PDF page to image
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Check if the page contains tables
            has_tables = False
            try:
                from ocr_module import detect_tables
                has_tables = detect_tables(img)
                if has_tables:
                    print("📊 Tables detected on page. Layout preservation enabled.")
                    page_preserve_layout = True
                else:
                    page_preserve_layout = preserve_layout
            except ImportError:
                page_preserve_layout = preserve_layout
            
            # Perform OCR on the page
            try:
                from ocr_module import perform_ocr
                page_text = perform_ocr(img, engine=engine, preserve_layout=page_preserve_layout)
            except ImportError:
                # Fall back to lite version
                from ocr_module_lite import LiteOCR
                lite_ocr = LiteOCR()
                page_text = lite_ocr.perform_ocr(img, engine="auto" if engine == "combined" else engine, preserve_layout=page_preserve_layout)
            
            if output_format == "json":
                page_results.append({
                    "page_number": i+1,
                    "has_tables": has_tables,
                    "text": page_text,
                    "processing_time": time.time() - page_start
                })
            else:
                # Add page separator and text
                extracted_text += f"\n--- PAGE {i+1} ---\n{page_text}\n"
            
            page_time = time.time() - page_start
            print(f"Page {i+1} processed in {page_time:.2f} seconds")
        
        processing_time = time.time() - start_time
        print(f"PDF processing completed in {processing_time:.2f} seconds")
        
        # Format output based on selected format
        if output_format == "json":
            import json
            result = {
                "filename": os.path.basename(pdf_path),
                "total_pages": len(doc),
                "processing_time": processing_time,
                "engine": engine,
                "preserve_layout": preserve_layout,
                "pages": page_results
            }
            return json.dumps(result, indent=2)
        elif output_format == "md":
            md_output = f"# OCR Results: {os.path.basename(pdf_path)}\n\n" + \
                       f"**Engine:** {engine}  \n" + \
                       f"**Pages:** {len(doc)}  \n" + \
                       f"**Processing Time:** {processing_time:.2f} seconds  \n" + \
                       f"**Layout Preserved:** {'Yes' if preserve_layout else 'No'}  \n\n" + \
                       "## Extracted Text\n\n```\n" + extracted_text + "\n```"
            return md_output
        else:  # txt format
            return extracted_text
            
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        traceback.print_exc()
        return f"ERROR: {str(e)}"

def save_output(text, output_path):
    """Save extracted text to a file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Output saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving output: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='OCR System Command Line Interface')
    parser.add_argument('input_path', help='Path to input image or PDF file')
    parser.add_argument('--output', '-o', help='Path to output text file')
    parser.add_argument('--engine', '-e', choices=['paddle', 'easy', 'combined', 'dolphin'], 
                        default='combined', help='OCR engine to use (paddle, easy, combined, dolphin)')
    parser.add_argument('--no-layout', action='store_true', 
                        help='Disable layout preservation')
    parser.add_argument('--format', '-f', choices=['txt', 'json', 'md'], 
                        default='txt', help='Output format')
    parser.add_argument('--batch', '-b', help='Process all files in a directory')
    parser.add_argument('--check', action='store_true', 
                        help='Check dependencies and exit')
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check:
        print("=== Dependency Check ===")
        missing = check_dependencies()
        if missing:
            print("\n⚠️ Missing dependencies:")
            for dep in missing:
                print(f"  - {dep}")
            print("\nPlease install the missing dependencies to enable full functionality.")
        else:
            print("\n✓ All dependencies installed!")
        return 0
    
    # Initialize models if available
    print("Checking OCR models...")
    try:
        import model_manager
        model_manager.initialize_models()
    except ImportError:
        print("Using lightweight OCR module (limited features)")
        pass
    
    # Batch processing
    if args.batch:
        if not os.path.isdir(args.batch):
            print(f"Batch directory not found: {args.batch}")
            return 1
        
        # Create output directory if it doesn't exist
        output_dir = args.output if args.output else os.path.join(args.batch, "ocr_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Process all supported files
        supported_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.pdf']
        success_count = 0
        fail_count = 0
        
        for filename in os.listdir(args.batch):
            file_path = os.path.join(args.batch, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            if os.path.isfile(file_path) and file_ext in supported_exts:
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.{args.format}")
                print(f"\nProcessing file {filename}...")
                
                try:
                    if file_ext in ['.pdf']:
                        text = process_pdf(file_path, args.engine, not args.no_layout, args.format)
                    else:
                        text = process_image(file_path, args.engine, not args.no_layout, args.format)
                    
                    if save_output(text, output_path):
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    fail_count += 1
        
        print(f"\nBatch processing completed. Success: {success_count}, Failed: {fail_count}")
        return 0
    
    # Single file processing
    # Set output path if not specified
    if not args.output:
        base_name = os.path.splitext(args.input_path)[0]
        args.output = f"{base_name}_ocr.{args.format}"
    
    # Process based on file type
    file_ext = os.path.splitext(args.input_path)[1].lower()
    
    try:
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            text = process_image(args.input_path, args.engine, not args.no_layout, args.format)
        elif file_ext == '.pdf':
            text = process_pdf(args.input_path, args.engine, not args.no_layout, args.format)
        else:
            print(f"Unsupported file type: {file_ext}")
            return 1
        
        # Save output
        save_output(text, args.output)
        
        # Print sample of extracted text
        print("\nSample of extracted text:")
        print("=" * 40)
        sample_text = text[:500] if args.format == 'txt' else "Output saved in requested format"
        print(sample_text + ("..." if len(text) > 500 and args.format == 'txt' else ""))
        print("=" * 40)
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())