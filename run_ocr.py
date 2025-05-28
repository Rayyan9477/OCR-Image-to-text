#!/usr/bin/env python
"""
OCR Runner Script - A wrapper script to run OCR with appropriate environment variables and options.
This script provides an easy-to-use entry point for the OCR system with multiple modes:
1. Web UI mode (default) - Launches the Streamlit web application
2. CLI mode - Processes files directly from the command line
3. Test mode - Tests the OCR functionality with sample images
4. Setup mode - Installs and configures dependencies
"""

import os
import sys
import subprocess
import importlib
import importlib.util
import argparse
import platform
from pathlib import Path

# Set environment variables for TensorFlow/Keras compatibility
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # Use legacy Keras with TF
os.environ["KERAS_BACKEND"] = "tensorflow"  # Ensure TF backend

def check_dependency(module_name):
    """Check if a dependency is installed"""
    try:
        importlib.util.find_spec(module_name)
        return True
    except ImportError:
        return False

def check_all_dependencies():
    """Check all key dependencies for the OCR system"""
    dependencies = {
        "streamlit": "For the web interface",
        "PIL": "For image processing",
        "transformers": "For question-answering (optional)",
        "paddleocr": "For PaddleOCR engine (optional)",
        "easyocr": "For EasyOCR engine (optional)",
        "pytesseract": "For Tesseract OCR fallback (optional)",
        "torch": "For deep learning models (optional)",
        "fitz": "For PDF processing (PyMuPDF)",
        "numpy": "For numerical operations",
        "tensorflow": "For machine learning operations (optional)"
    }
    
    print("Checking dependencies...")
    missing = []
    optional_missing = []
    
    for dep, desc in dependencies.items():
        if dep == "PIL":
            # PIL is imported as Pillow
            has_dep = check_dependency("PIL")
        elif dep == "fitz":
            # PyMuPDF is imported as fitz
            has_dep = check_dependency("fitz")
        else:
            has_dep = check_dependency(dep)
        
        status = "✓" if has_dep else "✗"
        print(f"  {status} {dep} - {desc}")
        
        # Check if it's a core dependency or optional
        if not has_dep:
            if "optional" in desc.lower():
                optional_missing.append(dep)
            else:
                missing.append(dep)
    
    # Check for Tesseract installation separately (not just the Python package)
    tesseract_available = False
    if check_dependency("pytesseract"):
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            tesseract_available = True
            print("  ✓ Tesseract - OCR engine binary installed")
        except:
            print("  ✗ Tesseract - OCR engine binary not found in PATH")
            optional_missing.append("Tesseract binary")
    
    # Check for OCR engines availability
    ocr_engines = []
    if check_dependency("paddleocr"):
        ocr_engines.append("PaddleOCR")
    if check_dependency("easyocr"):
        ocr_engines.append("EasyOCR")
    if tesseract_available:
        ocr_engines.append("PyTesseract")
    
    if ocr_engines:
        print(f"\n✓ Available OCR engines: {', '.join(ocr_engines)}")
    else:
        print("\n⚠️ No OCR engines available. OCR functionality will be limited.")
        
    return missing, optional_missing

def create_test_image():
    """Create a simple test image with text for OCR testing"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        width, height = 800, 400
        background_color = (255, 255, 255)
        text_color = (0, 0, 0)
        
        # Create a blank image
        image = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(image)
        
        # Try to get a font
        try:
            # Try to use a common font
            if platform.system() == "Windows":
                font_path = "C:\\Windows\\Fonts\\Arial.ttf"
            elif platform.system() == "Darwin":  # macOS
                font_path = "/System/Library/Fonts/Helvetica.ttc"
            else:  # Linux and others
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 32)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Add text to the image
        text1 = "OCR Test Image"
        text2 = "This is a sample image for testing OCR functionality."
        text3 = "It contains different text sizes and formatting."
        text4 = "1234567890 !@#$%^&*()"
        
        draw.text((width/2-150, 50), text1, fill=text_color, font=font)
        draw.text((50, 150), text2, fill=text_color, font=font)
        draw.text((50, 200), text3, fill=text_color, font=font)
        draw.text((50, 250), text4, fill=text_color, font=font)
        
        # Draw some lines and a rectangle
        draw.line([(50, 300), (750, 300)], fill=(0, 0, 0), width=3)
        draw.rectangle([(50, 320), (750, 380)], outline=(0, 0, 0), width=2)
        draw.text((400-100, 340), "Text in a box", fill=text_color, font=font)
        
        # Save the image
        image.save('simple_test_image.jpg', quality=95)
        print(f"Created test image: simple_test_image.jpg")
        return True
    except Exception as e:
        print(f"Error creating test image: {str(e)}")
        return False

def run_web_interface():
    """Launch the web interface using Streamlit"""
    print("\nLaunching OCR web application...")
    
    if not check_dependency("streamlit"):
        print("\nError: Streamlit is not installed!")
        print("Please install it with: pip install streamlit")
        print("Or install all dependencies with: pip install -r requirements.txt")
        return 1
    
    cmd = ["streamlit", "run", "app.py"]
    
    try:
        subprocess.run(cmd)
        return 0
    except Exception as e:
        print(f"Error executing OCR application: {e}")
        return 1

def run_cli_mode(args):
    """Run the OCR CLI with provided arguments"""
    print("\nRunning OCR in CLI mode...")
    
    if not os.path.exists("ocr_cli.py"):
        print("Error: ocr_cli.py not found!")
        return 1
    
    # Build the command based on provided arguments
    cmd = ["python", "ocr_cli.py"]
    
    if args.input:
        cmd.append(args.input)
    
    if args.output:
        cmd.extend(["--output", args.output])
    
    if args.engine:
        cmd.extend(["--engine", args.engine])
    
    if args.no_layout:
        cmd.append("--no-layout")
    
    if args.format:
        cmd.extend(["--format", args.format])
    
    if args.batch:
        cmd.extend(["--batch", args.batch])
    
    if args.check:
        cmd.append("--check")
    
    # Execute the command
    try:
        subprocess.run(cmd)
        return 0
    except Exception as e:
        print(f"Error executing OCR CLI: {e}")
        return 1

def run_test_mode():
    """Run the OCR testing script"""
    print("\nRunning OCR test mode...")
    
    # Create test image if it doesn't exist
    if not os.path.exists("simple_test_image.jpg"):
        created = create_test_image()
        if not created:
            print("Error: Couldn't create test image!")
    
    # Run the test script
    if os.path.exists("test_ocr.py"):
        try:
            subprocess.run(["python", "test_ocr.py"])
            return 0
        except Exception as e:
            print(f"Error running test script: {e}")
            return 1
    else:
        print("Error: test_ocr.py not found!")
        return 1

def main():
    """Main entry point for the OCR application"""
    parser = argparse.ArgumentParser(description='OCR Image-to-Text System')
    
    # Mode selection
    mode_group = parser.add_argument_group('Mode Selection')
    mode_group.add_argument('--web', '-w', action='store_true',
                         help='Run in web interface mode (default)')
    mode_group.add_argument('--cli', '-c', action='store_true',
                         help='Run in command-line interface mode')
    mode_group.add_argument('--test', '-t', action='store_true',
                         help='Run test mode to verify OCR functionality')
    mode_group.add_argument('--setup', '-s', action='store_true',
                         help='Setup and check dependencies')
    
    # CLI mode options
    cli_group = parser.add_argument_group('CLI Mode Options')
    cli_group.add_argument('--input', '-i', 
                        help='Path to input image or PDF file')
    cli_group.add_argument('--output', '-o', 
                        help='Path to output text file')
    cli_group.add_argument('--engine', '-e', choices=['paddle', 'easy', 'combined', 'dolphin'], 
                        help='OCR engine to use (paddle, easy, combined, dolphin)')
    cli_group.add_argument('--no-layout', action='store_true',
                        help='Disable layout preservation')
    cli_group.add_argument('--format', '-f', choices=['txt', 'json', 'md'],
                        help='Output format (txt, json, or md)')
    cli_group.add_argument('--batch', '-b', 
                        help='Process all files in a directory')
    cli_group.add_argument('--check', action='store_true',
                        help='Check dependencies and exit')
    
    args = parser.parse_args()
    
    print("OCR Image to Text - Multi-mode System")
    print("--------------------------------")
    
    # Always check for basic dependencies
    missing, optional_missing = check_all_dependencies()
    
    # Run in setup mode - just show dependency status and exit
    if args.setup:
        if missing:
            print("\n⚠️ Missing core dependencies:")
            for dep in missing:
                print(f"  - {dep}")
        
        if optional_missing:
            print("\n⚠️ Missing optional dependencies:")
            for dep in optional_missing:
                print(f"  - {dep}")
        
        if not missing and not optional_missing:
            print("\n✅ All dependencies installed!")
        
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
        
        print("\nFor system dependencies (like Tesseract):")
        if platform.system() == "Windows":
            print("  Download and install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
        elif platform.system() == "Darwin":  # macOS
            print("  brew install tesseract")
        else:  # Linux
            print("  sudo apt-get install tesseract-ocr")
        
        return 0
    
    # Choose the appropriate mode
    if args.cli:
        # CLI mode
        return run_cli_mode(args)
    elif args.test:
        # Test mode
        return run_test_mode()
    else:
        # Default to web interface
        return run_web_interface()

if __name__ == "__main__":
    sys.exit(main())
