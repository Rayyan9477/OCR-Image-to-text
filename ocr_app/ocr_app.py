#!/usr/bin/env python
"""
OCR Image-to-Text Application Launcher

This script serves as the main entry point for the OCR application with multiple modes:
1. Web UI mode (default) - Launches the Streamlit web application
2. CLI mode - Processes files directly from the command line
3. Check mode - Checks available OCR engines and dependencies
"""

import os
import sys
import argparse
import logging
from importlib import import_module

# Set environment variables for consistent behavior
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # Use legacy Keras with TF
os.environ["KERAS_BACKEND"] = "tensorflow"  # Ensure TF backend

def main():
    """Main entry point for the application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="OCR Image-to-Text Application")
    
    # Mode selection
    mode_group = parser.add_argument_group('Mode Selection')
    mode_group.add_argument('--web', '-w', action='store_true',
                         help='Run in web interface mode (default)')
    mode_group.add_argument('--cli', '-c', action='store_true',
                         help='Run in command-line interface mode')
    mode_group.add_argument('--check', action='store_true',
                         help='Check available OCR engines and dependencies')
    
    # Forward remaining arguments to the selected mode
    parser.add_argument('args', nargs=argparse.REMAINDER,
                      help='Arguments to pass to the selected mode')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add the parent directory to sys.path if needed
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Run in check mode - just verify engines and dependencies
    if args.check:
        return run_check_mode()
    
    # Run in CLI mode
    if args.cli:
        return run_cli_mode(args.args)
    
    # Default to web interface
    return run_web_mode()

def run_check_mode():
    """Run in check mode to verify engines and dependencies"""
    try:
        from ocr_app.core.ocr_engine import OCREngine
        from ocr_app.models.model_manager import ModelManager
        from ocr_app.config.settings import Settings
        
        print("OCR Image-to-Text System Check")
        print("==============================")
        
        # Check OCR engines
        print("\nChecking OCR engines...")
        settings = Settings()
        ocr_engine = OCREngine(settings)
        
        available_engines = ocr_engine.enabled_engines
        
        if available_engines:
            print(f"✓ Available OCR engines: {', '.join(available_engines)}")
        else:
            print("⚠️ No OCR engines available. OCR functionality will be limited.")
        
        # Check ML models
        print("\nChecking ML components...")
        model_manager = ModelManager(settings)
        module_status = model_manager.get_module_status()
        
        for module, available in module_status.items():
            status = "✓ Available" if available else "✗ Not available"
            print(f"{module}: {status}")
        
        # Print installation instructions
        print("\n=== Installation Instructions ===")
        if 'tesseract' not in available_engines:
            print("\nTo install Tesseract OCR:")
            if sys.platform == 'win32':
                print("  1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki")
                print("  2. Install and add to PATH")
                print("  3. pip install pytesseract")
            elif sys.platform == 'darwin':
                print("  brew install tesseract")
                print("  pip install pytesseract")
            else:
                print("  sudo apt-get install tesseract-ocr")
                print("  pip install pytesseract")
        
        if 'easyocr' not in available_engines:
            print("\nTo install EasyOCR:")
            print("  pip install easyocr")
        
        if 'paddleocr' not in available_engines:
            print("\nTo install PaddleOCR:")
            print("  pip install paddlepaddle paddleocr")
        
        if not module_status.get('transformers_available', False):
            print("\nTo install Transformers (for Q&A functionality):")
            print("  pip install transformers")
        
        if not module_status.get('sentence_transformers_available', False):
            print("\nTo install Sentence Transformers (for text embedding):")
            print("  pip install sentence-transformers")
        
        # Install all with requirements file
        print("\nTo install all dependencies at once:")
        print("  pip install -r requirements.txt")
        
        return 0
    
    except Exception as e:
        print(f"Error during system check: {e}")
        return 1

def run_cli_mode(cli_args):
    """Run in CLI mode to process files from command line"""
    try:
        from ocr_app.ui.cli import OCRCLI
        
        cli = OCRCLI()
        return cli.run(cli_args)
    
    except Exception as e:
        print(f"Error in CLI mode: {e}")
        return 1

def run_web_mode():
    """Run in web interface mode using Streamlit"""
    try:
        # We use subprocess to launch Streamlit in a separate process
        import subprocess
        import pkg_resources
        
        # Check if Streamlit is installed
        try:
            pkg_resources.get_distribution('streamlit')
        except pkg_resources.DistributionNotFound:
            print("Streamlit is not installed. Please install it with:")
            print("  pip install streamlit")
            return 1
        
        # Path to the Streamlit app entry point
        streamlit_app_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'streamlit_app.py'
        )
        
        # Launch Streamlit app
        cmd = [sys.executable, "-m", "streamlit", "run", streamlit_app_path, "--server.headless", "true"]
        
        print("Starting web interface...")
        process = subprocess.Popen(cmd)
        process.wait()
        return process.returncode
    
    except Exception as e:
        print(f"Error starting web interface: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
