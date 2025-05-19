#!/usr/bin/env python
"""
OCR Runner Script - A wrapper script to run OCR with appropriate environment variables
"""

import os
import sys
import subprocess
import importlib

def check_dependency(module_name):
    """Check if a dependency is installed"""
    try:
        importlib.util.find_spec(module_name)
        return True
    except ImportError:
        return False

def main():
    """Main entry point for running OCR with appropriate flags"""
    print("OCR Image to Text - Setup Utility")
    print("--------------------------------\n")
    
    # Set environment variables for TensorFlow/Keras compatibility
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings
    os.environ["TF_USE_LEGACY_KERAS"] = "1"   # Use legacy Keras with TF
    os.environ["KERAS_BACKEND"] = "tensorflow"  # Ensure TF backend
    
    # Check key dependencies
    dependencies = {
        "streamlit": "For the web interface",
        "PIL": "For image processing",
        "transformers": "For question-answering (optional)",
        "paddleocr": "For PaddleOCR (optional)",
        "easyocr": "For EasyOCR (optional)",
        "pytesseract": "For Tesseract OCR fallback (optional)",
        "torch": "For deep learning models (optional)"
    }
    
    print("Checking dependencies...")
    missing = []
    for dep, desc in dependencies.items():
        if dep == "PIL":
            # PIL is imported as Pillow
            has_dep = check_dependency("PIL")
        else:
            has_dep = check_dependency(dep)
        
        status = "✓" if has_dep else "✗"
        print(f"  {status} {dep} - {desc}")
        if not has_dep and dep == "streamlit":
            missing.append(dep)
    
    if missing and "streamlit" in missing:
        print("\nCritical dependency missing. Please install with:")
        print("  pip install -r requirements.txt")
        return 1
    
    print("\nRunning OCR application...")
    cmd = ["streamlit", "run", "app.py"]
    
    # Execute the command
    try:
        subprocess.run(cmd)
        return 0
    except Exception as e:
        print(f"Error executing OCR application: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
