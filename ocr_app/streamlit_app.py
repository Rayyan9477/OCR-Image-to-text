#!/usr/bin/env python
"""
Streamlit application entry point

This script runs the OCR web interface using Streamlit.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set environment variables for consistent behavior
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # Use legacy Keras with TF
os.environ["KERAS_BACKEND"] = "tensorflow"  # Ensure TF backend

# Add the parent directory to sys.path if needed
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def main():
    """Run the Streamlit web application"""
    from ocr_app.ui.web_app import StreamlitApp
    
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()
