#!/usr/bin/env python
"""
Final Optimized Multi-Engine OCR Streamlit Application
Uses enhanced multi-engine system with EasyOCR, PaddleOCR, and Tesseract
"""

import os
import sys
import logging
import streamlit as st
import time
from PIL import Image
import base64
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
import json

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set environment variables for consistent behavior
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"

# Import enhanced multi-engine system
try:
    from enhanced_multi_engine_ocr import extract_text_enhanced_multi_engine
    ENHANCED_OCR_AVAILABLE = True
except ImportError as e:
    ENHANCED_OCR_AVAILABLE = False
    logging.warning(f"Enhanced multi-engine OCR system not available: {e}")

# Fallback imports
try:
    from ocr_app.utils.text_utils import extract_entities, format_ocr_result
except ImportError:
    def extract_entities(text):
        return {"emails": [], "phones": [], "dates": [], "numbers": []}
    
    def format_ocr_result(text, format_type):
        return text

def load_css():
    """Load custom CSS for better styling"""
    css = """
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .engine-status {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    
    .engine-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        min-width: 150px;
    }
    
    .engine-success {
        border-left: 4px solid #28a745;
    }
    
    .engine-error {
        border-left: 4px solid #dc3545;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .progress-bar {
        background: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        height: 20px;
        margin: 0.5rem 0;
    }
    
    .progress-fill {
        background: linear-gradient(90deg, #28a745, #20c997);
        height: 100%;
        transition: width 0.3s ease;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def display_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Enhanced Multi-Engine OCR System</h1>
        <p>Powered by EasyOCR, PaddleOCR & Tesseract | Simultaneous Processing for Optimal Results</p>
    </div>
    """, unsafe_allow_html=True)

def display_engine_status(result: Dict[str, Any]):
    """Display status of all OCR engines"""
    st.subheader("🔧 Engine Performance")
    
    if 'results' in result:
        cols = st.columns(len(result['results']))
        
        for i, (engine_name, engine_result) in enumerate(result['results'].items()):
            with cols[i]:
                if engine_result['status'] == 'success':
                    st.success(f"✅ {engine_name.upper()}")
                    st.metric(
                        label="Confidence", 
                        value=f"{engine_result['confidence']:.2f}",
                        delta=f"{engine_result['processing_time']:.2f}s"
                    )
                else:
                    st.error(f"❌ {engine_name.upper()}")
                    st.caption(f"Error: {engine_result.get('error', 'Unknown error')}")

def display_image_quality(quality_data: Dict[str, float]):
    """Display image quality assessment"""
    st.subheader("📊 Image Quality Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Overall Score", f"{quality_data.get('overall_score', 0):.2f}")
        st.metric("Sharpness", f"{quality_data.get('sharpness', 0):.2f}")
    
    with col2:
        st.metric("Contrast", f"{quality_data.get('contrast', 0):.2f}")
        st.metric("Brightness", f"{quality_data.get('brightness', 0):.2f}")
    
    # Progress bar for overall quality
    quality_score = quality_data.get('overall_score', 0)
    st.markdown(f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {quality_score * 100}%"></div>
    </div>
    <p style="text-align: center; margin: 0;">Quality Score: {quality_score:.2f}/1.00</p>
    """, unsafe_allow_html=True)

def display_results_comparison(result: Dict[str, Any]):
    """Display comparison of results from different engines"""
    if 'results' not in result:
        return
    
    st.subheader("🔍 Engine Results Comparison")
    
    for engine_name, engine_result in result['results'].items():
        if engine_result['status'] == 'success' and engine_result['text'].strip():
            with st.expander(f"{engine_name.upper()} - Confidence: {engine_result['confidence']:.2f}"):
                st.text_area(
                    f"Text from {engine_name}",
                    engine_result['text'],
                    height=100,
                    key=f"result_{engine_name}"
                )
                st.caption(f"Processing time: {engine_result['processing_time']:.2f} seconds")

def process_image_with_ocr(image: Image.Image) -> Dict[str, Any]:
    """Process image with enhanced multi-engine OCR"""
    if not ENHANCED_OCR_AVAILABLE:
        return {
            'best_result': {
                'text': 'Enhanced OCR system not available',
                'confidence': 0.0,
                'engine': 'none'
            },
            'processing_time': 0.0,
            'results': {},
            'image_quality': {'overall_score': 0.0}
        }
    
    try:
        with st.spinner('🔄 Processing with multiple OCR engines...'):
            result = extract_text_enhanced_multi_engine(image)
        return result
    except Exception as e:
        st.error(f"OCR processing failed: {e}")
        return {
            'best_result': {
                'text': f'Error: {str(e)}',
                'confidence': 0.0,
                'engine': 'error'
            },
            'processing_time': 0.0,
            'results': {},
            'image_quality': {'overall_score': 0.0}
        }

def main():
    """Main application function"""
    st.set_page_config(
        page_title="Enhanced Multi-Engine OCR",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_css()
    display_header()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OCR Engine Status
        st.subheader("🔧 Available Engines")
        if ENHANCED_OCR_AVAILABLE:
            st.success("✅ Enhanced Multi-Engine OCR")
            st.info("• EasyOCR\n• PaddleOCR\n• Tesseract (fallback)")
        else:
            st.error("❌ Enhanced OCR Not Available")
        
        # Processing options
        st.subheader("📋 Options")
        show_details = st.checkbox("Show detailed results", value=True)
        show_comparison = st.checkbox("Show engine comparison", value=True)
        export_results = st.checkbox("Enable result export", value=False)
        
        # Help section
        with st.expander("❓ Help"):
            st.markdown("""
            **How to use:**
            1. Upload an image using the file uploader
            2. Or select a sample image
            3. The system will process with multiple OCR engines simultaneously
            4. View the best result and compare engine performance
            
            **Supported formats:** PNG, JPG, JPEG, WEBP, BMP
            
            **Features:**
            - Simultaneous multi-engine processing
            - Image quality assessment
            - Handwriting detection
            - Confidence scoring
            """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Image Input")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'webp', 'bmp'],
            help="Upload an image for OCR processing"
        )
        
        # Sample images
        sample_images = {
            "Simple Test": "simple_test_image.jpg",
            "Complex Document": "complex_test_image.jpg",
            "Basic Test": "test_image.png"
        }
        
        st.write("Or select a sample image:")
        selected_sample = st.selectbox(
            "Sample Images",
            options=["None"] + list(sample_images.keys())
        )
        
        # Process selected image
        image_to_process = None
        
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file)
            st.image(image_to_process, caption="Uploaded Image", use_column_width=True)
            
        elif selected_sample != "None":
            sample_path = sample_images[selected_sample]
            if os.path.exists(sample_path):
                image_to_process = Image.open(sample_path)
                st.image(image_to_process, caption=f"Sample: {selected_sample}", use_column_width=True)
            else:
                st.warning(f"Sample image not found: {sample_path}")
    
    with col2:
        st.subheader("📝 OCR Results")
        
        if image_to_process is not None:
            # Process the image
            start_time = time.time()
            result = process_image_with_ocr(image_to_process)
            total_time = time.time() - start_time
            
            # Display best result
            st.success(f"🏆 Best Result (Engine: {result['best_result']['engine'].upper()})")
            
            best_text = result['best_result']['text']
            confidence = result['best_result']['confidence']
            
            st.text_area(
                f"Extracted Text (Confidence: {confidence:.2f})",
                best_text,
                height=200,
                key="main_result"
            )
            
            # Processing metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Confidence", f"{confidence:.2f}")
            with col_b:
                st.metric("Processing Time", f"{total_time:.2f}s")
            with col_c:
                handwriting = result.get('is_handwritten', False)
                st.metric("Handwriting", "Yes" if handwriting else "No")
            
            # Export functionality
            if export_results and best_text.strip():
                export_data = {
                    'text': best_text,
                    'confidence': confidence,
                    'engine': result['best_result']['engine'],
                    'processing_time': total_time,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"ocr_results_{int(time.time())}.json",
                    mime="application/json"
                )
                
                st.download_button(
                    label="📄 Download Text",
                    data=best_text,
                    file_name=f"extracted_text_{int(time.time())}.txt",
                    mime="text/plain"
                )
        else:
            st.info("👆 Please upload an image or select a sample to begin OCR processing")
    
    # Detailed results section
    if image_to_process is not None and show_details:
        st.markdown("---")
        
        # Engine performance
        display_engine_status(result)
        
        # Image quality
        if 'image_quality' in result:
            display_image_quality(result['image_quality'])
        
        # Engine comparison
        if show_comparison:
            display_results_comparison(result)

if __name__ == "__main__":
    main()
