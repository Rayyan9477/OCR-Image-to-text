#!/usr/bin/env python
"""
OCR Application with Multi-Engine Support
Optimized version using EasyOCR, PaddleOCR, and TrOCR simultaneously
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

# Import multi-engine system
try:
    from multi_engine_ocr import get_multi_ocr, extract_text_multi_engine
    from enhanced_image_processor import get_image_processor
    MULTI_ENGINE_AVAILABLE = True
except ImportError as e:
    MULTI_ENGINE_AVAILABLE = False
    logging.warning(f"Multi-engine OCR system not available: {e}")

# Fallback imports
try:
    from ocr_app.utils.text_utils import extract_entities, format_ocr_result
except ImportError:
    def extract_entities(text):
        return {"emails": [], "phones": [], "dates": [], "numbers": []}
    
    def format_ocr_result(text, format_type):
        return text

logger = logging.getLogger(__name__)

class MultiEngineOCRApp:
    """Streamlit app with multi-engine OCR support"""
    
    def __init__(self):
        self.init_session_state()
        self.multi_ocr = None
        self.image_processor = None
        self.load_resources()
    
    def init_session_state(self):
        """Initialize Streamlit session state"""
        defaults = {
            'extracted_text': '',
            'ocr_result_details': {},
            'multi_engine_available': False,
            'available_engines': [],
            'processing_history': []
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def load_resources(self):
        """Load multi-engine OCR resources"""
        try:
            if MULTI_ENGINE_AVAILABLE:
                with st.spinner('🚀 Initializing Multi-Engine OCR System...'):
                    self.multi_ocr = get_multi_ocr()
                    self.image_processor = get_image_processor()
                    
                    available_engines = list(self.multi_ocr.engines.keys())
                    st.session_state['multi_engine_available'] = True
                    st.session_state['available_engines'] = available_engines
                    
                    logger.info(f"✅ Multi-Engine OCR initialized: {available_engines}")
            else:
                st.session_state['multi_engine_available'] = False
                st.session_state['available_engines'] = []
                logger.warning("❌ Multi-engine OCR not available")
                
        except Exception as e:
            logger.error(f"Failed to load resources: {e}")
            st.session_state['multi_engine_available'] = False
    
    def load_css(self):
        """Load enhanced CSS styles"""
        css_file = os.path.join(current_dir, 'static', 'styles.css')
        try:
            with open(css_file, encoding='utf-8') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        except Exception as e:
            logger.warning(f"CSS loading failed: {e}")
            # Enhanced fallback CSS
            st.markdown("""
            <style>
            .main-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 2rem;
            }
            
            .metric-card {
                background: white;
                border-radius: 10px;
                padding: 1rem;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                border-left: 4px solid #667eea;
            }
            
            .engine-status {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
            }
            
            .result-container {
                background: white;
                border-radius: 10px;
                padding: 1.5rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                margin: 1rem 0;
            }
            
            .processing-details {
                background: #e3f2fd;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
            }
            </style>
            """, unsafe_allow_html=True)
    
    def run(self):
        """Run the Streamlit application"""
        # Page configuration
        st.set_page_config(
            page_title="Multi-Engine OCR Tool",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Load styles
        self.load_css()
        
        # Header
        self.display_header()
        
        # Engine status
        self.display_engine_status()
        
        # Main interface
        if st.session_state['multi_engine_available']:
            self.multi_engine_interface()
        else:
            self.fallback_interface()
    
    def display_header(self):
        """Display application header"""
        st.markdown("""
        <div class="main-header">
            <h1>🔍 Multi-Engine OCR System</h1>
            <p>Simultaneous processing with EasyOCR, PaddleOCR, and TrOCR for optimal results</p>
            <p><strong>Advanced AI-Powered Text Extraction</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    def display_engine_status(self):
        """Display OCR engine status"""
        available_engines = st.session_state.get('available_engines', [])
        
        if available_engines:
            st.markdown("""
            <div class="engine-status">
                <h3>🎯 OCR Engine Status</h3>
            </div>
            """, unsafe_allow_html=True)
            
            cols = st.columns(len(available_engines))
            engine_info = {
                'easyocr': {'name': 'EasyOCR', 'icon': '🌍', 'specialty': 'Multi-language'},
                'paddleocr': {'name': 'PaddleOCR', 'icon': '🚀', 'specialty': 'Fast & Accurate'},
                'trocr': {'name': 'TrOCR', 'icon': '✍️', 'specialty': 'Handwriting'}
            }
            
            for idx, engine in enumerate(available_engines):
                with cols[idx]:
                    info = engine_info.get(engine, {'name': engine.upper(), 'icon': '🔧', 'specialty': 'General'})
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{info['icon']} {info['name']}</h4>
                        <p>{info['specialty']}</p>
                        <span style="color: green; font-weight: bold;">✅ Active</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("❌ No OCR engines available. Please install dependencies.")
    
    def multi_engine_interface(self):
        """Main multi-engine OCR interface"""
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📤 Upload Image")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
                help="Supported: JPG, PNG, BMP, TIFF, WebP"
            )
            
            if uploaded_file:
                # Display image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Image analysis
                if self.image_processor:
                    with st.expander("📊 Image Analysis", expanded=True):
                        try:
                            quality_info = self.image_processor.assess_image_quality(image)
                            has_tables = self.image_processor.detect_tables(image)
                            
                            # Quality metrics
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                quality_score = quality_info.get('quality_score', 0)
                                st.metric("Quality Score", f"{quality_score:.2f}")
                            with col_b:
                                st.metric("Tables Detected", "Yes" if has_tables else "No")
                            with col_c:
                                dimensions = quality_info.get('dimensions', (0, 0))
                                st.metric("Resolution", f"{dimensions[0]}x{dimensions[1]}")
                            
                            # Recommendations
                            if 'recommendations' in quality_info:
                                st.markdown("**💡 Recommendations:**")
                                for rec in quality_info['recommendations']:
                                    st.write(f"• {rec}")
                                    
                        except Exception as e:
                            st.warning(f"Image analysis failed: {e}")
                
                # OCR Processing
                st.markdown("### 🔍 Text Extraction")
                
                # Processing options
                col_opt1, col_opt2 = st.columns(2)
                with col_opt1:
                    timeout = st.slider("Timeout (seconds)", 10, 120, 30)
                with col_opt2:
                    enhance_image = st.checkbox("Enhance image before OCR", value=True)
                
                # Main processing button
                if st.button("🚀 Extract Text (All Engines)", type="primary", use_container_width=True):
                    self.process_with_multi_engine(image, timeout, enhance_image)
        
        with col2:
            st.markdown("### 📄 Extraction Results")
            
            # Display results
            if st.session_state.get('extracted_text'):
                self.display_results()
            else:
                st.info("Upload an image and click 'Extract Text' to see results here.")
                
                # Show processing history
                if st.session_state.get('processing_history'):
                    with st.expander("📈 Processing History"):
                        for idx, entry in enumerate(st.session_state['processing_history'][-5:]):
                            st.write(f"**{idx+1}.** {entry['timestamp']} - "
                                   f"Confidence: {entry['confidence']:.2f} - "
                                   f"Engines: {', '.join(entry['engines'])}")
    
    def process_with_multi_engine(self, image: Image.Image, timeout: int, enhance: bool):
        """Process image with all available engines"""
        try:
            with st.spinner('🔄 Processing with multiple OCR engines...'):
                # Enhance image if requested
                if enhance and self.image_processor:
                    image = self.image_processor.enhance_for_ocr(image)
                
                # Extract text using multi-engine approach
                result = extract_text_multi_engine(image)
                
                if result['text'].strip():
                    # Store results
                    st.session_state['extracted_text'] = result['text']
                    st.session_state['ocr_result_details'] = result
                    
                    # Add to history
                    history_entry = {
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'confidence': result['confidence'],
                        'engines': result['engines_used'],
                        'processing_time': result['processing_time']
                    }
                    
                    if 'processing_history' not in st.session_state:
                        st.session_state['processing_history'] = []
                    st.session_state['processing_history'].append(history_entry)
                    
                    # Success message with details
                    st.success(f"✅ Text extracted! Confidence: {result['confidence']:.2f}, "
                             f"Time: {result['processing_time']:.2f}s, "
                             f"Engines: {', '.join(result['engines_used'])}")
                    
                    # Rerun to update display
                    st.rerun()
                else:
                    st.error("❌ No text could be extracted from the image.")
                    
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            logger.error(f"Multi-engine processing error: {e}")
    
    def display_results(self):
        """Display OCR results with detailed analysis"""
        text = st.session_state['extracted_text']
        details = st.session_state.get('ocr_result_details', {})
        
        # Main text display
        st.text_area(
            "Extracted Text:",
            value=text,
            height=300,
            help="Copy this text or download it below"
        )
        
        # Processing details
        with st.expander("🔍 Processing Details", expanded=True):
            if details:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Confidence", f"{details.get('confidence', 0):.2f}")
                with col2:
                    st.metric("Processing Time", f"{details.get('processing_time', 0):.2f}s")
                with col3:
                    st.metric("Engines Used", len(details.get('engines_used', [])))
                with col4:
                    handwritten = details.get('is_handwritten', False)
                    st.metric("Handwritten", "Yes" if handwritten else "No")
                
                # Individual engine results
                if 'individual_results' in details:
                    st.markdown("**Individual Engine Results:**")
                    for engine_result in details['individual_results']:
                        if engine_result['text'].strip():
                            confidence = engine_result['confidence']
                            time_taken = engine_result['time']
                            engine_name = engine_result['engine'].upper()
                            
                            # Color code by confidence
                            color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                            st.markdown(f"**{engine_name}**: "
                                      f"<span style='color: {color}'>Confidence {confidence:.2f}</span>, "
                                      f"Time {time_taken:.2f}s", 
                                      unsafe_allow_html=True)
        
        # Text statistics
        with st.expander("📊 Text Statistics"):
            words = len(text.split())
            chars = len(text)
            lines = len(text.split('\n'))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Words", words)
            with col2:
                st.metric("Characters", chars)
            with col3:
                st.metric("Lines", lines)
        
        # Entity extraction
        try:
            entities = extract_entities(text)
            if any(entities.values()):
                with st.expander("🏷️ Detected Entities"):
                    for entity_type, values in entities.items():
                        if values:
                            st.write(f"**{entity_type.title()}**: {', '.join(values[:5])}")
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
        
        # Download options
        st.markdown("### 💾 Download Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📄 Download TXT",
                data=text,
                file_name="extracted_text.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            try:
                markdown_text = format_ocr_result(text, 'markdown')
                st.download_button(
                    "📝 Download Markdown",
                    data=markdown_text,
                    file_name="extracted_text.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            except:
                pass
        
        with col3:
            # JSON export with details
            import json
            export_data = {
                'text': text,
                'details': details,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            st.download_button(
                "📊 Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name="ocr_results.json",
                mime="application/json",
                use_container_width=True
            )
    
    def fallback_interface(self):
        """Fallback interface when multi-engine is not available"""
        st.markdown("---")
        st.error("❌ Multi-Engine OCR system is not available.")
        
        with st.expander("🔧 Installation Instructions"):
            st.markdown("""
            To enable the multi-engine OCR system, install the required packages:
            
            ```bash
            # Install OCR engines
            pip install easyocr paddlepaddle paddleocr
            
            # Install handwriting support (optional)
            pip install transformers torch
            
            # Install image processing
            pip install opencv-python pillow
            ```
            
            After installation, restart the application.
            """)
        
        st.info("Please install the required dependencies to use this application.")

def main():
    """Main entry point"""
    app = MultiEngineOCRApp()
    app.run()

if __name__ == "__main__":
    main()
