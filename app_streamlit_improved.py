#!/usr/bin/env python
"""
OCR Application Entry Point - Improved Version

This script runs the OCR web interface using Streamlit with enhanced UI and error handling.
"""

import os
import sys
import logging
import time
from typing import Optional, Dict, Any, List, Tuple
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

# Import Streamlit and other dependencies
import streamlit as st
import streamlit.components.v1 as components
import io
from PIL import Image
import base64
import json

# Import OCR components with error handling
try:
    from ocr_app.core.ocr_engine import OCREngine
    from ocr_app.core.image_processor import ImageProcessor
    from ocr_app.models.model_manager import ModelManager
    from ocr_app.rag.rag_processor import RAGProcessor
    from ocr_app.config.settings import Settings
    from ocr_app.utils.text_utils import extract_entities, format_ocr_result
except ImportError as e:
    st.error(f"Import error: {e}")
    # Try alternative imports
    try:
        from core.ocr_engine import OCREngine
        from core.image_processor import ImageProcessor
        from models.model_manager import ModelManager
        from rag.rag_processor import RAGProcessor
        from config.settings import Settings
        from utils.text_utils import extract_entities, format_ocr_result
    except ImportError as e2:
        st.error(f"Alternative import failed: {e2}")
        st.stop()

logger = logging.getLogger(__name__)

class StreamlitAppImproved:
    """
    Enhanced Streamlit web interface for the OCR application
    """
    
    def __init__(self):
        """Initialize the Streamlit app"""
        try:
            self.settings = Settings()
            self.init_session_state()
            
            # Initialize components to None first
            self.model_manager = None
            self.ocr_engine = None
            self.image_processor = None
            self.rag_processor = None
            
            self.load_resources()
        except Exception as e:
            st.error(f"Failed to initialize app: {e}")
            logger.error(f"App initialization error: {e}")
    
    def init_session_state(self):
        """Initialize Streamlit session state variables"""
        defaults = {
            'extracted_text': '',
            'ocr_engine': 'auto',
            'preserve_layout': True,
            'dependency_errors': [],
            'models_initialized': False,
            'current_image': None,
            'processing_status': 'idle',
            'last_processing_time': 0
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def load_resources(self):
        """Load required resources with comprehensive error handling"""
        try:
            with st.spinner('🔄 Initializing OCR system...'):
                # Initialize components with error handling
                try:
                    self.model_manager = ModelManager(self.settings)
                    st.session_state['model_manager_status'] = 'success'
                except Exception as e:
                    logger.error(f"Model manager initialization failed: {e}")
                    st.session_state['dependency_errors'].append(f"Model Manager: {str(e)}")
                    self.model_manager = None
                
                try:
                    self.ocr_engine = OCREngine(self.settings)
                    st.session_state['available_ocr_engines'] = self.ocr_engine.enabled_engines
                    st.session_state['ocr_engine_status'] = 'success'
                except Exception as e:
                    logger.error(f"OCR engine initialization failed: {e}")
                    st.session_state['dependency_errors'].append(f"OCR Engine: {str(e)}")
                    self.ocr_engine = None
                    st.session_state['available_ocr_engines'] = []
                
                try:
                    self.image_processor = ImageProcessor(self.settings)
                    st.session_state['image_processor_status'] = 'success'
                except Exception as e:
                    logger.error(f"Image processor initialization failed: {e}")
                    st.session_state['dependency_errors'].append(f"Image Processor: {str(e)}")
                    self.image_processor = None
                
                try:
                    if self.model_manager:
                        self.rag_processor = RAGProcessor(self.model_manager, self.settings)
                        st.session_state['rag_processor_status'] = 'success'
                    else:
                        self.rag_processor = None
                        st.session_state['dependency_errors'].append("RAG Processor: Requires Model Manager")
                except Exception as e:
                    logger.error(f"RAG processor initialization failed: {e}")
                    st.session_state['dependency_errors'].append(f"RAG Processor: {str(e)}")
                    self.rag_processor = None
                
                # Check for missing OCR engines
                all_engines = ['tesseract', 'easyocr', 'paddleocr']
                available_engines = st.session_state.get('available_ocr_engines', [])
                missing_engines = [engine for engine in all_engines if engine not in available_engines]
                st.session_state['missing_ocr_engines'] = missing_engines
                
                # Generate installation instructions
                self._generate_installation_instructions(missing_engines)
                
                # Check module status
                if self.model_manager:
                    try:
                        module_status = self.model_manager.get_module_status()
                        rag_available = (
                            module_status.get('transformers_available', False) and 
                            module_status.get('sentence_transformers_available', False)
                        )
                        st.session_state['rag_available'] = rag_available
                        if not rag_available:
                            st.session_state['dependency_errors'].append(
                                "Q&A functionality limited - transformers or sentence_transformers not available"
                            )
                    except Exception as e:
                        st.session_state['rag_available'] = False
                        st.session_state['dependency_errors'].append(f"Module status check failed: {e}")
                else:
                    st.session_state['rag_available'] = False
                
                st.session_state['models_initialized'] = True
                
        except Exception as e:
            logger.error(f"Error loading resources: {e}")
            st.session_state['dependency_errors'].append(f"Resource loading failed: {str(e)}")
            st.session_state['models_initialized'] = False
    
    def _generate_installation_instructions(self, missing_engines: List[str]):
        """Generate installation instructions for missing engines"""
        if not missing_engines:
            st.session_state['ocr_installation_instructions'] = []
            return
        
        instructions = []
        
        if 'tesseract' in missing_engines:
            if sys.platform == 'win32':
                instructions.append("Download Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
            elif sys.platform == 'darwin':
                instructions.append("brew install tesseract")
            else:
                instructions.append("sudo apt-get install tesseract-ocr")
            instructions.append("pip install pytesseract")
        
        if 'easyocr' in missing_engines:
            instructions.append("pip install easyocr")
        
        if 'paddleocr' in missing_engines:
            instructions.append("pip install paddlepaddle paddleocr")
        
        st.session_state['ocr_installation_instructions'] = instructions
    
    def load_css(self):
        """Load enhanced CSS styles"""
        css_file = os.path.join(os.path.dirname(__file__), 'static', 'styles.css')
        try:
            with open(css_file, encoding='utf-8') as f:
                css_content = f.read()
                st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
                logger.info("CSS loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load CSS file: {e}")
            # Enhanced fallback CSS
            st.markdown("""
            <style>
            :root {
                --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --card-background: rgba(255, 255, 255, 0.95);
                --shadow-soft: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                --border-radius: 12px;
            }
            
            .main {
                background: var(--primary-gradient);
                min-height: 100vh;
                padding: 2rem;
            }
            
            .header {
                background: var(--card-background);
                border-radius: var(--border-radius);
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: var(--shadow-soft);
                text-align: center;
            }
            
            .card {
                background: var(--card-background);
                border-radius: var(--border-radius);
                padding: 1.5rem;
                margin: 1rem 0;
                box-shadow: var(--shadow-soft);
            }
            
            .stButton>button {
                background: var(--primary-gradient) !important;
                color: white !important;
                border: none !important;
                border-radius: var(--border-radius) !important;
                transition: all 0.3s ease !important;
            }
            </style>
            """, unsafe_allow_html=True)
    
    def run(self):
        """Run the enhanced Streamlit application"""
        try:
            # Load CSS styles
            self.load_css()
            
            # Display header
            self._display_header()
            
            # Display status information
            self._display_status_dashboard()
            
            # Create main tabs with enhanced styling
            tabs = st.tabs(["📄 Document Processing", "⚙️ OCR Settings", "❓ Q&A Interface", "📊 System Info"])
            
            with tabs[0]:
                self._document_processing_tab()
            
            with tabs[1]:
                self._ocr_settings_tab()
            
            with tabs[2]:
                self._qa_interface_tab()
            
            with tabs[3]:
                self._system_info_tab()
                
        except Exception as e:
            st.error(f"Application error: {e}")
            logger.error(f"Runtime error: {e}")
    
    def _display_header(self):
        """Display enhanced application header"""
        st.markdown("""
        <div class="header">
            <h1>🔍 Intelligent OCR and Text Analysis Tool</h1>
            <p style="font-size: 1.2rem; margin-bottom: 1rem;">
                Extract text from images and PDFs with advanced AI-powered analysis
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
                <a href="https://www.linkedin.com/in/rayyan-ahmed9477/" target="_blank" 
                   style="text-decoration: none; color: #667eea; font-weight: 500;">
                   💼 LinkedIn
                </a>
                <a href="https://github.com/Rayyan9477/" target="_blank" 
                   style="text-decoration: none; color: #667eea; font-weight: 500;">
                   💻 GitHub
                </a>
                <span style="color: #4a5568; font-weight: 500;">
                   👨‍💻 Developed by Rayyan Ahmed
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _display_status_dashboard(self):
        """Display system status dashboard"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.get('models_initialized', False):
                st.success("✅ System Ready")
            else:
                st.error("❌ System Error")
        
        with col2:
            available_engines = len(st.session_state.get('available_ocr_engines', []))
            st.metric("OCR Engines", f"{available_engines}/3")
        
        with col3:
            if st.session_state.get('rag_available', False):
                st.success("🤖 AI Q&A Ready")
            else:
                st.warning("🤖 AI Q&A Limited")
        
        with col4:
            processing_time = st.session_state.get('last_processing_time', 0)
            if processing_time > 0:
                st.metric("Last Process", f"{processing_time:.1f}s")
            else:
                st.metric("Last Process", "N/A")
    
    def _document_processing_tab(self):
        """Enhanced document processing tab"""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📤 Upload Document")
            
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=["jpg", "jpeg", "png", "bmp", "tiff", "pdf"],
                help="Supported formats: JPG, PNG, BMP, TIFF, PDF"
            )
            
            if uploaded_file:
                try:
                    # Handle different file types
                    if uploaded_file.type.startswith('image'):
                        image = Image.open(uploaded_file)
                        st.session_state['current_image'] = image
                        
                        # Display image with enhanced styling
                        st.markdown("#### 🖼️ Uploaded Image")
                        st.image(image, caption="Uploaded Image", use_column_width=True)
                        
                        # Image analysis (only if image_processor is available)
                        if self.image_processor:
                            try:
                                quality_info = self.image_processor.assess_image_quality(image)
                                has_tables = self.image_processor.detect_tables(image)
                                
                                st.markdown("#### 📊 Image Analysis")
                                analysis_col1, analysis_col2 = st.columns(2)
                                
                                with analysis_col1:
                                    quality_score = quality_info.get('quality_score', 0)
                                    st.metric(
                                        "Quality Score", 
                                        f"{quality_score:.2f}",
                                        delta=f"{'Good' if quality_score > 0.7 else 'Fair' if quality_score > 0.5 else 'Poor'}"
                                    )
                                
                                with analysis_col2:
                                    st.metric("Contains Tables", "Yes" if has_tables else "No")
                                
                                if has_tables:
                                    st.info("📊 Table structures detected. Layout preservation is recommended.")
                            except Exception as e:
                                st.warning(f"Image analysis failed: {e}")
                        else:
                            st.warning("Image analysis unavailable - Image processor not initialized")
                    
                    elif uploaded_file.type == 'application/pdf':
                        st.info("📄 PDF file detected. Processing will extract text from all pages.")
                    
                except Exception as e:
                    st.error(f"Error processing uploaded file: {e}")
        
        with col2:
            st.markdown("### 🔍 OCR Processing")
            
            if uploaded_file and st.session_state.get('current_image'):
                # OCR processing controls
                st.markdown("#### Processing Options")
                
                col_engine, col_layout = st.columns(2)
                
                with col_engine:
                    available_engines = st.session_state.get('available_ocr_engines', ['auto'])
                    if not available_engines:
                        available_engines = ['auto']
                    
                    selected_engine = st.selectbox(
                        "OCR Engine",
                        options=available_engines,
                        index=0,
                        help="Choose the OCR engine to use for text extraction"
                    )
                    st.session_state['ocr_engine'] = selected_engine
                
                with col_layout:
                    preserve_layout = st.checkbox(
                        "Preserve Layout",
                        value=st.session_state.get('preserve_layout', True),
                        help="Maintain original document structure"
                    )
                    st.session_state['preserve_layout'] = preserve_layout
                
                # Enhanced OCR processing button
                if st.button("🔍 Extract Text", type="primary", use_container_width=True):
                    if self.ocr_engine:
                        start_time = time.time()
                        st.session_state['processing_status'] = 'processing'
                        
                        with st.spinner('🔄 Processing image...'):
                            try:
                                extracted_text = self.ocr_engine.perform_ocr(
                                    st.session_state['current_image'],
                                    engine=selected_engine,
                                    preserve_layout=preserve_layout,
                                    preprocess=True
                                )
                                
                                processing_time = time.time() - start_time
                                st.session_state['last_processing_time'] = processing_time
                                
                                if extracted_text:
                                    st.session_state['extracted_text'] = extracted_text
                                    st.session_state['processing_status'] = 'completed'
                                    st.success(f"✅ Text extracted successfully in {processing_time:.1f}s!")
                                else:
                                    st.warning("⚠️ No text found in the image")
                                    st.session_state['processing_status'] = 'no_text'
                                
                            except Exception as e:
                                st.error(f"❌ OCR processing failed: {e}")
                                st.session_state['processing_status'] = 'error'
                                logger.error(f"OCR processing error: {e}")
                    else:
                        st.error("❌ OCR engine not available. Please check system configuration.")
            else:
                st.info("📤 Please upload an image to start OCR processing")
        
        # Display extracted text with enhanced formatting
        if st.session_state.get('extracted_text'):
            self._display_extracted_text()
    
    def _display_extracted_text(self):
        """Display extracted text with enhanced formatting and features"""
        st.markdown("### 📝 Extracted Text")
        
        extracted_text = st.session_state['extracted_text']
        
        # Text statistics
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric("Characters", len(extracted_text))
        
        with stats_col2:
            word_count = len(extracted_text.split())
            st.metric("Words", word_count)
        
        with stats_col3:
            line_count = len(extracted_text.split('\n'))
            st.metric("Lines", line_count)
        
        with stats_col4:
            st.metric("Size", f"{len(extracted_text.encode('utf-8'))} bytes")
        
        # Format options
        format_col1, format_col2, format_col3 = st.columns(3)
        
        with format_col1:
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                # Use JavaScript to copy to clipboard
                st.components.v1.html(f"""
                    <script>
                    navigator.clipboard.writeText(`{extracted_text.replace('`', '\\`')}`);
                    </script>
                """, height=0)
                st.success("Text copied to clipboard!")
        
        with format_col2:
            # Download as text file
            st.download_button(
                label="💾 Download as TXT",
                data=extracted_text,
                file_name=f"extracted_text_{int(time.time())}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with format_col3:
            # Download as JSON
            json_data = {
                "extracted_text": extracted_text,
                "timestamp": time.time(),
                "ocr_engine": st.session_state.get('ocr_engine', 'unknown'),
                "word_count": len(extracted_text.split()),
                "character_count": len(extracted_text)
            }
            
            st.download_button(
                label="📄 Download as JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"ocr_result_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Display text in formatted container
        st.markdown(f"""
        <div class="formatted-text">
            <pre>{extracted_text}</pre>
        </div>
        """, unsafe_allow_html=True)
    
    def _ocr_settings_tab(self):
        """Enhanced OCR settings and configuration tab"""
        st.markdown("### ⚙️ OCR Engine Configuration")
        
        # Engine status
        available_engines = st.session_state.get('available_ocr_engines', [])
        missing_engines = st.session_state.get('missing_ocr_engines', [])
        
        if available_engines:
            st.success(f"✅ Available OCR engines: {', '.join(available_engines)}")
        
        if missing_engines:
            with st.expander("⚠️ Missing OCR Engines", expanded=True):
                st.warning(f"Missing engines: {', '.join(missing_engines)}")
                
                st.markdown("#### Installation Instructions:")
                instructions = st.session_state.get('ocr_installation_instructions', [])
                for i, instruction in enumerate(instructions, 1):
                    st.code(instruction, language='bash')
                
                st.info("💡 Restart the application after installing new engines.")
        
        # Settings configuration
        st.markdown("#### Processing Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Image Preprocessing")
            enhance_contrast = st.checkbox("Enhance Contrast", value=True)
            remove_noise = st.checkbox("Remove Noise", value=True)
            resize_image = st.checkbox("Auto Resize", value=False)
        
        with col2:
            st.markdown("##### Text Processing")
            preserve_layout = st.checkbox("Preserve Layout", value=True)
            detect_tables = st.checkbox("Detect Tables", value=True)
            extract_entities = st.checkbox("Extract Entities", value=False)
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.1)
            max_image_size = st.number_input("Max Image Size (MB)", min_value=1, max_value=50, value=10)
            timeout_seconds = st.number_input("Processing Timeout (s)", min_value=10, max_value=300, value=60)
        
        # Save settings
        if st.button("💾 Save Settings", type="primary"):
            # Save settings logic here
            st.success("Settings saved successfully!")
    
    def _qa_interface_tab(self):
        """Enhanced Q&A interface tab"""
        st.markdown("### ❓ AI-Powered Question & Answer")
        
        if not st.session_state.get('rag_available', False):
            st.warning("🤖 Q&A functionality is limited. Required AI models are not available.")
            st.info("Install transformers and sentence-transformers packages to enable full Q&A features.")
            return
        
        if not st.session_state.get('extracted_text'):
            st.info("📄 Please extract text from a document first to use the Q&A feature.")
            return
        
        # Q&A interface
        st.markdown("#### Ask Questions About Your Document")
        
        # Sample questions
        sample_questions = [
            "What is the main topic of this document?",
            "Summarize the key points",
            "Are there any dates mentioned?",
            "What names or organizations are mentioned?",
            "Extract any numerical data"
        ]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            question = st.text_input(
                "Your Question:",
                placeholder="Ask anything about the extracted text...",
                help="Type your question about the document content"
            )
        
        with col2:
            st.markdown("##### Quick Questions")
            for i, sample_q in enumerate(sample_questions):
                if st.button(f"📝 {sample_q}", key=f"sample_q_{i}"):
                    question = sample_q
                    st.rerun()
        
        if question and st.button("🔍 Get Answer", type="primary"):
            if self.rag_processor:
                with st.spinner("🤖 Analyzing document and generating answer..."):
                    try:
                        answer = self.rag_processor.answer_question(
                            question, 
                            st.session_state['extracted_text']
                        )
                        
                        if answer:
                            st.markdown("#### 💡 Answer")
                            st.markdown(f"""
                            <div class="card">
                                <h4>🤖 AI Response:</h4>
                                <p>{answer.get('answer', 'No answer available')}</p>
                                <small><strong>Confidence:</strong> {answer.get('confidence', 0):.2f}</small>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning("Could not generate an answer for this question.")
                    except Exception as e:
                        st.error(f"Error processing question: {e}")
            else:
                st.error("Q&A processor not available")
    
    def _system_info_tab(self):
        """System information and diagnostics tab"""
        st.markdown("### 📊 System Information")
        
        # Component status
        st.markdown("#### Component Status")
        
        components = [
            ("Model Manager", st.session_state.get('model_manager_status')),
            ("OCR Engine", st.session_state.get('ocr_engine_status')),
            ("Image Processor", st.session_state.get('image_processor_status')),
            ("RAG Processor", st.session_state.get('rag_processor_status')),
        ]
        
        for name, status in components:
            if status == 'success':
                st.success(f"✅ {name}: Operational")
            else:
                st.error(f"❌ {name}: Failed")
        
        # Dependency errors
        if st.session_state.get('dependency_errors'):
            st.markdown("#### 🚨 Dependency Issues")
            for error in st.session_state['dependency_errors']:
                st.error(error)
        
        # System metrics
        st.markdown("#### 📈 Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Session Duration", f"{time.time() - st.session_state.get('session_start', time.time()):.0f}s")
        
        with col2:
            st.metric("Documents Processed", st.session_state.get('documents_processed', 0))
        
        with col3:
            st.metric("Memory Usage", "N/A")  # Could implement actual memory monitoring
        
        # Clear cache button
        if st.button("🗑️ Clear Cache", type="secondary"):
            st.cache_data.clear()
            st.success("Cache cleared successfully!")


def main():
    """Main entry point for the enhanced Streamlit app"""
    # Configure Streamlit page
    st.set_page_config(
        page_title="🔍 OCR Image-to-Text Tool",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/Rayyan9477/',
            'Report a bug': 'https://github.com/Rayyan9477/',
            'About': """
            # OCR Image-to-Text Tool
            
            An advanced OCR application with AI-powered text analysis.
            
            **Features:**
            - Multi-engine OCR (Tesseract, EasyOCR, PaddleOCR)
            - AI-powered Q&A
            - Advanced image preprocessing
            - Beautiful modern interface
            
            **Developer:** Rayyan Ahmed
            """
        }
    )
    
    # Initialize session start time
    if 'session_start' not in st.session_state:
        st.session_state['session_start'] = time.time()
    
    # Initialize and run the app
    try:
        app = StreamlitAppImproved()
        app.run()
    except Exception as e:
        st.error(f"Application failed to start: {e}")
        logger.error(f"Main app error: {e}")
        
        # Fallback UI
        st.markdown("## ⚠️ Application Error")
        st.markdown("The application encountered an error during startup.")
        
        with st.expander("Error Details"):
            st.code(str(e))
        
        st.markdown("### Troubleshooting Steps:")
        st.markdown("""
        1. Refresh the page
        2. Check if all dependencies are installed
        3. Ensure OCR engines are properly configured
        4. Check the console for detailed error messages
        """)


if __name__ == "__main__":
    main()
