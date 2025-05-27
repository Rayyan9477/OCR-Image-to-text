#!/usr/bin/env python
"""
OCR Application Entry Point - Fixed Version

This script runs the OCR web interface using Streamlit with improved error handling and styling.
"""

import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add ocr_app package to path for imports
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'ocr_app'))

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables for consistent behavior
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # Use legacy Keras with TF
os.environ["KERAS_BACKEND"] = "tensorflow"  # Ensure TF backend

# Import Streamlit and other dependencies
import streamlit as st
import streamlit.components.v1 as components
import io
import time
from PIL import Image
import base64
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path

# Import OCR components
try:
    from ocr_app.core.ocr_engine import OCREngine
    from ocr_app.core.image_processor import ImageProcessor
    from ocr_app.models.model_manager import ModelManager
    from ocr_app.rag.rag_processor import RAGProcessor
    from ocr_app.config.settings import Settings
    from ocr_app.utils.text_utils import extract_entities, format_ocr_result
except ImportError as e:
    logger.error(f"OCR components import failed: {e}")
    st.session_state.setdefault('dependency_errors', []).append("OCR components import failed")

logger = logging.getLogger(__name__)

class StreamlitApp:
    """
    Streamlit web interface for the OCR application with improved error handling
    """
    
    def __init__(self):
        """Initialize the Streamlit app"""
        try:
            self.settings = Settings() if Settings else None
        except:
            self.settings = None
            
        self.init_session_state()
        
        # Initialize components to None first
        self.model_manager = None
        self.ocr_engine = None
        self.image_processor = None
        self.rag_processor = None
        
        self.load_resources()
    
    def init_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'extracted_text' not in st.session_state:
            st.session_state['extracted_text'] = ''
        
        if 'ocr_engine' not in st.session_state:
            st.session_state['ocr_engine'] = 'auto'
        
        if 'preserve_layout' not in st.session_state:
            st.session_state['preserve_layout'] = True
        
        if 'dependency_errors' not in st.session_state:
            st.session_state['dependency_errors'] = []
            
        if 'models_initialized' not in st.session_state:
            st.session_state['models_initialized'] = False
    
    def load_resources(self):
        """Load required resources and check dependencies"""
        try:
            # Check if all required classes are available
            if not all([OCREngine, ImageProcessor, ModelManager, RAGProcessor, Settings]):
                raise ImportError("Required OCR modules not available")
                
            # Initialize components
            with st.spinner('Initializing OCR system...'):
                self.model_manager = ModelManager(self.settings)
                self.ocr_engine = OCREngine(self.settings)
                self.image_processor = ImageProcessor(self.settings)
                self.rag_processor = RAGProcessor(self.model_manager, self.settings)
                
                # Store available OCR engines in session state
                st.session_state['available_ocr_engines'] = getattr(self.ocr_engine, 'enabled_engines', ['tesseract'])
                
                # Check for missing OCR engines
                all_engines = ['tesseract', 'easyocr', 'paddleocr']
                available_engines = st.session_state['available_ocr_engines']
                missing_engines = [engine for engine in all_engines if engine not in available_engines]
                st.session_state['missing_ocr_engines'] = missing_engines

                # Generate installation instructions for missing engines
                if missing_engines:
                    instructions = []
                    if 'tesseract' in missing_engines:
                        if sys.platform == 'win32':
                            instructions.append("Download and install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
                        elif sys.platform == 'darwin':  # macOS
                            instructions.append("brew install tesseract")
                        else:  # Linux
                            instructions.append("sudo apt-get install tesseract-ocr")
                        instructions.append("pip install pytesseract")
                    
                    if 'easyocr' in missing_engines:
                        instructions.append("pip install easyocr")
                    
                    if 'paddleocr' in missing_engines:
                        instructions.append("pip install paddlepaddle paddleocr")
                    
                    st.session_state['ocr_installation_instructions'] = instructions
                else:
                    st.session_state['ocr_installation_instructions'] = []
                
                # Check module status
                try:
                    module_status = self.model_manager.get_module_status()
                    if not module_status.get('transformers_available', False) or not module_status.get('sentence_transformers_available', False):
                        st.session_state['rag_available'] = False
                        st.session_state['dependency_errors'].append("Q&A functionality is limited - transformers or sentence_transformers modules not available")
                    else:
                        st.session_state['rag_available'] = True
                except:
                    st.session_state['rag_available'] = False
                    st.session_state['dependency_errors'].append("Q&A functionality not available - model manager error")
                
                st.session_state['models_initialized'] = True
                
        except Exception as e:
            logger.error(f"Error loading resources: {e}")
            st.session_state['dependency_errors'].append(f"Error initializing components: {str(e)}")
            st.session_state['models_initialized'] = False
            
            # Initialize components to None on error
            self.model_manager = None
            self.ocr_engine = None
            self.image_processor = None
            self.rag_processor = None
    
    def load_css(self):
        """Load custom CSS styles with modern, attractive design"""
        st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        /* Root variables for consistent theming */
        :root {
            --primary-color: #6366f1;
            --primary-hover: #5856eb;
            --secondary-color: #f1f5f9;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --card-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            --border-radius: 12px;
        }
        
        /* Main app styling */
        .main {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            min-height: 100vh;
            font-family: 'Inter', sans-serif;
        }
        
        /* Header styling */
        .header {
            background: var(--background-gradient);
            color: white;
            padding: 2rem;
            border-radius: var(--border-radius);
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: var(--card-shadow);
        }
        
        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
            margin: 0.5rem 0;
        }
        
        .header a {
            color: white;
            text-decoration: none;
            font-weight: 500;
            margin: 0 0.5rem;
        }
        
        .header a:hover {
            text-decoration: underline;
        }
        
        /* Card styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: white;
            padding: 8px;
            border-radius: var(--border-radius);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: var(--primary-color);
            color: white;
        }
        
        /* Button styling */
        .stButton > button {
            background: var(--background-gradient);
            color: white;
            border: none;
            border-radius: var(--border-radius);
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(102, 102, 255, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 102, 255, 0.4);
        }
        
        /* File uploader styling */
        .stFileUploader > div > div {
            background: white;
            border: 2px dashed var(--primary-color);
            border-radius: var(--border-radius);
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .stFileUploader > div > div:hover {
            border-color: var(--primary-hover);
            background: #f8faff;
        }
        
        /* Metrics styling */
        .metric-container {
            background: white;
            padding: 1.5rem;
            border-radius: var(--border-radius);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .metric-container:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        }
        
        /* Text area styling */
        .stTextArea > div > div > textarea {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: var(--border-radius);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            line-height: 1.6;
        }
        
        /* Success/Error messages */
        .stSuccess {
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            border-left: 4px solid var(--success-color);
        }
        
        .stError {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            border-left: 4px solid var(--error-color);
        }
        
        .stWarning {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border-left: 4px solid var(--warning-color);
        }
        
        .stInfo {
            background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
            border-left: 4px solid var(--primary-color);
        }
        
        /* Image styling */
        .stImage > img {
            border-radius: var(--border-radius);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background: white;
            border-radius: var(--border-radius);
            border: 1px solid #e2e8f0;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        </style>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Run the Streamlit application"""
        # Load CSS styles
        self.load_css()
        
        # Display header
        self._display_header()
        
        # Display system status
        self._display_engine_status()
        self._display_dependency_errors()
        
        # Create main tabs with attractive styling
        tabs = st.tabs(["📄 Document Processing", "⚙️ OCR Settings", "❓ Q&A Interface"])
        
        # Document Processing Tab
        with tabs[0]:
            self._document_processing_tab()
        
        # OCR Settings Tab
        with tabs[1]:
            self._ocr_settings_tab()
        
        # Q&A Interface Tab
        with tabs[2]:
            self._qa_interface_tab()
    
    def _display_header(self):
        """Display application header with modern design"""
        st.markdown("""
        <div class="header">
            <h1>🔍 Intelligent OCR & Text Analysis Tool</h1>
            <p>Extract text from images and PDFs with advanced AI-powered analysis</p>
            <p>
                <strong>Developed by Rayyan Ahmed</strong> |
                <a href="https://www.linkedin.com/in/rayyan-ahmed9477/" target="_blank">🔗 LinkedIn</a> |
                <a href="https://github.com/Rayyan9477/" target="_blank">🐙 GitHub</a>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def _display_engine_status(self):
        """Display OCR engine availability status"""
        available_engines = st.session_state.get('available_ocr_engines', [])
        missing_engines = st.session_state.get('missing_ocr_engines', [])
        
        if missing_engines:
            with st.expander("⚠️ OCR Engine Status", expanded=False):
                st.warning(f"Some OCR engines are not available: {', '.join(missing_engines)}")
                
                st.markdown("### 🛠️ Installation Instructions:")
                instructions = st.session_state.get('ocr_installation_instructions', [])
                for instruction in instructions:
                    st.code(instruction, language="bash")
                
                st.info("💡 After installing dependencies, please restart the application.")
        
        if available_engines:
            st.success(f"✅ Available OCR engines: {', '.join(available_engines)}")
    
    def _display_dependency_errors(self):
        """Display dependency errors if any"""
        if st.session_state.get('dependency_errors'):
            with st.expander("ℹ️ System Information", expanded=False):
                for error in st.session_state['dependency_errors']:
                    st.info(error)
    
    def _document_processing_tab(self):
        """Handle document processing tab with improved UI"""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📤 Upload Document")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                help="Supported formats: JPG, PNG, BMP, TIFF. Drag and drop or click to browse."
            )
            
            if uploaded_file:
                # Display uploaded image with styling
                image = Image.open(uploaded_file)
                st.image(image, caption="📸 Uploaded Image", use_column_width=True)
                
                # Image analysis (with error handling)
                if self.image_processor:
                    try:
                        quality_info = self.image_processor.assess_image_quality(image)
                        has_tables = self.image_processor.detect_tables(image)
                        
                        st.markdown("### 📊 Image Analysis")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("""
                            <div class="metric-container">
                                <h3>📈 Quality Score</h3>
                                <h2 style="color: var(--primary-color);">{:.2f}</h2>
                            </div>
                            """.format(quality_info.get('quality_score', 0)), unsafe_allow_html=True)
                        with col_b:
                            table_status = "✅ Yes" if has_tables else "❌ No"
                            st.markdown(f"""
                            <div class="metric-container">
                                <h3>📋 Contains Tables</h3>
                                <h2>{table_status}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if has_tables:
                            st.info("📊 Table structures detected. Layout preservation is recommended for better results.")
                            
                    except Exception as e:
                        st.warning(f"⚠️ Image analysis failed: {str(e)}")
                        quality_info = {'quality_score': 0}
                        has_tables = False
                else:
                    st.warning("⚠️ Image processing not available. Please check system dependencies.")
                    quality_info = {'quality_score': 0}
                    has_tables = False
                
                # OCR Processing with improved error handling
                if st.button("🔍 Extract Text", type="primary"):
                    if not self.ocr_engine:
                        st.error("❌ OCR engine not available. Please check system dependencies.")
                    else:
                        with st.spinner('🔄 Processing image...'):
                            try:
                                extracted_text = self.ocr_engine.perform_ocr(
                                    image,
                                    engine=st.session_state.get('ocr_engine', 'auto'),
                                    preserve_layout=st.session_state.get('preserve_layout', True),
                                    preprocess=True
                                )
                                
                                if extracted_text:
                                    st.session_state['extracted_text'] = extracted_text
                                    st.success("✅ Text extracted successfully!")
                                else:
                                    st.error("❌ No text could be extracted from the image.")
                                    
                            except Exception as e:
                                st.error(f"❌ Error during OCR processing: {str(e)}")
        
        with col2:
            st.markdown("### 📄 Extracted Text")
            
            if 'extracted_text' in st.session_state and st.session_state['extracted_text']:
                text = st.session_state['extracted_text']
                
                # Display text in attractive container
                st.text_area(
                    "Extracted text:",
                    value=text,
                    height=400,
                    help="📋 You can copy this text or use it for Q&A below"
                )
                
                # Text statistics with modern design
                st.markdown("### 📈 Text Statistics")
                word_count = len(text.split())
                char_count = len(text)
                line_count = len(text.split('\n'))
                
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>📝 Words</h4>
                        <h3 style="color: var(--primary-color);">{word_count:,}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                with col_stats2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>🔤 Characters</h4>
                        <h3 style="color: var(--primary-color);">{char_count:,}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                with col_stats3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>📄 Lines</h4>
                        <h3 style="color: var(--primary-color);">{line_count:,}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Download button
                st.download_button(
                    label="💾 Download Text",
                    data=text,
                    file_name=f"extracted_text_{int(time.time())}.txt",
                    mime="text/plain"
                )
            else:
                st.info("📄 Upload and process an image to see extracted text here.")
    
    def _ocr_settings_tab(self):
        """Handle OCR settings tab"""
        st.markdown("### ⚙️ OCR Configuration")
        
        # Engine selection
        available_engines = st.session_state.get('available_ocr_engines', ['auto'])
        if 'auto' not in available_engines:
            available_engines.insert(0, 'auto')
        
        selected_engine = st.selectbox(
            "🔧 OCR Engine",
            available_engines,
            index=available_engines.index(st.session_state.get('ocr_engine', 'auto')),
            help="Choose the OCR engine. 'auto' will automatically select the best available engine."
        )
        st.session_state['ocr_engine'] = selected_engine
        
        # Layout preservation
        preserve_layout = st.checkbox(
            "📐 Preserve Layout",
            value=st.session_state.get('preserve_layout', True),
            help="Maintain the original document layout and formatting"
        )
        st.session_state['preserve_layout'] = preserve_layout
        
        # Engine information
        with st.expander("ℹ️ Engine Information", expanded=False):
            st.markdown("""
            **Available OCR Engines:**
            
            - **Tesseract**: Traditional OCR engine, best for clean text
            - **EasyOCR**: Deep learning based, good for various fonts
            - **PaddleOCR**: Advanced AI model, excellent for complex layouts
            - **Auto**: Automatically selects the best available engine
            """)
    
    def _qa_interface_tab(self):
        """Handle Q&A interface tab"""
        st.markdown("### ❓ Ask Questions About Your Document")
        
        if not st.session_state.get('rag_available', False):
            st.warning("⚠️ Q&A functionality is not available. Missing required dependencies.")
            with st.expander("📋 Required Dependencies", expanded=False):
                st.markdown("""
                To enable Q&A functionality, install:
                ```bash
                pip install transformers sentence-transformers torch
                ```
                """)
        elif not st.session_state.get('extracted_text'):
            st.info("📄 Please upload and process a document first to use the Q&A feature.")
        else:
            # Q&A interface
            question = st.text_input(
                "💭 Ask a question about your document:",
                placeholder="e.g., What is the main topic of this document?"
            )
            
            if st.button("🔍 Get Answer", type="primary") and question:
                if not self.rag_processor:
                    st.error("❌ Q&A processor not available.")
                else:
                    with st.spinner('🤔 Analyzing document and generating answer...'):
                        try:
                            answer = self.rag_processor.answer_question(
                                question, 
                                st.session_state['extracted_text']
                            )
                            
                            if answer:
                                st.success("✅ Answer generated!")
                                st.markdown("### 💡 Answer")
                                st.write(answer.get('answer', 'No answer found.'))
                                
                                if 'confidence' in answer:
                                    confidence = answer['confidence']
                                    st.markdown(f"**Confidence:** {confidence:.2%}")
                                    
                                if 'sources' in answer:
                                    with st.expander("📚 Source References", expanded=False):
                                        for i, source in enumerate(answer['sources'], 1):
                                            st.markdown(f"**Source {i}:** {source}")
                            else:
                                st.error("❌ Could not generate an answer.")
                                
                        except Exception as e:
                            st.error(f"❌ Error during Q&A processing: {str(e)}")
        
        # About section
        if not st.session_state.get('extracted_text'):
            st.markdown("""
            ### 🤖 About the Q&A Feature
            
            This feature uses advanced AI to answer questions about your documents:
            
            - **🔍 Retrieval-Augmented Generation (RAG)**: Finds relevant parts of your document
            - **🧠 Question Answering**: Uses AI models to provide precise answers
            - **📊 Confidence Scoring**: Shows how confident the system is in its answer
            - **📚 Source Attribution**: Shows which parts of the document were used
            
            The system works best with clear, well-formatted documents and specific questions.
            """)

def main():
    """Main entry point for the Streamlit app"""
    # Configure Streamlit page - MUST be first Streamlit command
    st.set_page_config(
        page_title="OCR Image-to-Text Tool",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()
