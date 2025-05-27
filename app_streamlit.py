#!/usr/bin/env python
"""
OCR Application Entry Point

This script runs the OCR web interface using Streamlit.
"""

import os
import sys
import logging

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
except ImportError:
    # Try relative imports
    from core.ocr_engine import OCREngine
    from core.image_processor import ImageProcessor
    from models.model_manager import ModelManager
    from rag.rag_processor import RAGProcessor
    from config.settings import Settings
    from utils.text_utils import extract_entities, format_ocr_result

# Import our new multi-engine system
try:
    from multi_engine_ocr import get_multi_ocr, extract_text_multi_engine
    from enhanced_image_processor import get_image_processor
    MULTI_ENGINE_AVAILABLE = True
except ImportError:
    MULTI_ENGINE_AVAILABLE = False
    logger.warning("Multi-engine OCR system not available")

logger = logging.getLogger(__name__)

class StreamlitApp:
    """
    Streamlit web interface for the OCR application
    """    def __init__(self):
        """Initialize the Streamlit app"""
        self.init_session_state()
        
        # Initialize components to None first
        self.model_manager = None
        self.ocr_engine = None
        self.image_processor = None
        self.rag_processor = None
        self.multi_ocr = None
        
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
            # Initialize enhanced image processor
            if MULTI_ENGINE_AVAILABLE:
                with st.spinner('Initializing Multi-Engine OCR system...'):
                    self.multi_ocr = get_multi_ocr()
                    self.image_processor = get_image_processor()
                    st.session_state['multi_engine_available'] = True
                    
                    # Get available engines from multi-OCR system
                    available_engines = list(self.multi_ocr.engines.keys())
                    st.session_state['available_ocr_engines'] = available_engines
                    st.session_state['missing_ocr_engines'] = []
                    
                    if available_engines:
                        st.session_state['models_initialized'] = True
                        logger.info(f"✅ Multi-Engine OCR initialized with: {available_engines}")
                    else:
                        st.session_state['dependency_errors'].append("No OCR engines available")
            else:
                # Fallback to original system
                with st.spinner('Initializing standard OCR system...'):
                    try:
                        self.settings = Settings()
                        self.model_manager = ModelManager(self.settings)
                        self.ocr_engine = OCREngine(self.settings)
                        self.image_processor = ImageProcessor(self.settings)
                        self.rag_processor = RAGProcessor(self.model_manager, self.settings)
                        
                        st.session_state['available_ocr_engines'] = self.ocr_engine.enabled_engines if self.ocr_engine else []
                        st.session_state['multi_engine_available'] = False
                    except Exception as e:
                        logger.error(f"Fallback OCR initialization failed: {e}")
                        st.session_state['dependency_errors'].append(f"OCR initialization failed: {str(e)}")
                        st.session_state['available_ocr_engines'] = []
            
            # Initialize RAG processor if possible
            try:
                if not hasattr(self, 'rag_processor') or self.rag_processor is None:
                    if hasattr(self, 'model_manager') and self.model_manager:
                        self.rag_processor = RAGProcessor(self.model_manager, self.settings)
                        st.session_state['rag_available'] = True
                    else:
                        st.session_state['rag_available'] = False
                        st.session_state['dependency_errors'].append("Q&A functionality not available")
            except Exception as e:
                logger.warning(f"RAG processor initialization failed: {e}")
                st.session_state['rag_available'] = False
                
        except Exception as e:
            logger.error(f"Error loading resources: {e}")
            st.session_state['dependency_errors'].append(f"Error initializing components: {str(e)}")
            st.session_state['models_initialized'] = False
    
    def load_css(self):
        """Load custom CSS styles"""
        css_file = os.path.join(os.path.dirname(__file__), 'static', 'styles.css')
        try:
            with open(css_file, encoding='utf-8') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        except Exception as e:
            logger.warning(f"Could not load CSS: {e}")
            # Provide minimal inline CSS as fallback
            st.markdown("""
            <style>
            .formatted-text {
                position: relative;
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 1rem;
                max-height: 500px;
                overflow-y: auto;
                font-family: monospace;
                white-space: pre-wrap;
                margin-bottom: 1rem;
            }
            
            .copy-btn {
                position: absolute;
                top: 0.5rem;
                right: 0.5rem;
                background: rgba(0, 123, 255, 0.1);
                border: none;
                border-radius: 4px;
            }
            
            .card {
                background: #ffffff;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 1.5rem;
                margin: 1rem 0;
            }
            
            .header {
                text-align: center;
                padding: 1rem;
                margin-bottom: 2rem;
            }            </style>
            """, unsafe_allow_html=True)
    
    def run(self):
        """Run the Streamlit application"""
        # Load CSS styles
        self.load_css()
        
        # Display header
        self._display_header()
        
        # Check OCR engine availability
        self._display_engine_status()
        
        # Display dependency errors if any
        self._display_dependency_errors()
        
        # Create main tabs
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
        """Display application header"""
        st.markdown("""
        <div class="header">
            <h1>🔍 Intelligent OCR and Text Analysis Tool</h1>
            <p>Extract text from images and PDFs with advanced AI-powered analysis</p>
            <p>
                <strong>Developed by Rayyan Ahmed</strong> |
                <a href="https://www.linkedin.com/in/rayyan-ahmed9477/" target="_blank">LinkedIn</a> |
                <a href="https://github.com/Rayyan9477/" target="_blank">GitHub</a>
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
                
                st.markdown("### Installation Instructions:")
                instructions = st.session_state.get('ocr_installation_instructions', [])
                for instruction in instructions:
                    st.code(instruction)
                
                st.info("After installing dependencies, please restart the application.")
        
        if available_engines:
            st.success(f"✅ Available OCR engines: {', '.join(available_engines)}")
    
    def _display_dependency_errors(self):
        """Display dependency errors if any"""
        if st.session_state.get('dependency_errors'):
            with st.expander("ℹ️ Additional Information", expanded=False):
                for error in st.session_state['dependency_errors']:
                    st.info(error)
    
    def _document_processing_tab(self):
        """Handle document processing tab"""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📤 Upload Document")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                help="Supported formats: JPG, PNG, BMP, TIFF"            )
            if uploaded_file:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Image analysis (with error handling)
                if self.image_processor:
                    try:
                        quality_info = self.image_processor.assess_image_quality(image)
                        has_tables = self.image_processor.detect_tables(image)
                        
                        st.markdown("### 📊 Image Analysis")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Quality Score", f"{quality_info.get('quality_score', 0):.2f}")
                        with col_b:
                            st.metric("Contains Tables", "Yes" if has_tables else "No")
                            st.info("📊 Table structures detected. Layout preservation is recommended.")
                    except Exception as e:
                        st.warning(f"Image analysis failed: {str(e)}")
                        quality_info = {'quality_score': 0}
                        has_tables = False
                else:
                    st.warning("⚠️ Image processing not available. Please check system dependencies.")
                    quality_info = {'quality_score': 0}
                    has_tables = False
                
                # OCR Processing
                if st.button("🔍 Extract Text", type="primary"):
                    if not self.ocr_engine:
                        st.error("❌ OCR engine not available. Please check system dependencies.")
                    else:
                        with st.spinner('Processing image...'):
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
                
                # Display text in expandable container
                with st.container():
                    st.text_area(
                        "Extracted text:",
                        value=text,
                        height=400,
                        help="You can copy this text or use it for Q&A"
                    )
                
                # Text statistics
                st.markdown("### 📈 Text Statistics")
                word_count = len(text.split())
                char_count = len(text)
                line_count = len(text.split('\n'))
                
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Words", word_count)
                with col_stats2:
                    st.metric("Characters", char_count)
                with col_stats3:
                    st.metric("Lines", line_count)
                
                # Extract entities
                entities = extract_entities(text)
                if any(entities.values()):
                    st.markdown("### 🏷️ Detected Entities")
                    for entity_type, values in entities.items():
                        if values:
                            st.markdown(f"**{entity_type.title()}**: {', '.join(values[:5])}")
                
                # Download options
                st.markdown("### 💾 Download")
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        label="📄 Download as TXT",
                        data=text,
                        file_name="extracted_text.txt",
                        mime="text/plain"
                    )
                with col_dl2:
                    formatted_text = format_ocr_result(text, 'markdown')
                    st.download_button(
                        label="📝 Download as Markdown",
                        data=formatted_text,
                        file_name="extracted_text.md",
                        mime="text/markdown"
                    )
            else:
                st.info("Upload an image and click 'Extract Text' to see results here.")
    
    def _ocr_settings_tab(self):
        """Handle OCR settings tab"""
        st.markdown("### ⚙️ OCR Engine Settings")
        
        # Engine selection
        available_engines = st.session_state.get('available_ocr_engines', [])
        engine_options = ['auto'] + available_engines + ['combined']
        
        st.session_state['ocr_engine'] = st.selectbox(
            "Select OCR Engine",
            options=engine_options,
            index=0,
            help="Choose the OCR engine or let the system decide automatically"
        )
        
        # Layout preservation option
        st.session_state['preserve_layout'] = st.checkbox(
            "Preserve text layout and formatting",
            value=True,
            help="Maintains the original document's layout including line breaks and text positioning"
        )
        
        # Advanced settings
        st.markdown("### 🔧 Advanced Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Image Preprocessing")
            preprocess_enabled = st.checkbox(
                "Enable image preprocessing",
                value=True,
                help="Applies various image enhancement techniques before OCR"
            )
            
            if preprocess_enabled:
                st.checkbox("Enhance contrast", value=True)
                st.checkbox("Remove noise", value=True)
                st.checkbox("Correct skew", value=True)
        
        with col2:
            st.markdown("#### Performance")
            st.slider("Processing timeout (seconds)", 30, 300, 60)
            st.checkbox("Use GPU acceleration (if available)", value=False)
        
        # Engine comparison
        st.markdown("---")
        st.markdown("### 🔍 OCR Engine Comparison")
        
        engine_info = {
            "Tesseract": {
                "description": "Open-source OCR engine with good general accuracy",
                "strengths": "Clear typography, well-formatted documents",
                "best_for": "Scanned documents, books, articles"
            },
            "PaddleOCR": {
                "description": "Fast and accurate OCR optimized for multiple languages",
                "strengths": "Speed, multi-language support, handwriting",
                "best_for": "Forms, receipts, mixed content"
            },
            "EasyOCR": {
                "description": "General-purpose OCR with 80+ language support",
                "strengths": "Wide language support, good for various fonts",
                "best_for": "International documents, signage"
            }
        }
        
        for engine, info in engine_info.items():
            if engine.lower() in [e.lower() for e in available_engines]:
                with st.expander(f"ℹ️ {engine}"):
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**Strengths:** {info['strengths']}")
                    st.write(f"**Best for:** {info['best_for']}")
    
    def _qa_interface_tab(self):
        """Handle Q&A interface tab"""
        if 'extracted_text' in st.session_state and st.session_state['extracted_text']:
            st.markdown("### 🤔 Ask Questions About Your Document")
            
            # Pre-defined question templates
            st.markdown("#### Quick Questions")
            quick_questions = [
                "What is the main topic?",
                "What are the key dates mentioned?",
                "Who are the people mentioned?",
                "What numbers or amounts are mentioned?",
                "Summarize the content",
                "What is the document type?"
            ]
            
            selected_question = st.selectbox("Select a quick question:", ["Custom question..."] + quick_questions)
            
            if selected_question != "Custom question...":
                query = selected_question
            else:
                query = st.text_input("Enter your custom question:")
            
            if query and st.button("🔍 Get Answer"):
                with st.spinner('🤔 Analyzing document and finding answer...'):
                    try:
                        result = self.rag_processor.process_query(st.session_state['extracted_text'], query)
                        
                        # Display answer
                        st.markdown("### 💡 Answer")
                        st.markdown(f"**{result['answer']}**")
                        
                        # Show confidence
                        confidence = result.get('confidence', 0) * 100
                        st.progress(min(confidence / 100, 1.0))
                        st.caption(f"Confidence: {confidence:.1f}%")
                        
                        # Show source passages
                        if 'chunks_used' in result and result['chunks_used']:
                            with st.expander("📖 View source passages"):
                                for i, chunk in enumerate(result['chunks_used']):
                                    score = result.get('chunk_scores', [0])[i] if i < len(result.get('chunk_scores', [])) else 0
                                    st.markdown(f"**Passage {i+1}** (relevance: {score:.2f})")
                                    st.markdown(f"> {chunk}")
                                    
                    except Exception as e:
                        st.error(f"❌ Error processing question: {str(e)}")
                        st.info("💡 Try rephrasing your question or check if the answer might be in the document.")
            
            # Tips for better questions
            with st.expander("💡 Tips for Better Questions"):
                st.markdown("""
                - Be specific in your questions
                - Ask about information that's likely to be in the document
                - Use keywords that might appear in the text
                - For dates: "What date..." or "When..."
                - For people: "Who..." or "What person..."
                - For numbers: "How much..." or "What amount..."
                - For summaries: "What is the main point..." or "Summarize..."
                """)
                
        else:
            st.info("📄 Please upload and process a document first to use the Q&A feature.")
            
            st.markdown("""
            ### About the Q&A Feature
            
            This feature uses advanced AI to answer questions about your documents:
            
            - **Retrieval-Augmented Generation (RAG)**: Finds relevant parts of your document
            - **Question Answering**: Uses AI models to provide precise answers
            - **Confidence Scoring**: Shows how confident the system is in its answer
            - **Source Attribution**: Shows which parts of the document were used
            
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
