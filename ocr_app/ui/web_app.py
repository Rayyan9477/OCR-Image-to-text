"""
Web UI module for the OCR application using Streamlit
"""

import streamlit as st
import os
import sys
import io
import logging
import time
from PIL import Image
import base64
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path

from ..core.ocr_engine import OCREngine
from ..core.image_processor import ImageProcessor
from ..models.model_manager import ModelManager
from ..rag.rag_processor import RAGProcessor
from ..config.settings import Settings
from ..utils.text_utils import extract_entities, format_ocr_result

logger = logging.getLogger(__name__)

class StreamlitApp:
    """
    Streamlit web interface for the OCR application
    """
    
    def __init__(self):
        """Initialize the Streamlit app"""
        self.settings = Settings()
        self.init_session_state()
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
            # Initialize components
            with st.spinner('Initializing OCR system...'):
                self.model_manager = ModelManager(self.settings)
                self.ocr_engine = OCREngine(self.settings)
                self.image_processor = ImageProcessor(self.settings)
                self.rag_processor = RAGProcessor(self.model_manager, self.settings)
                
                # Store available OCR engines in session state
                st.session_state['available_ocr_engines'] = self.ocr_engine.enabled_engines
                
                # Check for missing OCR engines
                all_engines = ['tesseract', 'easyocr', 'paddleocr']
                missing_engines = [engine for engine in all_engines if engine not in self.ocr_engine.enabled_engines]
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
                module_status = self.model_manager.get_module_status()
                if not module_status.get('transformers_available', False) or not module_status.get('sentence_transformers_available', False):
                    st.session_state['rag_available'] = False
                    st.session_state['dependency_errors'].append("Q&A functionality is limited - transformers or sentence_transformers modules not available")
                else:
                    st.session_state['rag_available'] = True
                
                st.session_state['models_initialized'] = True
                
        except Exception as e:
            logger.error(f"Error loading resources: {e}")
            st.session_state['dependency_errors'].append(f"Error initializing components: {str(e)}")
            st.session_state['models_initialized'] = False
    
    def load_css(self):
        """Load custom CSS styles"""
        css_file = os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'styles.css')
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
            }
            </style>
            """, unsafe_allow_html=True)
    
    def load_js(self):
        """Load JavaScript functions"""
        js_file = os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'script.js')
        try:
            with open(js_file, encoding='utf-8') as f:
                js_content = f.read()
        except Exception as e:
            logger.warning(f"Could not load JavaScript: {e}")
            # Fallback JavaScript
            js_content = """
            function copyText(button) {
                const textContainer = button.parentElement.querySelector('pre');
                if (!textContainer) {
                    console.error('Text container not found');
                    button.innerHTML = '❌';
                    setTimeout(() => button.innerHTML = '📋', 2000);
                    return;
                }
                
                const text = textContainer.innerText || textContainer.textContent;
                
                if (navigator.clipboard && window.isSecureContext) {
                    navigator.clipboard.writeText(text)
                        .then(() => {
                            button.innerHTML = '✓';
                            setTimeout(() => button.innerHTML = '📋', 2000);
                        })
                        .catch(err => {
                            console.error('Failed to copy:', err);
                        });
                } else {
                    const textArea = document.createElement('textarea');
                    textArea.value = text;
                    textArea.style.position = 'fixed';
                    textArea.style.opacity = '0';
                    document.body.appendChild(textArea);
                    
                    try {
                        textArea.select();
                        document.execCommand('copy');
                        button.innerHTML = '✓';
                        setTimeout(() => button.innerHTML = '📋', 2000);
                    } catch (err) {
                        console.error('Fallback copy failed:', err);
                        button.innerHTML = '❌';
                        setTimeout(() => button.innerHTML = '📋', 2000);
                    } finally {
                        document.body.removeChild(textArea);
                    }
                }
            }
            """
        
        # Inject JavaScript using components.html
        js_code = f"""
        <script>
        {js_content}
        
        // Initialize at document load to ensure script is loaded properly
        document.addEventListener('DOMContentLoaded', function() {{
            // Attach click handlers to any copy buttons that exist
            const copyButtons = document.querySelectorAll('.copy-btn');
            copyButtons.forEach(btn => {{
                btn.addEventListener('click', function() {{
                    copyText(this);
                }});
            }});
        }});
        
        // Re-attach event listeners when Streamlit reruns
        const observer = new MutationObserver(function(mutations) {{
            const copyButtons = document.querySelectorAll('.copy-btn');
            copyButtons.forEach(btn => {{
                if (!btn.hasAttribute('data-listener')) {{
                    btn.setAttribute('data-listener', 'true');
                    btn.addEventListener('click', function() {{
                        copyText(this);
                    }});
                }}
            }});
        }});
        
        // Start observing the document with the configured parameters
        observer.observe(document.body, {{ childList: true, subtree: true }});
        </script>
        """
        st.components.v1.html(js_code, height=0)
    
    def perform_ocr(self, image: Image.Image, engine: str = "auto", preserve_layout: bool = True) -> str:
        """
        Perform OCR on the given image
        
        Args:
            image: PIL Image to process
            engine: OCR engine to use
            preserve_layout: Whether to preserve text layout
            
        Returns:
            Extracted text
        """
        try:
            return self.ocr_engine.perform_ocr(image, engine, preserve_layout)
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return f"Error: OCR processing failed - {str(e)}"
    
    def process_pdf(self, pdf_bytes: bytes) -> str:
        """
        Process PDF document and extract text
        
        Args:
            pdf_bytes: PDF file bytes
            
        Returns:
            Extracted text
        """
        try:
            import fitz  # PyMuPDF
            
            # Create a PDF document object
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                full_text = []
                
                # Process each page
                for i, page in enumerate(doc):
                    # Get page as an image
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Perform OCR on the page image
                    try:
                        page_text = self.perform_ocr(
                            img, 
                            engine=st.session_state['ocr_engine'],
                            preserve_layout=st.session_state['preserve_layout']
                        )
                        
                        # Check if OCR failed and try fallback
                        if page_text.startswith("Error:"):
                            st.warning(f"OCR failed on page {i+1}: {page_text}")
                            
                            # Try alternative engine
                            if self.ocr_engine.enabled_engines:
                                alt_engine = next(iter(self.ocr_engine.enabled_engines))
                                if alt_engine != st.session_state['ocr_engine']:
                                    st.info(f"Trying fallback OCR for page {i+1}...")
                                    page_text = self.perform_ocr(
                                        img,
                                        engine=alt_engine,
                                        preserve_layout=st.session_state['preserve_layout']
                                    )
                        
                        full_text.append(f"--- Page {i+1} ---\n{page_text}")
                        
                    except Exception as e:
                        st.error(f"Error processing page {i+1}: {str(e)}")
                        full_text.append(f"--- Page {i+1} ---\nError: {str(e)}")
                
                # Combine all pages
                return "\n\n".join(full_text)
                
        except ImportError:
            st.error("PyMuPDF (fitz) is not installed. Cannot process PDF files.")
            return "Error: PDF processing requires PyMuPDF. Install with 'pip install pymupdf'."
        except Exception as e:
            st.error(f"PDF processing error: {str(e)}")
            return f"Error: PDF processing failed - {str(e)}"
    
    def display_formatted_text(self, text: str, title: str = "Extracted Text"):
        """
        Display formatted text with copy button
        
        Args:
            text: Text to display
            title: Title to show above the text
        """
        st.markdown(f"### {title}")
        
        if not text:
            st.info("No text extracted.")
            return
            
        # Create container with copy button
        container_html = f"""
        <div class="formatted-text">
            <button class="copy-btn" onclick="copyText(this)">📋</button>
            <pre>{text}</pre>
        </div>
        """
        st.markdown(container_html, unsafe_allow_html=True)
    
    def display_entities(self, text: str):
        """
        Extract and display entities from the text
        
        Args:
            text: Text to analyze
        """
        if not text:
            return
            
        entities = extract_entities(text)
        
        # Only show the section if we found entities
        if any(entities.values()):
            st.markdown("### Detected Information")
            
            # Create three columns
            cols = st.columns(3)
            
            # Display dates in the first column
            if entities['dates']:
                with cols[0]:
                    st.markdown("#### Dates")
                    for date in entities['dates']:
                        st.markdown(f"- {date}")
            
            # Display emails in the second column
            if entities['emails']:
                with cols[1]:
                    st.markdown("#### Emails")
                    for email in entities['emails']:
                        st.markdown(f"- {email}")
            
            # Display phones in the third column
            if entities['phones']:
                with cols[2]:
                    st.markdown("#### Phone Numbers")
                    for phone in entities['phones']:
                        st.markdown(f"- {phone}")
            
            # Display URLs below in their own section
            if entities['urls']:
                st.markdown("#### URLs")
                for url in entities['urls']:
                    st.markdown(f"- {url}")
    
    def run(self):
        """Run the Streamlit application"""
        # Set page configuration
        st.set_page_config(
            page_title="OCR Image to Text",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Load resources
        try:
            self.load_css()
        except Exception as e:
            st.warning(f"Could not load custom styles: {str(e)}. Using default styles.")
        
        try:
            self.load_js()
        except Exception as e:
            st.warning(f"Could not load JavaScript functions: {str(e)}. Some interactive features may be limited.")
        
        # Page header
        st.markdown("""
        <div class="header">
            <h1>Intelligent OCR and Text Analysis Tool</h1>
            <p>Developed by <strong>Rayyan Ahmed</strong></p>
            <p>
                <a href="https://www.linkedin.com/in/rayyan-ahmed9477/" target="_blank">LinkedIn</a> |
                <a href="https://github.com/Rayyan9477/" target="_blank">GitHub</a> |
                Email: <a href="mailto:rayyanahmed265@yahoo.com">rayyanahmed265@yahoo.com</a>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display OCR engine availability
        if 'missing_ocr_engines' in st.session_state and st.session_state['missing_ocr_engines']:
            with st.expander("⚠️ OCR Engine Availability Warning", expanded=True):
                st.warning(
                    f"Some OCR engines are not available: {', '.join(st.session_state['missing_ocr_engines'])}. "
                    "This may limit OCR accuracy and functionality."
                )
                
                st.markdown("### How to install missing OCR engines:")
                
                for instruction in st.session_state['ocr_installation_instructions']:
                    st.code(instruction)
                    
                st.markdown("""
                    **Note:** After installing the required dependencies, please restart the application.
                    For full functionality, we recommend having at least one OCR engine installed.
                """)
        
        # Display available OCR engines
        if 'available_ocr_engines' in st.session_state and st.session_state['available_ocr_engines']:
            st.success(f"Available OCR engines: {', '.join(st.session_state['available_ocr_engines'])}")
        
        # Display any dependency errors
        if st.session_state['dependency_errors']:
            with st.expander("Additional Dependency Notes", expanded=False):
                for error in st.session_state['dependency_errors']:
                    st.info(error)
        
        # Create tabs
        tabs = st.tabs(["📄 Document Processing", "⚙️ OCR Settings", "❓ Q&A Interface"])
        
        # Document Processing Tab
        with tabs[0]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Upload Document")
                uploaded_file = st.file_uploader(
                    "Choose an image or PDF file",
                    type=["jpg", "jpeg", "png", "pdf", "bmp", "tiff"],
                    help="Supported formats: JPG, PNG, PDF, BMP, TIFF"
                )
                
                if uploaded_file:
                    if uploaded_file.type.startswith('image'):
                        image = Image.open(uploaded_file)
                        st.image(image, caption="Uploaded Image", use_column_width=True)
                        
                        # Check if image contains tables
                        has_tables = self.image_processor.detect_tables(image)
                        if has_tables:
                            st.info("📊 Table structures detected in the image. Layout preservation is recommended.")
                        
                        with st.spinner('🔍 Performing OCR...'):
                            try:
                                extracted_text = self.perform_ocr(
                                    image, 
                                    engine=st.session_state['ocr_engine'], 
                                    preserve_layout=st.session_state['preserve_layout']
                                )
                                
                                # Check if the OCR returned an error message
                                if extracted_text.startswith("Error:"):
                                    st.error(extracted_text)
                                    st.info("Trying fallback OCR methods...")
                                    
                                    # Try different engines as fallback
                                    for fallback_engine in self.ocr_engine.enabled_engines:
                                        if fallback_engine != st.session_state['ocr_engine']:
                                            with st.spinner(f'Trying {fallback_engine} OCR engine...'):
                                                fallback_text = self.perform_ocr(
                                                    image,
                                                    engine=fallback_engine,
                                                    preserve_layout=st.session_state['preserve_layout']
                                                )
                                                if not fallback_text.startswith("Error:"):
                                                    extracted_text = fallback_text
                                                    st.success(f"Successfully extracted text using {fallback_engine} engine")
                                                    break
                                
                                # Store the extracted text in session state
                                st.session_state['extracted_text'] = extracted_text
                                
                            except Exception as e:
                                st.error(f"OCR processing error: {str(e)}")
                                st.session_state['extracted_text'] = f"Error: {str(e)}"
                        
                    elif uploaded_file.type == 'application/pdf':
                        st.success("PDF file uploaded successfully")
                        
                        with st.spinner('🔍 Processing PDF document...'):
                            try:
                                pdf_bytes = uploaded_file.getvalue()
                                extracted_text = self.process_pdf(pdf_bytes)
                                st.session_state['extracted_text'] = extracted_text
                            except Exception as e:
                                st.error(f"PDF processing error: {str(e)}")
                                st.session_state['extracted_text'] = f"Error: {str(e)}"
                    
                    else:
                        st.error("Unsupported file type")
            
            with col2:
                if 'extracted_text' in st.session_state and st.session_state['extracted_text']:
                    self.display_formatted_text(st.session_state['extracted_text'])
                    
                    # Display entities
                    self.display_entities(st.session_state['extracted_text'])
                    
                    # Export options
                    st.markdown("### Export Options")
                    export_format = st.selectbox(
                        "Export format",
                        options=["Text (TXT)", "Markdown (MD)", "HTML"]
                    )
                    
                    if st.button("Export Text"):
                        text = st.session_state['extracted_text']
                        
                        if export_format == "Text (TXT)":
                            content = text
                            mime_type = "text/plain"
                            file_ext = "txt"
                        elif export_format == "Markdown (MD)":
                            content = format_ocr_result(text, 'markdown')
                            mime_type = "text/markdown"
                            file_ext = "md"
                        else:  # HTML
                            content = format_ocr_result(text, 'html')
                            mime_type = "text/html"
                            file_ext = "html"
                        
                        # Create download link
                        b64 = base64.b64encode(content.encode()).decode()
                        href = f'<a href="data:{mime_type};base64,{b64}" download="extracted_text.{file_ext}">Click here to download</a>'
                        st.markdown(href, unsafe_allow_html=True)
                else:
                    st.info("Upload a document to extract text")
        
        # OCR Settings Tab
        with tabs[1]:
            st.markdown("### OCR Engine Settings")
            
            # Engine selection
            engine_options = ["Auto (Recommended)"]
            
            if 'available_ocr_engines' in st.session_state and st.session_state['available_ocr_engines']:
                engine_options.extend(st.session_state['available_ocr_engines'])
                
                # Add combined if multiple engines available
                if len(st.session_state['available_ocr_engines']) > 1:
                    engine_options.append("Combined")
            
            engine = st.selectbox(
                "Select OCR Engine",
                options=engine_options,
                index=0,
                help="Choose which OCR engine to use for text extraction"
            )
            
            # Map selection to engine name
            engine_map = {
                "Auto (Recommended)": "auto",
                "Combined": "combined"
            }
            
            # Add available engines to the map
            if 'available_ocr_engines' in st.session_state:
                for available_engine in st.session_state['available_ocr_engines']:
                    engine_map[available_engine] = available_engine
            
            st.session_state['ocr_engine'] = engine_map.get(engine, "auto")
            
            # Layout preservation option
            st.session_state['preserve_layout'] = st.checkbox(
                "Preserve text layout and formatting",
                value=True,
                help="Maintains the original document's layout including line breaks and approximate text positioning"
            )
            
            # Add more OCR settings
            st.markdown("### Advanced Settings")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Image Preprocessing")
                apply_preprocessing = st.checkbox(
                    "Enable image preprocessing",
                    value=True,
                    help="Applies various image enhancement techniques before OCR"
                )
            
            with col2:
                st.markdown("#### Language Settings")
                language = st.selectbox(
                    "Language",
                    options=["English", "Multi-language (Slower)"],
                    index=0,
                    help="Select the primary language of your document"
                )
                
            st.markdown("---")
            st.markdown("""
            ### OCR Engine Comparison
            
            Each OCR engine has different strengths and weaknesses:
            
            - **Tesseract**: Open-source engine with good general accuracy, works well with clear typography.
            - **PaddleOCR**: Fast and accurate, optimized for Asian languages but works well for English.
            - **EasyOCR**: Good general-purpose OCR with support for 80+ languages.
            - **Combined**: Uses multiple engines and selects the best result (recommended but slower).
            """)
        
        # Q&A Interface Tab
        with tabs[2]:
            if 'extracted_text' in st.session_state and st.session_state['extracted_text']:
                st.markdown("### Ask Questions")
                query = st.text_input("Enter your question about the document")
                
                if query:
                    with st.spinner('🤔 Finding answer...'):
                        result = self.rag_processor.process_query(st.session_state['extracted_text'], query)
                    
                    st.markdown(f"**Answer:** {result['answer']}")
                    
                    # Show confidence score
                    confidence = result.get('confidence', 0) * 100
                    st.progress(min(confidence / 100, 1.0))
                    st.caption(f"Confidence: {confidence:.1f}%")
                    
                    # Show source passages
                    if 'chunks_used' in result and result['chunks_used']:
                        with st.expander("View source passages"):
                            for i, chunk in enumerate(result['chunks_used']):
                                score = result.get('chunk_scores', [0])[i] if i < len(result.get('chunk_scores', [])) else 0
                                st.markdown(f"**Passage {i+1}** (relevance: {score:.2f})")
                                st.markdown(f"> {chunk}")
            else:
                st.info("Please upload and process a document first.")
