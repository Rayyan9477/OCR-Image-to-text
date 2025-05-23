#!/usr/bin/env python3
# app.py - Streamlit app for OCR and Text Analysis
import streamlit as st
from PIL import Image
import io
import importlib
import base64
import os
import time
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Initialize session state for error tracking
if 'dependency_errors' not in st.session_state:
    st.session_state['dependency_errors'] = []

# Set environment variables to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Try to fix the Keras/TensorFlow compatibility issues
try:
    # Try importlib first to check availability without importing
    tf_keras_spec = importlib.util.find_spec('tf_keras')
    if tf_keras_spec:
        # Only import if it's available
        tf_keras = importlib.import_module('tf_keras')
        st.info("Using tf-keras for compatibility")
    else:
        # Check if regular keras is available
        keras_spec = importlib.util.find_spec('keras')
        if keras_spec:
            keras = importlib.import_module('keras')
            keras_version = keras.__version__
            if keras_version.startswith('3.'):
                st.info("Using Keras 3 compatibility mode")
                st.session_state['dependency_errors'].append(
                    "Using Keras 3 compatibility mode. Some advanced features may be limited."
                )
        else:
            # Try with tensorflow's built-in keras
            tensorflow_spec = importlib.util.find_spec('tensorflow')
            if tensorflow_spec:
                tf = importlib.import_module('tensorflow')
                if hasattr(tf, 'keras'):
                    st.info("Using TensorFlow's built-in Keras")
                else:
                    st.session_state['dependency_errors'].append(
                        "TensorFlow detected but Keras is not available. Some features may be limited."
                    )
except Exception as e:
    st.session_state['dependency_errors'].append(f"Keras/TensorFlow note: {str(e)}")
    pass

# Check OCR engines availability
def check_ocr_engines():
    """Check which OCR engines are available and provide installation instructions if needed"""
    # First try the ocr_manager implementation
    try:
        import ocr_manager
        return ocr_manager.check_ocr_engines()
    except ImportError:
        # Fall back to manual checking
        available_engines = []
        missing_engines = []
        installation_instructions = []
        
        # Check PaddleOCR
        paddle_spec = importlib.util.find_spec('paddleocr')
        if paddle_spec:
            available_engines.append("PaddleOCR")
        else:
            missing_engines.append("PaddleOCR")
            installation_instructions.append("pip install paddlepaddle paddleocr")
        
        # Check EasyOCR
        easy_spec = importlib.util.find_spec('easyocr')
        if easy_spec:
            available_engines.append("EasyOCR")
        else:
            missing_engines.append("EasyOCR")
            installation_instructions.append("pip install easyocr")
        
        # Check Tesseract
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            available_engines.append("Tesseract")
        except:
            missing_engines.append("Tesseract")
            if os.name == 'nt':  # Windows
                installation_instructions.append(
                    "1. Download Tesseract installer from https://github.com/UB-Mannheim/tesseract/wiki\n"
                    "2. Install and add to PATH\n"
                    "3. pip install pytesseract"
                )
            elif os.name == 'posix':  # Linux/macOS
                if os.path.exists('/usr/bin/apt'):  # Debian/Ubuntu
                    installation_instructions.append("sudo apt-get install tesseract-ocr && pip install pytesseract")
                elif os.path.exists('/usr/bin/brew'):  # macOS with Homebrew
                    installation_instructions.append("brew install tesseract && pip install pytesseract")
                else:
                    installation_instructions.append(
                        "Install tesseract-ocr using your package manager and then: pip install pytesseract"
                    )
        
        return available_engines, missing_engines, installation_instructions

# Now import our modules
try:
    # Try loading from ocr_manager first (new implementation)
    try:
        import ocr_manager
        from ocr_manager import perform_ocr, detect_tables
        st.session_state['ocr_module'] = 'ocr_manager'
        st.session_state['modules_loaded'] = True
        logger.info("Using ocr_manager module")
    except ImportError:
        # Fall back to original modules
        import model_manager
        from model_manager import initialize_models, get_ocr_config, update_ocr_config
        from ocr_module import perform_ocr, detect_tables
        st.session_state['ocr_module'] = 'ocr_module'
        st.session_state['modules_loaded'] = True
        logger.warning("Using original ocr_module (ocr_manager not found)")
    
    try:
        from rag_module import process_query
        st.session_state['rag_available'] = True
    except ImportError:
        st.session_state['rag_available'] = False
        st.session_state['dependency_errors'].append("RAG module not available - Q&A functionality will be limited")
    
    # Check OCR engines and store in session state
    available_engines, missing_engines, installation_instructions = check_ocr_engines()
    st.session_state['available_ocr_engines'] = available_engines
    st.session_state['missing_ocr_engines'] = missing_engines
    st.session_state['ocr_installation_instructions'] = installation_instructions
    
except Exception as e:
    st.error(f"Error loading modules: {str(e)}")
    st.session_state['modules_loaded'] = False
    st.session_state['dependency_errors'].append(f"Module loading error: {str(e)}")
    st.session_state['available_ocr_engines'] = []
    st.session_state['missing_ocr_engines'] = ["PaddleOCR", "EasyOCR", "Tesseract"]
    st.session_state['ocr_installation_instructions'] = ["pip install -r requirements.txt"]

# Dynamic import for fitz (PyMuPDF)
try:
    fitz = importlib.import_module('fitz')  # PyMuPDF
except ImportError:
    st.error("PyMuPDF not installed. Install with 'pip install pymupdf'")
    fitz = None

def load_css():
    css_file = os.path.join(os.path.dirname(__file__), 'static', 'styles.css')
    try:
        with open(css_file, encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        # Provide basic fallback CSS if there's an encoding issue
        fallback_css = """
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
            padding: 0.25rem 0.5rem;
            cursor: pointer;
        }
        """
        st.markdown(f'<style>{fallback_css}</style>', unsafe_allow_html=True)
        logger.warning(f"Error loading CSS: {str(e)}")

def load_js():
    js_file = os.path.join(os.path.dirname(__file__), 'static', 'script.js')
    try:
        with open(js_file, encoding='utf-8') as f:
            js_content = f.read()
    except Exception as e:
        # Fallback to another encoding or use a hardcoded version of the script
        js_content = """
        // Fallback script
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
        logger.warning(f"Error loading JS: {str(e)}")
    
    # Inject the JavaScript using components.html
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

def main():
    try:
        load_css()
    except Exception as e:
        st.warning(f"Could not load custom styles: {str(e)}. Using default styles.")
    
    try:
        load_js()
    except Exception as e:
        st.warning(f"Could not load JavaScript functions: {str(e)}. Some interactive features may be limited.")

    if 'extracted_text' not in st.session_state:
        st.session_state['extracted_text'] = ''
    
    # Initialize models with progress indicator
    if 'models_initialized' not in st.session_state:
        with st.spinner('Initializing OCR models...'):
            try:
                if st.session_state.get('ocr_module') == 'ocr_module':
                    # Only initialize if using the old module
                    initialize_models()
                st.session_state['models_initialized'] = True
            except Exception as e:
                st.error(f"Error initializing models: {str(e)}")
                st.info("The app will continue with limited functionality.")

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

    tabs = st.tabs(["📄 Document Processing", "⚙️ OCR Settings", "❓ Q&A Interface"])

    # OCR settings in session state
    if 'ocr_engine' not in st.session_state:
        st.session_state['ocr_engine'] = 'combined'
    if 'preserve_layout' not in st.session_state:
        st.session_state['preserve_layout'] = True

    with tabs[0]:  # Document Processing Tab
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Upload Document")
            uploaded_file = st.file_uploader(
                "Choose an image or PDF file",
                type=["jpg", "jpeg", "png", "pdf"],
                help="Supported formats: JPG, PNG, PDF"
            )
            if uploaded_file:
                if uploaded_file.type.startswith('image'):
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    # Check if image contains tables
                    try:
                        has_tables = detect_tables(image)
                        if has_tables:
                            st.info("📊 Table structures detected in the image. Layout preservation is recommended.")
                    except Exception as e:
                        logger.error(f"Error detecting tables: {e}")
                        pass  # Continue even if table detection fails
                    
                    with st.spinner('🔍 Performing OCR...'):
                        try:
                            extracted_text = perform_ocr(
                                image, 
                                engine=st.session_state['ocr_engine'], 
                                preserve_layout=st.session_state['preserve_layout']
                            )
                            
                            # Check if the OCR returned an error message
                            if isinstance(extracted_text, str) and extracted_text.startswith("Error:"):
                                st.error(extracted_text)
                                st.info("Trying fallback OCR methods...")
                                
                                # Try different engines as fallback
                                available_engines = st.session_state['available_ocr_engines']
                                
                                # Map UI names to engine parameter values
                                engine_map = {
                                    "PaddleOCR": "paddle",
                                    "EasyOCR": "easy",
                                    "Tesseract": "tesseract"
                                }
                                
                                # Try each available engine
                                for engine_name in available_engines:
                                    fallback_engine = engine_map.get(engine_name)
                                    if fallback_engine and fallback_engine != st.session_state['ocr_engine']:
                                        with st.spinner(f'Trying {engine_name} OCR engine...'):
                                            fallback_text = perform_ocr(
                                                image,
                                                engine=fallback_engine,
                                                preserve_layout=st.session_state['preserve_layout']
                                            )
                                            if not (isinstance(fallback_text, str) and fallback_text.startswith("Error:")):
                                                extracted_text = fallback_text
                                                st.success(f"Successfully extracted text using {engine_name} engine")
                                                break
                        except Exception as e:
                            st.error(f"OCR processing error: {str(e)}")
                            extracted_text = f"Error during OCR processing: {str(e)}"
                        
                        st.session_state['extracted_text'] = extracted_text
                            
                elif uploaded_file.type == 'application/pdf':
                    with st.spinner('🔍 Performing OCR on PDF...'):
                        try:
                            pdf_data = uploaded_file.read()
                            doc = fitz.open(stream=pdf_data, filetype="pdf")
                            extracted_text = ""
                            total_pages = len(doc)
                            
                            # Add a progress bar for PDF processing
                            progress_bar = st.progress(0)
                            
                            for i, page in enumerate(doc):
                                pix = page.get_pixmap()
                                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                try:
                                    page_text = perform_ocr(
                                        img, 
                                        engine=st.session_state['ocr_engine'],
                                        preserve_layout=st.session_state['preserve_layout']
                                    )
                                    
                                    # Check if OCR failed and try fallback
                                    if isinstance(page_text, str) and page_text.startswith("Error:"):
                                        st.warning(f"OCR failed on page {i+1}: {page_text}")
                                        st.info(f"Trying fallback OCR for page {i+1}...")
                                        
                                        # Try alternative engine
                                        available_engines = st.session_state['available_ocr_engines']
                                        engine_map = {
                                            "PaddleOCR": "paddle",
                                            "EasyOCR": "easy",
                                            "Tesseract": "tesseract"
                                        }
                                        
                                        for engine_name in available_engines:
                                            alt_engine = engine_map.get(engine_name)
                                            if alt_engine and alt_engine != st.session_state['ocr_engine']:
                                                page_text = perform_ocr(
                                                    img,
                                                    engine=alt_engine,
                                                    preserve_layout=st.session_state['preserve_layout']
                                                )
                                                if not (isinstance(page_text, str) and page_text.startswith("Error:")):
                                                    st.success(f"Used {engine_name} as fallback for page {i+1}")
                                                    break
                                except Exception as e:
                                    st.error(f"Error processing page {i+1}: {str(e)}")
                                    page_text = f"[Error on page {i+1}: {str(e)}]"
                                
                                extracted_text += f"--- PAGE {i+1} ---\n{page_text}\n\n"
                                progress_bar.progress((i + 1) / total_pages)
                            
                            st.session_state['extracted_text'] = extracted_text
                        except Exception as e:
                            st.error(f"Error processing PDF: {str(e)}")
                            st.session_state['extracted_text'] = f"Error processing PDF: {str(e)}"
                else:
                    st.error("Unsupported file type!")
                    return

        with col2:
            if st.session_state['extracted_text']:
                st.markdown("### Extracted Text")
                
                # Add options for displaying the text
                view_format = st.radio(
                    "View format", 
                    options=["Formatted", "Plain text"],
                    horizontal=True
                )
                
                # Display the text based on the selected format
                if view_format == "Formatted":
                    # Generate a unique ID for this text block
                    text_id = f"output-text-{hash(st.session_state['extracted_text'])}"
                    
                    st.markdown(f"""
                        <div class="formatted-text">
                            <pre id="{text_id}" style="white-space: pre-wrap;">{st.session_state['extracted_text']}</pre>
                            <button class="copy-btn" onclick="copyText(this)" aria-label="Copy text">📋</button>
                        </div>
                        <script>
                        // Ensure the button works immediately
                        document.querySelectorAll('.copy-btn').forEach(function(btn) {{
                            btn.addEventListener('click', function() {{
                                copyText(this);
                            }});
                        }});
                        </script>
                    """, unsafe_allow_html=True)
                else:
                    st.text_area("Plain text", st.session_state['extracted_text'], height=400)
                
                # Add file format options for download
                file_format = st.selectbox(
                    "Download format",
                    options=["TXT", "JSON", "Markdown"],
                    index=0
                )
                
                if file_format == "TXT":
                    download_data = st.session_state['extracted_text']
                    file_name = "extracted_text.txt"
                elif file_format == "JSON":
                    import json
                    download_data = json.dumps({"text": st.session_state['extracted_text']})
                    file_name = "extracted_text.json"
                else:  # Markdown
                    download_data = f"# Extracted Text\n\n```\n{st.session_state['extracted_text']}\n```"
                    file_name = "extracted_text.md"
                
                st.download_button(
                    "💾 Download Text",
                    download_data,
                    file_name=file_name,
                    help=f"Download the extracted text as a {file_format} file"
                )
    
    with tabs[1]:  # OCR Settings Tab
        st.markdown("### OCR Engine Settings")
        
        # OCR engine selection based on available engines
        available_engines = st.session_state.get('available_ocr_engines', [])
        engine_options = []
        
        if "PaddleOCR" in available_engines:
            engine_options.append("PaddleOCR (Recommended)")
        if "EasyOCR" in available_engines:
            engine_options.append("EasyOCR")
        if "Tesseract" in available_engines:
            engine_options.append("Tesseract")
            
        # Always include Combined option if at least one engine is available
        if engine_options:
            engine_options.append("Combined (Best results)")
        
        if not engine_options:
            st.error("No OCR engines are available. Please install at least one OCR engine.")
            engine_options = ["No engines available"]
        
        engine = st.radio(
            "Select OCR Engine",
            options=engine_options,
            index=len(engine_options) - 1 if engine_options else 0,
            horizontal=True
        )
        
        # Map UI options to engine parameter values
        engine_map = {
            "PaddleOCR (Recommended)": "paddle",
            "EasyOCR": "easy",
            "Tesseract": "tesseract",
            "Combined (Best results)": "combined",
            "No engines available": "auto"
        }
        
        st.session_state['ocr_engine'] = engine_map[engine]
        
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
        - **PaddleOCR**: Fast and accurate, optimized for Asian languages but works well for English.
        - **EasyOCR**: Good general-purpose OCR with support for 80+ languages.
        - **Tesseract**: Widely used open-source OCR engine, good for clean documents.
        - **Combined**: Uses multiple engines and selects the best result (recommended but slower).
        """)

    with tabs[2]:  # Q&A Interface Tab
        if 'extracted_text' in st.session_state and st.session_state['extracted_text'] and st.session_state.get('rag_available', False):
            st.markdown("### Ask Questions")
            query = st.text_input("Enter your question about the document")
            if query:
                with st.spinner('🤔 Finding answer...'):
                    try:
                        result = process_query(st.session_state['extracted_text'], query)
                        st.markdown(f"**Answer:** {result['answer']}")
                        
                        # Show confidence score
                        confidence = result.get('confidence', 0) * 100
                        st.progress(min(confidence / 100, 1.0))
                        st.caption(f"Confidence: {confidence:.1f}%")
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
                        st.info("The Q&A functionality may be limited due to missing dependencies.")
        elif not st.session_state.get('rag_available', False):
            st.info("Q&A functionality is not available. The RAG module could not be loaded.")
        else:
            st.info("Please upload and process a document first.")

if __name__ == '__main__':
    main()
