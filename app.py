# Initialize imports, environment variables and dependencies
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
import fitz

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import UI components
from src.ui.components import display_ocr_settings, display_extracted_text, display_qa_interface

# Set page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="OCR Image to Text",
    page_icon="📝",
    layout="wide"
)

# Initialize session state for error tracking
if 'dependency_errors' not in st.session_state:
    st.session_state['dependency_errors'] = []

# Set environment variables for TensorFlow/Keras compatibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_USE_LEGACY_KERAS'] = '1'   # Use legacy Keras
os.environ['KERAS_BACKEND'] = 'tensorflow'  # Set Keras backend to TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU for compatibility

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_css():
    """Load custom CSS styles"""
    st.markdown("""
        <style>
        .header {
            text-align: center;
            padding: 1rem;
            background-color: #f0f2f6;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
        }
        .header h1 {
            color: #1f77b4;
            margin-bottom: 0.5rem;
        }
        .header p {
            color: #666;
            margin: 0.5rem 0;
        }
        .formatted-text {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 0.25rem;
            padding: 1rem;
            white-space: pre-wrap;
            font-family: monospace;
        }
        .stButton>button {
            width: 100%;
            margin-top: 1rem;
        }
        .stProgress>div>div>div {
            background-color: #1f77b4;
        }
        </style>
    """, unsafe_allow_html=True)

def load_js():
    """Load custom JavaScript functions"""
    st.markdown("""
        <script>
        function copyText(button) {
            const textElement = button.parentElement.querySelector('pre');
            const text = textElement.textContent;
            
            navigator.clipboard.writeText(text).then(() => {
                const originalText = button.textContent;
                button.textContent = '✓';
                setTimeout(() => {
                    button.textContent = originalText;
                }, 2000);
            });
        }
        </script>
    """, unsafe_allow_html=True)

def check_dependencies():
    """Check and initialize required dependencies"""
    try:
        # Try to import TensorFlow first
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging
        
        # Initialize TensorFlow session
        tf.compat.v1.disable_eager_execution()
        
        # Import OCR components
        from src.core.ocr_engine import OCREngine
        from src.core.model_manager import ModelManager
        from src.ui.components import display_ocr_settings, display_extracted_text, display_qa_interface
        
        return True, None
    except ImportError as e:
        error_msg = f"Failed to import required modules: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Error during initialization: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def initialize_models():
    """Initialize OCR models with proper error handling"""
    try:
        # Initialize OCR components
        from src.core.ocr_engine import OCREngine
        from src.core.model_manager import ModelManager
        
        # Create instances
        model_manager = ModelManager()
        ocr_engine = OCREngine()
        
        # Check if OCR engines are available
        if ocr_engine.paddle_ocr is None and ocr_engine.easy_ocr is None:
            st.error("No OCR engines could be initialized. Please check the installation of PaddleOCR and EasyOCR.")
            return None, None
        
        # Store in session state
        st.session_state['model_manager'] = model_manager
        st.session_state['ocr_engine'] = ocr_engine
        
        return model_manager, ocr_engine
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        st.error(f"Error initializing models: {str(e)}")
        return None, None

def main():
    """Main application entry point"""
    # Initialize variables
    model_manager = None
    ocr_engine = None
    
    try:
        # Load custom CSS and JS
        load_css()
        load_js()
    except Exception as e:
        st.warning(f"Could not load custom styles: {str(e)}. Using default styles.")

    # Initialize session state
    if 'extracted_text' not in st.session_state:
        st.session_state['extracted_text'] = ''
    
    # Initialize models with progress indicator
    if 'models_initialized' not in st.session_state:
        with st.spinner('Initializing OCR models... This may take a few minutes on first run.'):
            try:
                model_manager, ocr_engine = initialize_models()
                if model_manager is not None and ocr_engine is not None:
                    st.session_state['models_initialized'] = True
                    st.success("OCR models initialized successfully!")
                else:
                    st.warning("Some components failed to initialize. The app will continue with limited functionality.")
            except Exception as e:
                st.error(f"Error initializing models: {str(e)}")
                st.info("The app will continue with limited functionality.")

    # Display header
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

    # Set up tabs
    tabs = st.tabs(["OCR Processing", "Settings", "Q&A"])
    
    with tabs[0]:  # OCR Processing Tab
        st.title("OCR Image to Text")
        st.markdown("Upload an image or PDF to extract text using OCR")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['png', 'jpg', 'jpeg', 'pdf'],
                help="Supported formats: PNG, JPG, JPEG, PDF"
            )
            
            if uploaded_file is not None:
                if uploaded_file.type.startswith('image/'):
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    if st.button("Extract Text"):
                        with st.spinner("Processing image..."):
                            try:
                                # Use the OCR engine from session state
                                ocr_engine = st.session_state.get('ocr_engine')
                                if ocr_engine is None:
                                    ocr_engine = OCREngine()
                                    st.session_state['ocr_engine'] = ocr_engine
                                
                                # Get OCR settings
                                settings = display_ocr_settings(st.session_state.get('model_manager'))
                                
                                # Process the image using perform_ocr
                                extracted_text = ocr_engine.perform_ocr(
                                    image,
                                    engine=settings['engine'],
                                    preserve_layout=settings['preserve_layout']
                                )
                                
                                # Store in session state
                                st.session_state['extracted_text'] = extracted_text
                                
                                # Display results
                                st.success("Text extracted successfully!")
                                display_extracted_text(extracted_text)
                                
                            except Exception as e:
                                st.error(f"Error processing image: {str(e)}")
                                
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
                                
                                # Process each page
                                page_text = ocr_engine.perform_ocr(
                                    img,
                                    engine=settings['engine'],
                                    preserve_layout=settings['preserve_layout']
                                )
                                
                                extracted_text += f"--- PAGE {i+1} ---\n{page_text}\n\n"
                                
                                # Update progress
                                progress = (i + 1) / total_pages
                                progress_bar.progress(progress)
                            
                            # Store in session state
                            st.session_state['extracted_text'] = extracted_text
                            
                            # Display results
                            st.success("PDF processing completed!")
                            display_extracted_text(extracted_text)
                            
                        except Exception as e:
                            st.error(f"Error processing PDF: {str(e)}")
    
    with tabs[1]:  # Settings Tab
        st.title("OCR Settings")
        if model_manager is not None:
            display_ocr_settings(model_manager)
        else:
            st.warning("OCR settings are not available. Please wait for initialization to complete.")
    
    with tabs[2]:  # Q&A Tab
        st.title("Document Q&A")
        if 'extracted_text' in st.session_state:
            display_qa_interface(st.session_state['extracted_text'])
        else:
            st.info("Please upload and process a document first.")

if __name__ == "__main__":
    main()