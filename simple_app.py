# simple_app.py - A simplified version of the OCR app without heavy dependencies
import streamlit as st
from PIL import Image
import os
import sys
import time
import importlib
from ocr_module_lite import LiteOCR

# Set page configuration
st.set_page_config(page_title="OCR Tool", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-bottom: 1rem;
    }
    .formatted-text {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.25rem;
        padding: 1rem;
        white-space: pre-wrap;
        font-family: monospace;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("<h1 class='main-header'>OCR Image to Text Tool</h1>", unsafe_allow_html=True)
    st.markdown("<p>Extract text from images with advanced OCR technology</p>", unsafe_allow_html=True)
    
    # Check available OCR engines
    ocr = LiteOCR()
    available_engines = ocr.available_engines
    engines_list = [engine for engine, available in available_engines.items() if available]
    
    if not engines_list:
        st.error("No OCR engines are available. Please install at least one of: pytesseract, EasyOCR, or PaddleOCR.")
        st.info("You can install them with:\n```\npip install pytesseract easyocr paddleocr\n```")
        return
    
    # Add "combined" if multiple engines are available
    if len(engines_list) > 1:
        engines_list.append("combined")
    
    # Add "auto" option
    engines_list.insert(0, "auto")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h2 class='sub-header'>Upload Image</h2>", unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
        
        # OCR settings
        st.markdown("<h3>OCR Settings</h3>", unsafe_allow_html=True)
        
        # Engine selection
        engine = st.selectbox(
            "OCR Engine", 
            options=engines_list,
            format_func=lambda x: x.capitalize() if x != "auto" else "Auto (Best available)"
        )
        
        # Layout preservation
        preserve_layout = st.checkbox("Preserve text layout", value=True, 
                                    help="Maintains the original document layout including line breaks")
        
        # Process button
        process_button = st.button("Process Image")
        
    with col2:
        st.markdown("<h2 class='sub-header'>Extracted Text</h2>", unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Display the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process the image when the button is clicked
            if process_button:
                with st.spinner("Extracting text..."):
                    # Perform OCR
                    start_time = time.time()
                    extracted_text = ocr.perform_ocr(image, engine=engine, preserve_layout=preserve_layout)
                    end_time = time.time()
                    
                    # Display OCR results
                    st.markdown("<div class='formatted-text'>{}</div>".format(extracted_text), unsafe_allow_html=True)
                    
                    # Show processing time
                    st.caption(f"Processing time: {end_time - start_time:.2f} seconds")
                    
                    # Download button
                    st.download_button(
                        "Download Text",
                        data=extracted_text,
                        file_name="extracted_text.txt",
                        mime="text/plain"
                    )
    
    # Footer
    st.markdown("<div class='footer'>Developed by Rayyan Ahmed</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()