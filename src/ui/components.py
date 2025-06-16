import streamlit as st
from typing import Optional, Dict, Any
import json

def display_ocr_settings(model_manager) -> Dict[str, Any]:
    """Display OCR settings and return selected options"""
    st.markdown("### OCR Engine Settings")
    
    # OCR engine selection
    engine = st.radio(
        "Select OCR Engine",
        options=["PaddleOCR (Recommended)"],
        index=0,
        horizontal=True
    )
    
    # Map UI options to engine parameter values
    engine_map = {
        "PaddleOCR (Recommended)": "paddle"
    }
    
    selected_engine = engine_map[engine]
    
    # Layout preservation option
    preserve_layout = st.checkbox(
        "Preserve text layout and formatting",
        value=True,
        help="Maintains the original document's layout including line breaks and approximate text positioning"
    )
    
    # Advanced settings
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
            options=["English"],
            index=0,
            help="Select the primary language of your document"
        )
    
    return {
        'engine': selected_engine,
        'preserve_layout': preserve_layout,
        'preprocessing': apply_preprocessing,
        'language': language
    }

def display_extracted_text(text: str):
    """Display extracted text with formatting options"""
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
        text_id = f"output-text-{hash(text)}"
        
        st.markdown(f"""
            <div class="formatted-text">
                <pre id="{text_id}" style="white-space: pre-wrap;">{text}</pre>
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
        st.text_area("Plain text", text, height=400)
    
    # Add file format options for download
    file_format = st.selectbox(
        "Download format",
        options=["TXT", "JSON", "Markdown"],
        index=0
    )
    
    if file_format == "TXT":
        download_data = text
        file_name = "extracted_text.txt"
    elif file_format == "JSON":
        download_data = json.dumps({"text": text})
        file_name = "extracted_text.json"
    else:  # Markdown
        download_data = f"# Extracted Text\n\n```\n{text}\n```"
        file_name = "extracted_text.md"
    
    st.download_button(
        "💾 Download Text",
        download_data,
        file_name=file_name,
        help=f"Download the extracted text as a {file_format} file"
    )

def display_qa_interface(text: str):
    """Display Q&A interface for document analysis"""
    if not text:
        st.info("Please upload and process a document first.")
        return
        
    st.markdown("### Ask Questions")
    query = st.text_input("Enter your question about the document")
    
    if query:
        with st.spinner('🤔 Finding answer...'):
            # TODO: Implement actual Q&A processing
            result = {
                'answer': "This is a placeholder answer. Q&A functionality needs to be implemented.",
                'confidence': 0.8
            }
            
        st.markdown(f"**Answer:** {result['answer']}")
        
        # Show confidence score
        confidence = result.get('confidence', 0) * 100
        st.progress(min(confidence / 100, 1.0))
        st.caption(f"Confidence: {confidence:.1f}%") 