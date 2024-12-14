import streamlit as st
from PIL import Image
import io
import fitz  # PyMuPDF for better PDF handling
from ocr_module import perform_ocr
from rag_module import process_query
import base64

def load_css():
    with open('static/styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    load_css()
    
    st.markdown("""
        <div class="header">
            <h1>Intelligent OCR and Text Analysis Tool</h1>
        </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["📄 Document Processing", "❓ Q&A Interface"])

    with tabs[0]:
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
                    with st.spinner('🔍 Performing OCR...'):
                        extracted_text = perform_ocr(image)
                elif uploaded_file.type == 'application/pdf':
                    with st.spinner('🔍 Performing OCR on PDF...'):
                        pdf_data = uploaded_file.read()
                        doc = fitz.open(stream=pdf_data, filetype="pdf")
                        extracted_text = ""
                        for page in doc:
                            pix = page.get_pixmap()
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            extracted_text += perform_ocr(img) + "\n"
                else:
                    st.error("Unsupported file type!")
                    return
                st.session_state['extracted_text'] = extracted_text

        with col2:
            if 'extracted_text' in st.session_state:
                st.markdown("### Extracted Text")
                extracted_text_html = st.session_state['extracted_text'].replace('\n', '<br>')
                st.markdown(f"""
                    <div class="text-container">
                        <pre>{extracted_text_html}</pre>
                        <button class="copy-btn" onclick="copyText()">📋</button>
                    </div>
                    <script>
                        function copyText() {{
                            const text = `{st.session_state['extracted_text']}`;
                            navigator.clipboard.writeText(text).then(() => {{
                                alert('Text copied to clipboard');
                            }}).catch(err => {{
                                console.error('Failed to copy text: ', err);
                            }});
                        }}
                    </script>
                """, unsafe_allow_html=True)
                st.download_button(
                    "💾 Download Text",
                    st.session_state['extracted_text'],
                    file_name="extracted_text.txt",
                    help="Download the extracted text as a file"
                )

    with tabs[1]:
        if 'extracted_text' in st.session_state:
            st.markdown("### Ask Questions")
            query = st.text_input("Enter your question about the document")
            if query:
                with st.spinner('🤔 Finding answer...'):
                    result = process_query(st.session_state['extracted_text'], query)
                st.markdown(f"**Answer:** {result}")
        else:
            st.info("Please upload and process a document first.")

if __name__ == '__main__':
    main()