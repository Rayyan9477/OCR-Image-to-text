import streamlit as st
from PIL import Image
import io
import PyPDF2
from ocr_module import perform_ocr
from rag_module import process_query
import pandas as pd
import base64

def load_css():
    with open('static/styles.css') as f:
        return f'<style>{f.read()}</style>'

def load_js():
    with open('static/script.js') as f:
        return f'<script>{f.read()}</script>'

def format_output(text):
    """Format text with preserved formatting"""
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        leading_spaces = len(line) - len(line.lstrip())
        indentation = '&nbsp;' * leading_spaces
        formatted_line = (indentation + line.lstrip()
                          .replace(' ', '&nbsp;')
                          .replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;'))
        formatted_lines.append(formatted_line)
    
    return f"""
    <div class="card">
        <div class="formatted-text">
            <button class="copy-btn" onclick="copyText(this)">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 0 0 1 2-2h2"></path>
                    <rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect>
                </svg>
            </button>
            {'<br>'.join(formatted_lines)}
        </div>
    </div>
    """

def add_footer():
    footer_html = """
    <div class="footer">
        <p>
            <a href="https://github.com/Rayyan9477" target="_blank">GitHub</a> |
            <a href="https://www.linkedin.com/in/rayyan-ahmed9477/" target="_blank">LinkedIn</a> |
            <a href="mailto:rayyanahmed265@yahoo.com">Email</a>
        </p>
        <p>© 2024 Rayyan Ahmed. All rights reserved.</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Smart Document Analysis", page_icon="📝", layout="wide")
    st.markdown(load_css(), unsafe_allow_html=True)
    st.markdown(load_js(), unsafe_allow_html=True)
    
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px;">
            <img src="data:image/svg+xml,%3Csvg width='32' height='32' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5' stroke='%23667eea' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E" 
                 alt="App Icon" style="width: 40px; height: 40px;">
            <h1>Smart Document Analysis</h1>
        </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["📄 Document Processing", "❓ Q&A Interface"])
    
    with tabs[0]:
        col1, col2 = st.columns([1, 1])
        
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
                    with st.spinner('📑 Extracting text from PDF...'):
                        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
                        extracted_text = ""
                        for page in pdf_reader.pages:
                            extracted_text += page.extract_text() + "\n"
                else:
                    st.error("Unsupported file type!")
                    return
                st.session_state['extracted_text'] = extracted_text
        
        with col2:
            if 'extracted_text' in st.session_state:
                st.markdown("### Extracted Text")
                st.markdown(
                    format_output(st.session_state['extracted_text']), 
                    unsafe_allow_html=True
                )
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
                st.markdown(f"""
                    <div class="qa-section">
                        <h3>Answer</h3>
                        <p>{result['answer']}</p>
                        <div class="confidence-bar" style="background: linear-gradient(to right, #667eea {result['confidence']*100}%, transparent {(1 - result['confidence'])*100}%); height: 4px; border-radius: 2px;"></div>
                        <p style="color: #666; font-size: 0.9em;">Confidence: {result['confidence']:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
                with st.expander("Show Context"):
                    st.markdown(f"```\n{result['context']}\n```")
                if 'qa_history' not in st.session_state:
                    st.session_state['qa_history'] = []
                st.session_state['qa_history'].append({
                    'Question': query,
                    'Answer': result['answer'],
                    'Confidence': f"{result['confidence']:.2%}"
                })
                if st.session_state['qa_history']:
                    st.markdown("### Previous Questions")
                    df = pd.DataFrame(st.session_state['qa_history'])
                    st.dataframe(df)
        else:
            st.warning("Please upload and process a document first.")
    
    add_footer()

if __name__ == '__main__':
    main()