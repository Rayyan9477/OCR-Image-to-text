import streamlit as st
from PIL import Image
import io
import PyPDF2
from ocr_module import perform_ocr
from rag_module import process_query

st.set_page_config(page_title="RAG-based OCR App", layout="wide")

st.title("RAG-based OCR App")

uploaded_file = st.file_uploader("Choose an image or PDF file", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        extracted_text = perform_ocr(image)
    elif uploaded_file.type == 'application/pdf':
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
        extracted_text = ""
        for page in pdf_reader.pages:
            extracted_text += page.extract_text() + "\n"
    
    st.subheader("Extracted Text:")
    st.text(extracted_text)
    
    st.subheader("Ask a question about the extracted text:")
    query = st.text_input("Enter your question")
    
    if query:
        answer = process_query(extracted_text, query)
        st.subheader("Answer:")
        st.write(answer)