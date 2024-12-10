# Intelligent OCR and Text Analysis Tool

## Description

An advanced application that performs Optical Character Recognition (OCR) on images and PDFs, extracts text, and provides a question-answering interface based on the extracted content. It leverages machine learning models and modern NLP techniques to enable users to interactively query their documents.

## Techniques and Tools Used

- **Streamlit**: For building the interactive web application.
- **PyPDF2**: To read and extract text from PDF files.
- **Pillow (PIL)**: For image processing and manipulation.
- **OCR Module**: Custom module (`ocr_module.py`) for performing OCR on images.
- **RAG Module**: Custom module (`rag_module.py`) implementing Retrieval-Augmented Generation for processing queries.
- **Transformers**: HuggingFace library for loading pre-trained models.
- **SentenceTransformers**: For generating sentence embeddings.
- **PyTorch**: Deep learning framework underpinning the ML models.

## Features

- **Upload Images or PDFs**: Accepts multiple image formats and PDFs for text extraction.
- **Perform OCR**: Extracts text from images using the `perform_ocr` function.
- **Text Analysis**: Enables users to ask questions about the extracted text using the `process_query` function.
- **Custom Styling**: Utilizes custom CSS and JavaScript for an enhanced UI/UX.

## Code Snippets

**Loading Custom CSS**

```python
def load_css():
    with open('static/styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
```

**Handling File Uploads**

```python
def handle_file_upload(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    else:
        image = Image.open(uploaded_file)
        text = perform_ocr(image)
    return text
```

**Processing User Queries**

```python
def get_answer(query, context):
    answer = process_query(query, context)
    return answer
```

## Contact

For inquiries or feedback:

- **Email**: [rayyanahmed265@yahoo.com](mailto:rayyanahmed265@yahoo.com)
- **LinkedIn**: [Rayyan Ahmed](https://www.linkedin.com/in/rayyan-ahmed9477/)
- **GitHub**: [Rayyan9477](https://github.com/Rayyan9477)