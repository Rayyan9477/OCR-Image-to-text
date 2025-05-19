# Intelligent OCR and Text Analysis Tool

## Description

An advanced application that performs Optical Character Recognition (OCR) on images and PDFs, extracts text with layout preservation, and provides a question-answering interface based on the extracted content. It leverages machine learning models, state-of-the-art OCR engines, and modern NLP techniques to enable users to interactively query their documents.

## Techniques and Tools Used

- **Streamlit**: For building the interactive web application.
- **PyMuPDF (fitz)**: For improved PDF handling and processing.
- **Pillow (PIL)**: For image processing and manipulation.
- **EasyOCR**: Neural network-based OCR engine.
- **PaddleOCR**: State-of-the-art OCR system with high accuracy and language support.
- **OpenCV**: For advanced image preprocessing and layout analysis.
- **Pytesseract**: Tesseract OCR Python wrapper for orientation detection.
- **Transformers**: HuggingFace library for loading pre-trained models.
- **SentenceTransformers**: For generating sentence embeddings.
- **FAISS**: Facebook AI Similarity Search for efficient similarity search.
- **PyTorch**: Deep learning framework underpinning the ML models.

## Features

- **Multiple OCR Engines**: Choose between PaddleOCR, EasyOCR, or a combined approach for optimal results.
- **Layout Preservation**: Maintains the original document formatting, including line breaks and text positioning.
- **Image Preprocessing**: Automatically enhances images for better OCR accuracy.
- **Table Detection**: Identifies table structures in documents.
- **Format Output Options**: Download extracted text in various formats (TXT, JSON, Markdown).
- **Interactive Q&A**: Ask questions about the extracted text using the RAG (Retrieval-Augmented Generation) system.
- **Multi-page PDF Support**: Process multi-page PDFs with progress tracking.
- **Modern UI/UX**: Enhanced user interface with custom styling and interactive elements.

## Usage

1. Launch the application with `streamlit run app.py`
2. Upload an image or PDF document
3. Configure OCR settings (engine, layout preservation) if needed
4. View and interact with the extracted text
5. Download the text in your preferred format
6. Use the Q&A interface to ask questions about the document content

## OCR Engine Comparison

- **PaddleOCR**: Fast and accurate, particularly good for structured documents and Asian languages.
- **EasyOCR**: Good all-around OCR with support for 80+ languages.
- **Combined Mode**: Uses both engines and selects the best result for optimal accuracy.

## Advanced Settings

The application provides several advanced settings to fine-tune OCR performance:

- **Layout Preservation**: Toggle to maintain original document formatting.
- **Image Preprocessing**: Apply enhancements like adaptive thresholding and noise removal.
- **Language Selection**: Choose between English-only (faster) or multi-language support.

## Code Examples

### Using the OCR Module

```python
from ocr_module import perform_ocr
from PIL import Image

# Open an image
image = Image.open("document.jpg")

# Perform OCR with layout preservation
text = perform_ocr(image, engine="paddle", preserve_layout=True)
print(text)
```

### Processing PDF Documents

```python
import fitz  # PyMuPDF
from ocr_module import perform_ocr
from PIL import Image

# Open PDF
doc = fitz.open("document.pdf")
for page in doc:
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = perform_ocr(img, engine="combined", preserve_layout=True)
    print(text)
```

## Installation

1. Clone the repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Ensure system dependencies are installed:
   ```
   apt-get update && apt-get install -y $(cat packages.txt)
   ```
4. Run the application:
   ```
   streamlit run app.py
   ```

## Contact

For inquiries or feedback:

- **Email**: [rayyanahmed265@yahoo.com](mailto:rayyanahmed265@yahoo.com)
- **LinkedIn**: [Rayyan Ahmed](https://www.linkedin.com/in/rayyan-ahmed9477/)
- **GitHub**: [Rayyan9477](https://github.com/Rayyan9477)