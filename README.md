# Intelligent OCR and Text Analysis Tool

## Description

An advanced application that performs Optical Character Recognition (OCR) on images and PDFs, extracts text with layout preservation, and provides a question-answering interface based on the extracted content. It leverages machine learning models, state-of-the-art OCR engines, and modern NLP techniques to enable users to interactively query their documents.

## Features

- **Multiple OCR Engines**: Choose between PaddleOCR, EasyOCR, or a combined approach for optimal results
- **Layout Preservation**: Maintains the original document formatting, including line breaks and text positioning
- **Image Preprocessing**: Automatically enhances images for better OCR accuracy
- **Table Detection**: Identifies table structures in documents
- **Format Output Options**: Download extracted text in various formats (TXT, JSON, Markdown)
- **Interactive Q&A**: Ask questions about the extracted text using the RAG (Retrieval-Augmented Generation) system
- **Multi-page PDF Support**: Process multi-page PDFs with progress tracking
- **Modern UI/UX**: Enhanced user interface with custom styling and interactive elements
- **Robust Design**: Gracefully handles missing dependencies with fallbacks

## Installation

### Prerequisites

- Python 3.8+ recommended
- Pip package manager
- Optional: Tesseract OCR engine installed on your system (for fallback OCR)

### Basic Installation

1. Clone the repository:
   ```
   git clone https://github.com/youruser/OCR-Image-to-text.git
   cd OCR-Image-to-text
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. For optimal performance, install system dependencies:

   **For Windows:**
   - Download and install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
   - Add Tesseract installation directory to your PATH

   **For macOS:**
   ```
   brew install tesseract
   ```

   **For Linux:**
   ```
   sudo apt-get update
   sudo apt-get install -y tesseract-ocr
   ```

4. Check your installation:
   ```
   python run_ocr.py --setup
   ```

### Optimizing Installation

The system can work with just one OCR engine, but for best results, install multiple engines:

- **For best accuracy:** Install PaddleOCR AND EasyOCR
- **For lightweight usage:** Install only PyTesseract
- **For offline usage:** Install PyTesseract (no internet required)

## Usage

The application can be run in multiple modes:

### Web Interface Mode (Default)

The easiest way to use the application with a full graphical interface:

```
python run_ocr.py
```

or explicitly:

```
python run_ocr.py --web
```

### Command-Line Interface

Process files directly from the command line:

```
python run_ocr.py --cli --input image.jpg --output results.txt
```

Process multiple files in a directory:

```
python run_ocr.py --cli --batch ./images/ --output ./results/
```

Support for different output formats:

```
python run_ocr.py --cli --input document.pdf --format json
```

### Test Mode

Verify your OCR functionality:

```
python run_ocr.py --test
```

## OCR Engine Comparison

- **PaddleOCR**: Fast and accurate, particularly good for structured documents and Asian languages
- **EasyOCR**: Good all-around OCR with support for 80+ languages
- **Combined Mode**: Uses both engines and selects the best result for optimal accuracy
- **Tesseract**: Great for offline usage, no internet required, but less accurate on complex layouts

## Advanced Usage

### Using the OCR Module in Your Code

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

### Command-Line Options

```
usage: run_ocr.py [-h] [--web] [--cli] [--test] [--setup] [--input INPUT] [--output OUTPUT]
                  [--engine {paddle,easy,combined}] [--no-layout]
                  [--format {txt,json,md}] [--batch BATCH] [--check]

OCR Image-to-Text System

Mode Selection:
  --web, -w           Run in web interface mode (default)
  --cli, -c           Run in command-line interface mode
  --test, -t          Run test mode to verify OCR functionality
  --setup, -s         Setup and check dependencies

CLI Mode Options:
  --input INPUT, -i INPUT
                      Path to input image or PDF file
  --output OUTPUT, -o OUTPUT
                      Path to output text file
  --engine {paddle,easy,combined}, -e {paddle,easy,combined}
                      OCR engine to use (paddle, easy, or combined)
  --no-layout         Disable layout preservation
  --format {txt,json,md}, -f {txt,json,md}
                      Output format (txt, json, or md)
  --batch BATCH, -b BATCH
                      Process all files in a directory
  --check             Check dependencies and exit
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: If you encounter import errors, run `python run_ocr.py --setup` to check which dependencies are missing.

2. **OCR Engine Not Found**: The system will fall back to alternative engines if your primary choice isn't available.

3. **TensorFlow/Keras Compatibility**: The application handles TensorFlow/Keras compatibility issues automatically, but you might need to set environment variables manually in some environments:
   ```
   set TF_CPP_MIN_LOG_LEVEL=2
   set TF_USE_LEGACY_KERAS=1
   ```

4. **Tesseract Not Found**: Make sure Tesseract is installed and properly added to your system PATH.

## Technologies Used

- **Streamlit**: For building the interactive web application
- **PyMuPDF (fitz)**: For improved PDF handling and processing
- **Pillow (PIL)**: For image processing and manipulation
- **EasyOCR**: Neural network-based OCR engine
- **PaddleOCR**: State-of-the-art OCR system with high accuracy
- **OpenCV**: For advanced image preprocessing and layout analysis
- **Pytesseract**: Tesseract OCR Python wrapper
- **Transformers**: HuggingFace library for loaded pre-trained models
- **SentenceTransformers**: For generating sentence embeddings
- **FAISS**: Facebook AI Similarity Search for efficient similarity search
- **PyTorch**: Deep learning framework underpinning the ML models

## Contact

For inquiries or feedback:

- **Email**: [rayyanahmed265@yahoo.com](mailto:rayyanahmed265@yahoo.com)
- **LinkedIn**: [Rayyan Ahmed](https://www.linkedin.com/in/rayyan-ahmed9477/)
- **GitHub**: [Rayyan9477](https://github.com/Rayyan9477)