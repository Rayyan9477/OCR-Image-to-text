import os
from typing import Optional, Dict, Any
import fitz
from PIL import Image
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self):
        self.available_engines = {
            'paddle': self._paddle_ocr,
            'combined': self._paddle_ocr  # Use PaddleOCR for combined mode too
        }
        self._initialize_engine()
        
    def _initialize_engine(self):
        """Initialize PaddleOCR with proper error handling"""
        try:
            # Initialize PaddleOCR
            from paddleocr import PaddleOCR
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                show_log=False,
                use_gpu=False  # Force CPU mode
            )
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            self.paddle_ocr = None
        
    def perform_ocr(self, image: Image.Image, engine: str = 'paddle', preserve_layout: bool = True) -> str:
        """
        Perform OCR on the given image using PaddleOCR.
        
        Args:
            image: PIL Image object
            engine: OCR engine to use ('paddle' or 'combined')
            preserve_layout: Whether to preserve text layout
            
        Returns:
            Extracted text as string
        """
        if engine not in self.available_engines:
            return f"Error: Unsupported OCR engine '{engine}'"
            
        try:
            return self.available_engines[engine](image, preserve_layout)
        except Exception as e:
            logger.error(f"Error in OCR: {str(e)}")
            return f"Error: {str(e)}"
            
    def _paddle_ocr(self, image: Image.Image, preserve_layout: bool) -> str:
        """PaddleOCR implementation"""
        if self.paddle_ocr is None:
            return "Error: PaddleOCR not initialized"
            
        try:
            result = self.paddle_ocr.ocr(image)
            if not result or not result[0]:
                return "No text detected"
                
            if preserve_layout:
                return self._format_paddle_result(result)
            return ' '.join([line[1][0] for line in result[0]])
        except Exception as e:
            logger.error(f"PaddleOCR processing error: {str(e)}")
            return f"Error: PaddleOCR failed - {str(e)}"
        
    def _format_paddle_result(self, result: list) -> str:
        """Format PaddleOCR result preserving layout"""
        if not result or not result[0]:
            return ""
            
        # Sort by y-coordinate first, then x-coordinate
        sorted_result = sorted(result[0], key=lambda x: (x[0][0][1], x[0][0][0]))
        
        current_y = sorted_result[0][0][0][1]
        text_lines = []
        current_line = []
        
        for line in sorted_result:
            y_coord = line[0][0][1]
            text = line[1][0]
            
            # If y-coordinate changes significantly, start new line
            if abs(y_coord - current_y) > 10:
                if current_line:
                    text_lines.append(' '.join(current_line))
                current_line = [text]
                current_y = y_coord
            else:
                current_line.append(text)
                
        if current_line:
            text_lines.append(' '.join(current_line))
            
        return '\n'.join(text_lines)
        
    def process_pdf(self, pdf_data: bytes, engine: str = 'combined', preserve_layout: bool = True) -> str:
        """
        Process a PDF document and perform OCR on each page.
        
        Args:
            pdf_data: PDF file content as bytes
            engine: OCR engine to use
            preserve_layout: Whether to preserve text layout
            
        Returns:
            Extracted text from all pages
        """
        try:
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            extracted_text = ""
            
            for i, page in enumerate(doc):
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                page_text = self.perform_ocr(
                    img,
                    engine=engine,
                    preserve_layout=preserve_layout
                )
                
                extracted_text += f"--- PAGE {i+1} ---\n{page_text}\n\n"
                
            return extracted_text
        except Exception as e:
            return f"Error processing PDF: {str(e)}" 