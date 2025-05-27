"""
CLI module for the OCR application
"""

import argparse
import os
import sys
import logging
import json
import glob
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image

from ..core.ocr_engine import OCREngine
from ..config.settings import Settings

logger = logging.getLogger(__name__)

class OCRCLI:
    """
    Command Line Interface for the OCR application
    """
    
    def __init__(self):
        """Initialize CLI application"""
        self.settings = Settings()
        self.ocr_engine = OCREngine(self.settings)
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse command line arguments
        
        Args:
            args: Command line arguments (if None, use sys.argv)
            
        Returns:
            Parsed arguments
        """
        parser = argparse.ArgumentParser(
            description="OCR Command Line Tool - Extract text from images and PDFs",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Input options
        input_group = parser.add_argument_group('Input Options')
        input_group.add_argument('--input', '-i', type=str, required=False,
                              help='Path to input image or PDF file')
        input_group.add_argument('--batch', '-b', type=str, required=False,
                              help='Path to directory containing images/PDFs to process')
        
        # Output options
        output_group = parser.add_argument_group('Output Options')
        output_group.add_argument('--output', '-o', type=str,
                               help='Path to output file (if not specified, prints to stdout)')
        output_group.add_argument('--format', '-f', type=str, choices=['txt', 'json', 'md'], default='txt',
                               help='Output format: txt, json, or md (markdown) [default: txt]')
        
        # OCR options
        ocr_group = parser.add_argument_group('OCR Options')
        ocr_group.add_argument('--engine', '-e', type=str, 
                             choices=['auto', 'tesseract', 'easyocr', 'paddleocr', 'combined'],
                             default='auto',
                             help='OCR engine to use [default: auto]')
        ocr_group.add_argument('--no-layout', action='store_true',
                             help='Disable layout preservation')
        ocr_group.add_argument('--preprocess', action='store_true',
                             help='Apply image preprocessing')
                             
        # System options
        sys_group = parser.add_argument_group('System Options')
        sys_group.add_argument('--verbose', '-v', action='store_true',
                             help='Enable verbose logging')
        sys_group.add_argument('--check', action='store_true',
                             help='Check available OCR engines and exit')
        
        return parser.parse_args(args)
    
    def check_engines(self):
        """Check and display available OCR engines"""
        print("Checking available OCR engines...")
        available_engines = self.ocr_engine.enabled_engines
        
        if available_engines:
            print(f"✓ Available OCR engines: {', '.join(available_engines)}")
        else:
            print("⚠️ No OCR engines available. OCR functionality will be limited.")
            print("\nTo install OCR engines:")
            print("  - Tesseract: pip install pytesseract")
            print("  - EasyOCR: pip install easyocr")
            print("  - PaddleOCR: pip install paddlepaddle paddleocr")
            
            # System-specific instructions for Tesseract
            if sys.platform == 'win32':
                print("\nOn Windows, for Tesseract:")
                print("  Download and install from: https://github.com/UB-Mannheim/tesseract/wiki")
            elif sys.platform == 'darwin':
                print("\nOn macOS, for Tesseract:")
                print("  brew install tesseract")
            else:
                print("\nOn Linux, for Tesseract:")
                print("  sudo apt-get install tesseract-ocr")
    
    def process_file(self, file_path: str, args: argparse.Namespace) -> str:
        """
        Process a single file
        
        Args:
            file_path: Path to the file to process
            args: Command line arguments
            
        Returns:
            Extracted text
        """
        file_path = os.path.abspath(file_path)
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return f"Error: File not found: {file_path}"
        
        logger.info(f"Processing file: {file_path}")
        
        # Process based on file type
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            try:
                image = Image.open(file_path)
                text = self.ocr_engine.perform_ocr(
                    image,
                    engine=args.engine,
                    preserve_layout=not args.no_layout,
                    preprocess=args.preprocess
                )
                return text
            except Exception as e:
                logger.error(f"Error processing image {file_path}: {e}")
                return f"Error: {str(e)}"
                
        elif file_ext == '.pdf':
            try:
                import fitz  # PyMuPDF
                
                logger.info(f"Processing PDF: {file_path}")
                doc = fitz.open(file_path)
                full_text = []
                
                for i, page in enumerate(doc):
                    logger.info(f"Processing page {i+1}/{len(doc)}")
                    
                    # Get page as an image
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Perform OCR
                    try:
                        page_text = self.ocr_engine.perform_ocr(
                            img,
                            engine=args.engine,
                            preserve_layout=not args.no_layout,
                            preprocess=args.preprocess
                        )
                        full_text.append(f"--- Page {i+1} ---\n{page_text}")
                    except Exception as e:
                        logger.error(f"Error processing page {i+1}: {e}")
                        full_text.append(f"--- Page {i+1} ---\nError: {str(e)}")
                
                return "\n\n".join(full_text)
                
            except ImportError:
                logger.error("PyMuPDF (fitz) is not installed. Cannot process PDF files.")
                return "Error: PDF processing requires PyMuPDF. Install with 'pip install pymupdf'."
            except Exception as e:
                logger.error(f"Error processing PDF {file_path}: {e}")
                return f"Error: {str(e)}"
        else:
            return f"Error: Unsupported file type: {file_ext}"
    
    def process_batch(self, directory: str, args: argparse.Namespace) -> Dict[str, str]:
        """
        Process all compatible files in a directory
        
        Args:
            directory: Directory path
            args: Command line arguments
            
        Returns:
            Dictionary of {filename: extracted_text}
        """
        directory = os.path.abspath(directory)
        
        if not os.path.isdir(directory):
            logger.error(f"Directory not found: {directory}")
            return {"error": f"Directory not found: {directory}"}
        
        logger.info(f"Processing files in directory: {directory}")
        
        # Find all compatible files
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.pdf']
        files = []
        
        for ext in extensions:
            pattern = os.path.join(directory, f"*{ext}")
            files.extend(glob.glob(pattern))
            
            # Also check uppercase extensions
            pattern = os.path.join(directory, f"*{ext.upper()}")
            files.extend(glob.glob(pattern))
        
        if not files:
            logger.warning(f"No compatible files found in {directory}")
            return {"error": f"No compatible files found in {directory}"}
        
        # Process each file
        results = {}
        for file_path in files:
            file_name = os.path.basename(file_path)
            logger.info(f"Processing {file_name}")
            
            try:
                text = self.process_file(file_path, args)
                results[file_name] = text
            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")
                results[file_name] = f"Error: {str(e)}"
        
        return results
    
    def format_output(self, text: Union[str, Dict[str, str]], format_type: str) -> str:
        """
        Format OCR output based on specified format
        
        Args:
            text: Extracted text or dictionary of results
            format_type: Format type (txt, json, md)
            
        Returns:
            Formatted output
        """
        if format_type == 'json':
            if isinstance(text, dict):
                return json.dumps(text, indent=2)
            else:
                return json.dumps({"text": text}, indent=2)
                
        elif format_type == 'md':
            if isinstance(text, dict):
                # Format as markdown
                md_lines = ["# OCR Results\n"]
                
                for filename, content in text.items():
                    md_lines.append(f"## {filename}\n")
                    md_lines.append("```")
                    md_lines.append(content)
                    md_lines.append("```\n")
                
                return "\n".join(md_lines)
            else:
                # Format single file as markdown
                return f"# OCR Result\n\n```\n{text}\n```"
                
        else:  # txt (default)
            if isinstance(text, dict):
                # Format as plain text with separators
                txt_lines = ["OCR RESULTS", "=" * 80]
                
                for filename, content in text.items():
                    txt_lines.append(f"\nFILE: {filename}")
                    txt_lines.append("-" * 80)
                    txt_lines.append(content)
                    txt_lines.append("=" * 80)
                
                return "\n".join(txt_lines)
            else:
                # Return as is for plain text
                return text
    
    def save_output(self, text: str, output_path: str) -> bool:
        """
        Save output to file
        
        Args:
            text: Text to save
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
                
            logger.info(f"Output saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving output: {e}")
            return False
    
    def run(self, args: Optional[List[str]] = None):
        """
        Run the CLI application
        
        Args:
            args: Command line arguments (if None, use sys.argv)
        """
        parsed_args = self.parse_args(args)
        
        # Configure logging
        log_level = logging.DEBUG if parsed_args.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Just check engines if requested
        if parsed_args.check:
            self.check_engines()
            return 0
            
        # Validate inputs
        if not parsed_args.input and not parsed_args.batch:
            logger.error("No input specified. Use --input or --batch.")
            return 1
            
        # Process input
        if parsed_args.batch:
            results = self.process_batch(parsed_args.batch, parsed_args)
        else:
            text = self.process_file(parsed_args.input, parsed_args)
            results = text
            
        # Format output
        formatted_output = self.format_output(results, parsed_args.format)
            
        # Output results
        if parsed_args.output:
            if self.save_output(formatted_output, parsed_args.output):
                print(f"Results saved to {parsed_args.output}")
                return 0
            else:
                print("Failed to save results")
                return 1
        else:
            # Print to stdout
            print(formatted_output)
            return 0
