#!/usr/bin/env python
"""
Layout-Aware Enhanced Multi-Engine OCR System
Combines multiple OCR engines with layout preservation capabilities
"""

import os
import cv2
import numpy as np
from PIL import Image
import concurrent.futures
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import time
import threading
from dataclasses import dataclass

# Import the layout preservation module
try:
    from layout_preserving_ocr import LayoutPreservingOCR, integrate_with_enhanced_ocr
    LAYOUT_PRESERVATION_AVAILABLE = True
except ImportError:
    LAYOUT_PRESERVATION_AVAILABLE = False
    logging.warning("Layout preservation module not available")

# Import enhanced OCR
try:
    from enhanced_multi_engine_ocr import extract_text_enhanced_multi_engine
    ENHANCED_OCR_AVAILABLE = True
except ImportError:
    ENHANCED_OCR_AVAILABLE = False
    logging.warning("Enhanced multi-engine OCR not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_with_layout_preservation(image: Union[np.ndarray, Image.Image], 
                                        preserve_layout: bool = True) -> Dict[str, Any]:
    """
    Extract text with layout preservation using enhanced multi-engine OCR
    
    Args:
        image: Input image (PIL Image or numpy array)
        preserve_layout: Whether to preserve original layout and formatting
        
    Returns:
        Dict containing OCR results with layout information
    """
    try:
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_np = image.copy()
        
        if not ENHANCED_OCR_AVAILABLE:
            return {
                'error': 'Enhanced OCR system not available',
                'text': '',
                'layout_preserved_text': '',
                'formatting_applied': False
            }
        
        # Get enhanced OCR results
        enhanced_result = extract_text_enhanced_multi_engine(image)
        
        if not preserve_layout or not LAYOUT_PRESERVATION_AVAILABLE:
            # Return basic result without layout preservation
            enhanced_result['layout_preserved_text'] = enhanced_result.get('text', '')
            enhanced_result['formatting_applied'] = False
            return enhanced_result
        
        # Apply layout preservation
        try:
            # Prepare OCR results for layout analysis
            ocr_results_for_layout = {}
            
            # Extract raw results from individual engines for layout analysis
            for result_data in enhanced_result.get('individual_results', []):
                engine_name = result_data['engine']
                if result_data['status'] == 'success':
                    ocr_results_for_layout[engine_name] = {
                        'status': 'success',
                        'confidence': result_data['confidence'],
                        'text': result_data['text'],
                        'raw_results': []  # Will be populated if available
                    }
            
            # If we have the results dictionary with more detailed info
            if 'results' in enhanced_result:
                for engine_name, engine_data in enhanced_result['results'].items():
                    if engine_name in ocr_results_for_layout:
                        ocr_results_for_layout[engine_name].update(engine_data)
            
            # Apply layout preservation
            layout_ocr = LayoutPreservingOCR()
            layout_result = layout_ocr.extract_with_layout_preservation(image_np, ocr_results_for_layout)
            
            # Merge layout results with enhanced OCR results
            enhanced_result.update({
                'layout_preserved_text': layout_result['layout_preserved_text'],
                'layout_analysis': layout_result['layout_analysis'],
                'text_regions': layout_result.get('text_regions', []),
                'visual_layout': layout_result.get('visual_layout'),
                'formatting_applied': True,
                'original_text': enhanced_result.get('text', ''),
                'layout_engine_used': layout_result.get('best_engine', 'unknown')
            })
            
            logger.info(f"Layout preservation applied successfully using {layout_result.get('best_engine', 'unknown')} engine")
            
        except Exception as e:
            logger.error(f"Layout preservation failed: {e}")
            enhanced_result['layout_preserved_text'] = enhanced_result.get('text', '')
            enhanced_result['formatting_applied'] = False
            enhanced_result['layout_error'] = str(e)
        
        return enhanced_result
        
    except Exception as e:
        logger.error(f"Text extraction with layout preservation failed: {e}")
        return {
            'error': f'Text extraction failed: {str(e)}',
            'text': '',
            'layout_preserved_text': '',
            'formatting_applied': False,
            'processing_time': 0.0,
            'confidence': 0.0
        }

def create_formatted_output(ocr_result: Dict[str, Any], output_format: str = "preserved") -> str:
    """
    Create formatted output in various formats
    
    Args:
        ocr_result: OCR result dictionary
        output_format: "preserved", "plain", "structured", "markdown"
        
    Returns:
        Formatted text string
    """
    try:
        if output_format == "preserved" and ocr_result.get('formatting_applied', False):
            return ocr_result.get('layout_preserved_text', '')
        
        elif output_format == "plain":
            return ocr_result.get('text', '')
        
        elif output_format == "structured":
            # Create structured output with metadata
            text = ocr_result.get('layout_preserved_text', ocr_result.get('text', ''))
            metadata = []
            
            if 'layout_analysis' in ocr_result:
                analysis = ocr_result['layout_analysis']
                metadata.append(f"Lines: {analysis.get('total_lines', 0)}")
                metadata.append(f"Columns: {analysis.get('columns_detected', 0)}")
                metadata.append(f"Titles: {analysis.get('titles_detected', 0)}")
                metadata.append(f"Bullets: {analysis.get('bullet_points', 0)}")
            
            if metadata:
                return f"DOCUMENT ANALYSIS:\n{' | '.join(metadata)}\n\n{text}"
            else:
                return text
        
        elif output_format == "markdown":
            # Convert preserved layout to markdown format
            text = ocr_result.get('layout_preserved_text', ocr_result.get('text', ''))
            
            # Simple markdown conversion
            lines = text.split('\n')
            markdown_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    markdown_lines.append('')
                    continue
                
                # Check for title patterns
                if line.isupper() and len(line.split()) <= 6:
                    markdown_lines.append(f"# {line}")
                elif line.startswith('•') or line.startswith('-'):
                    markdown_lines.append(f"- {line[1:].strip()}")
                elif any(line.startswith(f"{i}.") for i in range(1, 10)):
                    markdown_lines.append(f"1. {line[2:].strip()}")
                else:
                    markdown_lines.append(line)
            
            return '\n'.join(markdown_lines)
        
        else:
            return ocr_result.get('text', '')
            
    except Exception as e:
        logger.error(f"Format creation failed: {e}")
        return ocr_result.get('text', '')

def compare_formatting_results(image: Union[np.ndarray, Image.Image]) -> Dict[str, str]:
    """
    Compare different formatting approaches
    
    Returns:
        Dictionary with different formatted versions
    """
    try:
        # Get OCR result with layout preservation
        result = extract_text_with_layout_preservation(image, preserve_layout=True)
        
        return {
            'original_ocr': result.get('text', ''),
            'layout_preserved': result.get('layout_preserved_text', ''),
            'plain_format': create_formatted_output(result, 'plain'),
            'structured_format': create_formatted_output(result, 'structured'),
            'markdown_format': create_formatted_output(result, 'markdown'),
            'formatting_applied': result.get('formatting_applied', False),
            'layout_analysis': result.get('layout_analysis', {}),
            'confidence': result.get('confidence', 0.0),
            'processing_time': result.get('processing_time', 0.0)
        }
        
    except Exception as e:
        logger.error(f"Formatting comparison failed: {e}")
        return {
            'error': str(e),
            'original_ocr': '',
            'layout_preserved': '',
            'plain_format': '',
            'structured_format': '',
            'markdown_format': '',
            'formatting_applied': False
        }

# Convenience functions for different use cases
def extract_text_simple(image: Union[np.ndarray, Image.Image]) -> str:
    """Simple text extraction without layout preservation"""
    result = extract_text_with_layout_preservation(image, preserve_layout=False)
    return result.get('text', '')

def extract_text_formatted(image: Union[np.ndarray, Image.Image]) -> str:
    """Text extraction with layout preservation"""
    result = extract_text_with_layout_preservation(image, preserve_layout=True)
    return result.get('layout_preserved_text', result.get('text', ''))

def extract_text_markdown(image: Union[np.ndarray, Image.Image]) -> str:
    """Text extraction formatted as markdown"""
    result = extract_text_with_layout_preservation(image, preserve_layout=True)
    return create_formatted_output(result, 'markdown')

def extract_text_structured(image: Union[np.ndarray, Image.Image]) -> str:
    """Text extraction with structural analysis"""
    result = extract_text_with_layout_preservation(image, preserve_layout=True)
    return create_formatted_output(result, 'structured')

# Testing and validation
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            # Load image
            image = cv2.imread(image_path)
            
            print("🔍 Layout-Aware OCR Processing")
            print("=" * 50)
            
            # Compare different formatting approaches
            results = compare_formatting_results(image)
            
            print(f"Processing Time: {results.get('processing_time', 0):.2f}s")
            print(f"Confidence: {results.get('confidence', 0):.2f}")
            print(f"Layout Preservation: {results.get('formatting_applied', False)}")
            
            if 'layout_analysis' in results and results['layout_analysis']:
                analysis = results['layout_analysis']
                print(f"\nLayout Analysis:")
                print(f"  Lines: {analysis.get('total_lines', 0)}")
                print(f"  Columns: {analysis.get('columns_detected', 0)}")
                print(f"  Titles: {analysis.get('titles_detected', 0)}")
                print(f"  Bullet Points: {analysis.get('bullet_points', 0)}")
            
            print("\n" + "="*50)
            print("ORIGINAL OCR OUTPUT:")
            print("="*50)
            print(results['original_ocr'])
            
            print("\n" + "="*50)
            print("LAYOUT-PRESERVED OUTPUT:")
            print("="*50)
            print(results['layout_preserved'])
            
            print("\n" + "="*50)
            print("MARKDOWN FORMAT:")
            print("="*50)
            print(results['markdown_format'])
            
        else:
            print(f"Image file not found: {image_path}")
    else:
        print("Usage: python layout_aware_ocr.py <image_path>")
        print("\nFeatures:")
        print("- Layout preservation")
        print("- Multiple output formats")
        print("- Structural analysis")
        print("- Markdown conversion")
