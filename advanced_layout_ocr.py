#!/usr/bin/env python
"""
Advanced Layout-Preserving OCR System
Preserves document formatting with enhanced multi-engine integration
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
import re
import time

# Import enhanced OCR
try:
    from enhanced_multi_engine_ocr import extract_text_enhanced_multi_engine
    ENHANCED_OCR_AVAILABLE = True
except ImportError:
    ENHANCED_OCR_AVAILABLE = False
    logging.warning("Enhanced multi-engine OCR not available")

logger = logging.getLogger(__name__)

class AdvancedLayoutPreserver:
    """Advanced system for preserving document layout and formatting"""
    
    def __init__(self):
        self.line_threshold = 15  # Pixels to consider same line
        self.column_threshold = 0.4  # Relative position for column detection
        
    def analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """Analyze text structure and identify formatting elements"""
        lines = text.split('\n') if '\n' in text else text.split('. ')
        
        structure = {
            'total_lines': len(lines),
            'titles': [],
            'bullet_points': [],
            'numbered_lists': [],
            'tables': [],
            'paragraphs': [],
            'has_columns': False
        }
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Detect titles (short, capitalized, or common title words)
            if self._is_title(line):
                structure['titles'].append({'line': i, 'text': line})
            
            # Detect bullet points
            elif self._is_bullet_point(line):
                structure['bullet_points'].append({'line': i, 'text': line})
            
            # Detect numbered lists
            elif self._is_numbered_item(line):
                structure['numbered_lists'].append({'line': i, 'text': line})
            
            # Detect table-like content
            elif self._is_table_row(line):
                structure['tables'].append({'line': i, 'text': line})
            
            else:
                structure['paragraphs'].append({'line': i, 'text': line})
        
        # Detect columns by analyzing text distribution
        structure['has_columns'] = self._detect_columns(text)
        
        return structure
    
    def _is_title(self, text: str) -> bool:
        """Determine if text is likely a title"""
        # Check for title characteristics
        words = text.split()
        is_short = len(words) <= 8
        is_caps = text.isupper() or text.istitle()
        
        # Common title indicators
        title_words = ['document', 'section', 'chapter', 'header', 'title', 'test', 'report']
        has_title_words = any(word.lower() in title_words for word in words)
        
        # No punctuation at end (except colon)
        ends_cleanly = not text.endswith('.') or text.endswith(':')
        
        return (is_short and is_caps) or has_title_words or (is_short and ends_cleanly)
    
    def _is_bullet_point(self, text: str) -> bool:
        """Determine if text is a bullet point"""
        bullet_patterns = [
            r'^[•·▪▫◦‣⁃]\s',  # Unicode bullets
            r'^[-*+]\s',        # ASCII bullets
            r'^\s*[•·▪▫◦‣⁃]\s', # Indented bullets
            r'^\s*[-*+]\s'      # Indented ASCII bullets
        ]
        return any(re.match(pattern, text) for pattern in bullet_patterns)
    
    def _is_numbered_item(self, text: str) -> bool:
        """Determine if text is a numbered list item"""
        numbered_patterns = [
            r'^\d+\.\s',        # 1. 2. 3.
            r'^[a-zA-Z]\.\s',   # a. b. c.
            r'^\([a-zA-Z0-9]+\)\s', # (1) (a) (i)
            r'^\d+\)\s'         # 1) 2) 3)
        ]
        return any(re.match(pattern, text) for pattern in numbered_patterns)
    
    def _is_table_row(self, text: str) -> bool:
        """Determine if text appears to be a table row"""
        # Look for patterns that suggest tabular data
        words = text.split()
        
        # Multiple short words/numbers
        if len(words) >= 3 and all(len(word) <= 10 for word in words):
            # Check for mix of text and numbers
            has_numbers = any(re.search(r'\d+', word) for word in words)
            has_text = any(re.search(r'[a-zA-Z]', word) for word in words)
            if has_numbers and has_text:
                return True
        
        # Common table headers
        table_headers = ['name', 'age', 'city', 'score', 'date', 'value', 'total', 'amount']
        if any(header in text.lower() for header in table_headers):
            return True
            
        return False
    
    def _detect_columns(self, text: str) -> bool:
        """Simple column detection based on text patterns"""
        # Look for words that commonly appear side by side in columns
        column_indicators = ['header', 'column', 'left', 'right', 'section']
        words = text.lower().split()
        
        # If we have column-related words, likely multi-column
        has_column_words = any(word in column_indicators for word in words)
        
        # Or if text has very long lines suggesting side-by-side content
        lines = text.split('\n') if '\n' in text else [text]
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
        
        return has_column_words or avg_line_length > 100
    
    def format_with_structure(self, text: str, structure: Dict[str, Any]) -> str:
        """Format text based on detected structure"""
        if not text.strip():
            return text
            
        # Split text into manageable parts
        lines = []
        current_text = text
        
        # Handle different scenarios based on structure
        if structure['has_columns']:
            formatted_text = self._format_columns(current_text, structure)
        else:
            formatted_text = self._format_single_column(current_text, structure)
        
        # Apply specific formatting for detected elements
        formatted_text = self._apply_element_formatting(formatted_text, structure)
        
        return formatted_text
    
    def _format_columns(self, text: str, structure: Dict[str, Any]) -> str:
        """Format multi-column text"""
        # Simple approach: try to separate left and right content
        words = text.split()
        
        # Find potential column break points
        potential_breaks = []
        for i, word in enumerate(words):
            if word.lower() in ['header', 'column', 'section'] and i < len(words) - 1:
                potential_breaks.append(i)
        
        if potential_breaks:
            # Split at the first potential break
            break_point = potential_breaks[0]
            left_column = ' '.join(words[:break_point + 2])  # Include header word
            right_column = ' '.join(words[break_point + 2:])
            
            return f"LEFT COLUMN:\n{left_column}\n\nRIGHT COLUMN:\n{right_column}"
        else:
            # Fallback: split roughly in middle
            mid_point = len(words) // 2
            left_column = ' '.join(words[:mid_point])
            right_column = ' '.join(words[mid_point:])
            
            return f"COLUMN 1:\n{left_column}\n\nCOLUMN 2:\n{right_column}"
    
    def _format_single_column(self, text: str, structure: Dict[str, Any]) -> str:
        """Format single column text with proper structure"""
        # Split into sentences/segments
        segments = re.split(r'(?<=[.!?])\s+', text)
        formatted_segments = []
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
                
            # Check if this segment matches any structural elements
            if self._is_title(segment):
                formatted_segments.append(f"\n=== {segment.upper()} ===\n")
            elif self._is_bullet_point(segment):
                formatted_segments.append(f"  • {segment.lstrip('•·▪▫◦‣⁃-*+ ')}")
            elif self._is_numbered_item(segment):
                formatted_segments.append(f"  {segment}")
            elif self._is_table_row(segment):
                formatted_segments.append(f"    {segment}")
            else:
                formatted_segments.append(segment)
        
        return '\n'.join(formatted_segments)
    
    def _apply_element_formatting(self, text: str, structure: Dict[str, Any]) -> str:
        """Apply specific formatting for detected elements"""
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue
            
            # Format tables
            if self._is_table_row(line):
                # Try to format as table
                words = line.split()
                if len(words) >= 3:
                    # Simple table formatting
                    formatted_line = ' | '.join(f"{word:12}" for word in words)
                    formatted_lines.append(formatted_line)
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def create_formatted_output(self, ocr_result: Dict[str, Any]) -> Dict[str, str]:
        """Create multiple formatted versions of the OCR result"""
        original_text = ocr_result.get('text', '')
        
        if not original_text.strip():
            return {
                'original': original_text,
                'structured': 'No text detected',
                'formatted': 'No text detected',
                'markdown': 'No text detected'
            }
        
        # Analyze structure
        structure = self.analyze_text_structure(original_text)
        
        # Create formatted version
        formatted_text = self.format_with_structure(original_text, structure)
        
        # Create structured version with metadata
        structured_text = self._create_structured_output(original_text, structure, ocr_result)
        
        # Create markdown version
        markdown_text = self._create_markdown_output(formatted_text, structure)
        
        return {
            'original': original_text,
            'structured': structured_text,
            'formatted': formatted_text,
            'markdown': markdown_text,
            'structure_analysis': structure
        }
    
    def _create_structured_output(self, text: str, structure: Dict[str, Any], ocr_result: Dict[str, Any]) -> str:
        """Create structured output with metadata"""
        metadata = []
        metadata.append(f"DOCUMENT ANALYSIS")
        metadata.append(f"=" * 40)
        metadata.append(f"Lines detected: {structure['total_lines']}")
        metadata.append(f"Titles found: {len(structure['titles'])}")
        metadata.append(f"Bullet points: {len(structure['bullet_points'])}")
        metadata.append(f"Numbered items: {len(structure['numbered_lists'])}")
        metadata.append(f"Table rows: {len(structure['tables'])}")
        metadata.append(f"Columns detected: {'Yes' if structure['has_columns'] else 'No'}")
        
        if 'confidence' in ocr_result:
            metadata.append(f"OCR Confidence: {ocr_result['confidence']:.2f}")
        if 'processing_time' in ocr_result:
            metadata.append(f"Processing time: {ocr_result['processing_time']:.2f}s")
        
        metadata.append("")
        metadata.append("FORMATTED TEXT:")
        metadata.append("-" * 40)
        
        formatted_text = self.format_with_structure(text, structure)
        
        return '\n'.join(metadata) + '\n' + formatted_text
    
    def _create_markdown_output(self, formatted_text: str, structure: Dict[str, Any]) -> str:
        """Convert formatted text to markdown"""
        lines = formatted_text.split('\n')
        markdown_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                markdown_lines.append('')
                continue
            
            # Convert formatting to markdown
            if line.startswith('===') and line.endswith('==='):
                # Title
                title = line.replace('===', '').strip()
                markdown_lines.append(f"# {title}")
            elif line.startswith('•'):
                # Bullet point
                markdown_lines.append(f"- {line[1:].strip()}")
            elif re.match(r'^\d+\.', line):
                # Numbered list
                markdown_lines.append(line)
            elif '|' in line and len(line.split('|')) > 2:
                # Table row
                markdown_lines.append(line)
            elif line.isupper() and len(line.split()) <= 5:
                # Potential header
                markdown_lines.append(f"## {line}")
            else:
                markdown_lines.append(line)
        
        return '\n'.join(markdown_lines)

def extract_text_with_advanced_formatting(image: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
    """Extract text with advanced layout preservation"""
    try:
        if not ENHANCED_OCR_AVAILABLE:
            return {
                'error': 'Enhanced OCR system not available',
                'original': '',
                'formatted': '',
                'structured': '',
                'markdown': ''
            }
        
        # Get OCR results
        start_time = time.time()
        ocr_result = extract_text_enhanced_multi_engine(image)
        
        # Apply advanced formatting
        layout_preserver = AdvancedLayoutPreserver()
        formatted_results = layout_preserver.create_formatted_output(ocr_result)
        
        # Combine results
        final_result = ocr_result.copy()
        final_result.update({
            'layout_preserved_text': formatted_results['formatted'],
            'structured_output': formatted_results['structured'],
            'markdown_output': formatted_results['markdown'],
            'structure_analysis': formatted_results['structure_analysis'],
            'formatting_applied': True,
            'total_processing_time': time.time() - start_time
        })
        
        return final_result
        
    except Exception as e:
        logger.error(f"Advanced formatting failed: {e}")
        return {
            'error': str(e),
            'original': '',
            'formatted': '',
            'structured': '',
            'markdown': '',
            'formatting_applied': False
        }

# Convenience functions
def get_formatted_text(image: Union[np.ndarray, Image.Image]) -> str:
    """Get formatted text preserving layout"""
    result = extract_text_with_advanced_formatting(image)
    return result.get('layout_preserved_text', result.get('text', ''))

def get_structured_text(image: Union[np.ndarray, Image.Image]) -> str:
    """Get structured text with analysis metadata"""
    result = extract_text_with_advanced_formatting(image)
    return result.get('structured_output', result.get('text', ''))

def get_markdown_text(image: Union[np.ndarray, Image.Image]) -> str:
    """Get text formatted as markdown"""
    result = extract_text_with_advanced_formatting(image)
    return result.get('markdown_output', result.get('text', ''))

# Testing
if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            print("🎯 Advanced Layout-Preserving OCR")
            print("=" * 60)
            
            # Load and process image
            image = cv2.imread(image_path)
            result = extract_text_with_advanced_formatting(image)
            
            # Display results
            print(f"Processing Time: {result.get('total_processing_time', 0):.2f}s")
            print(f"OCR Confidence: {result.get('confidence', 0):.2f}")
            print(f"Best Engine: {result.get('best_result', {}).get('engine', 'unknown')}")
            
            if 'structure_analysis' in result:
                structure = result['structure_analysis']
                print(f"\nStructure Analysis:")
                print(f"  Lines: {structure.get('total_lines', 0)}")
                print(f"  Titles: {len(structure.get('titles', []))}")
                print(f"  Bullet Points: {len(structure.get('bullet_points', []))}")
                print(f"  Tables: {len(structure.get('tables', []))}")
                print(f"  Columns: {'Yes' if structure.get('has_columns', False) else 'No'}")
            
            print("\n" + "="*60)
            print("ORIGINAL OCR:")
            print("="*60)
            print(result.get('text', ''))
            
            print("\n" + "="*60)
            print("FORMATTED OUTPUT:")
            print("="*60)
            print(result.get('layout_preserved_text', ''))
            
            print("\n" + "="*60)
            print("MARKDOWN FORMAT:")
            print("="*60)
            print(result.get('markdown_output', ''))
            
        else:
            print(f"Image not found: {image_path}")
    else:
        print("Usage: python advanced_layout_ocr.py <image_path>")
