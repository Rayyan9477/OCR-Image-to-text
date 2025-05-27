#!/usr/bin/env python
"""
Precision Layout OCR System
Ultra-accurate layout preservation with pixel-level positioning
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
import re
import math
import time
import os
import json
from dataclasses import dataclass
from collections import defaultdict

# Import enhanced OCR
try:
    from enhanced_multi_engine_ocr import extract_text_enhanced_multi_engine
    ENHANCED_OCR_AVAILABLE = True
except ImportError:
    ENHANCED_OCR_AVAILABLE = False
    logging.warning("Enhanced multi-engine OCR not available")

logger = logging.getLogger(__name__)

@dataclass
class TextElement:
    """Represents a text element with precise positioning"""
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    font_size: int = 12
    is_bold: bool = False
    is_italic: bool = False
    alignment: str = 'left'  # left, center, right
    
class PrecisionLayoutOCR:
    """Ultra-precise layout preservation system"""
    
    def __init__(self):
        self.min_line_height = 15
        self.max_line_gap = 30
        self.column_detection_threshold = 0.3
        self.indent_threshold = 20
        self.title_size_ratio = 1.5
        
    def extract_with_precision_layout(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Extract text with ultra-precise layout preservation
        
        Args:
            image: Input image
            
        Returns:
            Dict with layout-preserved results
        """
        try:
            # Convert image format if needed
            if isinstance(image, Image.Image):
                image_np = np.array(image)
                if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_np = image.copy()
            
            # Get enhanced OCR results
            if not ENHANCED_OCR_AVAILABLE:
                return {
                    'error': 'Enhanced OCR not available',
                    'layout_preserved_text': '',
                    'precision_formatted': ''
                }
            
            ocr_result = extract_text_enhanced_multi_engine(image)
            
            # Extract detailed positioning from OCR engines
            text_elements = self._extract_text_elements(image_np, ocr_result)
            
            # Analyze layout structure
            layout_analysis = self._analyze_layout_structure(text_elements, image_np.shape)
            
            # Generate precision-formatted output
            precision_formatted = self._generate_precision_format(text_elements, layout_analysis)
            
            # Generate HTML representation for visual formatting
            html_output = self._generate_html_layout(text_elements, layout_analysis, image_np.shape)
            
            # Generate markdown with precise spacing
            markdown_output = self._generate_markdown_layout(text_elements, layout_analysis)
            
            return {
                'status': 'success',
                'original_text': ocr_result.get('text', ''),
                'text_elements': [self._element_to_dict(elem) for elem in text_elements],
                'layout_analysis': layout_analysis,
                'precision_formatted': precision_formatted,
                'html_layout': html_output,
                'markdown_layout': markdown_output,
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0
            }
            
        except Exception as e:
            logger.error(f"Error in precision layout extraction: {e}")
            return {
                'error': str(e),
                'layout_preserved_text': '',
                'precision_formatted': ''
            }
    
    def _extract_text_elements(self, image: np.ndarray, ocr_result: Dict) -> List[TextElement]:
        """Extract individual text elements with precise positioning"""
        text_elements = []
        
        try:
            # Use PaddleOCR results if available (best positioning accuracy)
            if 'individual_results' in ocr_result:
                for result in ocr_result['individual_results']:
                    if result['engine'] == 'PaddleOCR' and result['status'] == 'success':
                        if 'detailed_results' in result:
                            for item in result['detailed_results']:
                                if isinstance(item, list) and len(item) >= 2:
                                    bbox = item[0]
                                    text_info = item[1]
                                    
                                    if isinstance(text_info, tuple) and len(text_info) >= 2:
                                        text = text_info[0]
                                        confidence = text_info[1]
                                    else:
                                        text = str(text_info)
                                        confidence = 0.9
                                    
                                    # Calculate bounding box
                                    x_coords = [point[0] for point in bbox]
                                    y_coords = [point[1] for point in bbox]
                                    
                                    x = int(min(x_coords))
                                    y = int(min(y_coords))
                                    width = int(max(x_coords) - min(x_coords))
                                    height = int(max(y_coords) - min(y_coords))
                                    
                                    # Estimate font size from height
                                    font_size = max(8, min(72, int(height * 0.8)))
                                    
                                    element = TextElement(
                                        text=text.strip(),
                                        x=x, y=y, width=width, height=height,
                                        confidence=confidence,
                                        font_size=font_size
                                    )
                                    
                                    if element.text:
                                        text_elements.append(element)
                        break
            
            # Fallback: create text elements from basic OCR results
            if not text_elements and ocr_result.get('text'):
                lines = ocr_result['text'].split('\n')
                y_pos = 50
                for line in lines:
                    if line.strip():
                        element = TextElement(
                            text=line.strip(),
                            x=50, y=y_pos, width=len(line.strip()) * 8, height=20,
                            confidence=0.8
                        )
                        text_elements.append(element)
                        y_pos += 25
            
            # Sort by position (top to bottom, left to right)
            text_elements.sort(key=lambda e: (e.y, e.x))
            
        except Exception as e:
            logger.error(f"Error extracting text elements: {e}")
        
        return text_elements
    
    def _analyze_layout_structure(self, text_elements: List[TextElement], image_shape: Tuple) -> Dict[str, Any]:
        """Analyze the layout structure of text elements"""
        if not text_elements:
            return {}
        
        height, width = image_shape[:2]
        
        # Group elements by lines
        lines = self._group_elements_by_lines(text_elements)
        
        # Detect columns
        columns = self._detect_columns(text_elements, width)
        
        # Detect document structure
        structure = self._detect_document_structure(text_elements, lines)
        
        # Calculate spacing patterns
        spacing = self._calculate_spacing_patterns(text_elements, lines)
        
        return {
            'image_dimensions': {'width': width, 'height': height},
            'total_elements': len(text_elements),
            'line_groups': len(lines),
            'columns': columns,
            'structure': structure,
            'spacing': spacing,
            'text_density': len(text_elements) / (width * height) * 10000  # elements per 10k pixels
        }
    
    def _group_elements_by_lines(self, text_elements: List[TextElement]) -> List[List[TextElement]]:
        """Group text elements that belong to the same line"""
        if not text_elements:
            return []
        
        lines = []
        current_line = [text_elements[0]]
        
        for i in range(1, len(text_elements)):
            element = text_elements[i]
            prev_element = text_elements[i-1]
            
            # Check if elements are on the same line
            y_diff = abs(element.y - prev_element.y)
            if y_diff <= self.min_line_height:
                current_line.append(element)
            else:
                # Sort current line by x position
                current_line.sort(key=lambda e: e.x)
                lines.append(current_line)
                current_line = [element]
        
        if current_line:
            current_line.sort(key=lambda e: e.x)
            lines.append(current_line)
        
        return lines
    
    def _detect_columns(self, text_elements: List[TextElement], image_width: int) -> Dict[str, Any]:
        """Detect column layout in the document"""
        if not text_elements:
            return {'count': 1, 'boundaries': []}
        
        # Analyze x-position distribution
        x_positions = [elem.x for elem in text_elements]
        x_positions.sort()
        
        # Find potential column boundaries
        boundaries = []
        column_starts = []
        
        # Group by similar x positions
        tolerance = 20
        current_group = [x_positions[0]]
        
        for x in x_positions[1:]:
            if x - current_group[-1] <= tolerance:
                current_group.append(x)
            else:
                # New column start detected
                avg_x = sum(current_group) / len(current_group)
                column_starts.append(int(avg_x))
                current_group = [x]
        
        if current_group:
            avg_x = sum(current_group) / len(current_group)
            column_starts.append(int(avg_x))
        
        # Determine column boundaries
        column_starts.sort()
        if len(column_starts) > 1:
            for i in range(len(column_starts) - 1):
                mid_point = (column_starts[i] + column_starts[i + 1]) // 2
                boundaries.append(mid_point)
        
        return {
            'count': len(column_starts),
            'starts': column_starts,
            'boundaries': boundaries
        }
    
    def _detect_document_structure(self, text_elements: List[TextElement], lines: List[List[TextElement]]) -> Dict[str, Any]:
        """Detect document structure elements (titles, paragraphs, lists, etc.)"""
        structure = {
            'titles': [],
            'headings': [],
            'paragraphs': [],
            'bullet_points': [],
            'numbered_lists': [],
            'tables': []
        }
        
        for line_idx, line_elements in enumerate(lines):
            if not line_elements:
                continue
            
            line_text = ' '.join([elem.text for elem in line_elements])
            
            # Detect titles (larger font, centered, or short lines)
            avg_font_size = sum([elem.font_size for elem in line_elements]) / len(line_elements)
            is_short = len(line_text) < 50
            is_large_font = avg_font_size > 16
            is_all_caps = line_text.isupper() and len(line_text) > 3
            
            if (is_short and is_large_font) or is_all_caps:
                structure['titles'].append({
                    'line': line_idx,
                    'text': line_text,
                    'position': (line_elements[0].x, line_elements[0].y),
                    'font_size': avg_font_size
                })
            
            # Detect bullet points
            bullet_patterns = [r'^\s*[•·▪▫◦‣⁃]\s+', r'^\s*[-*+]\s+', r'^\s*[○●]\s+']
            for pattern in bullet_patterns:
                if re.match(pattern, line_text):
                    structure['bullet_points'].append({
                        'line': line_idx,
                        'text': line_text,
                        'position': (line_elements[0].x, line_elements[0].y)
                    })
                    break
            
            # Detect numbered lists
            if re.match(r'^\s*\d+[\.\)]\s+', line_text):
                structure['numbered_lists'].append({
                    'line': line_idx,
                    'text': line_text,
                    'position': (line_elements[0].x, line_elements[0].y)
                })
        
        return structure
    
    def _calculate_spacing_patterns(self, text_elements: List[TextElement], lines: List[List[TextElement]]) -> Dict[str, Any]:
        """Calculate spacing patterns for accurate reproduction"""
        if not lines:
            return {}
        
        line_heights = []
        line_gaps = []
        indentations = []
        
        # Calculate line heights and gaps
        for i, line_elements in enumerate(lines):
            if line_elements:
                max_height = max([elem.height for elem in line_elements])
                line_heights.append(max_height)
                
                # Calculate indentation
                min_x = min([elem.x for elem in line_elements])
                indentations.append(min_x)
                
                # Calculate gap to next line
                if i < len(lines) - 1 and lines[i + 1]:
                    current_bottom = line_elements[0].y + max_height
                    next_top = lines[i + 1][0].y
                    gap = next_top - current_bottom
                    line_gaps.append(max(0, gap))
        
        return {
            'avg_line_height': sum(line_heights) / len(line_heights) if line_heights else 20,
            'avg_line_gap': sum(line_gaps) / len(line_gaps) if line_gaps else 5,
            'common_indentations': self._find_common_values(indentations),
            'line_heights': line_heights,
            'line_gaps': line_gaps,
            'indentations': indentations
        }
    
    def _find_common_values(self, values: List[int], tolerance: int = 10) -> List[int]:
        """Find commonly occurring values within tolerance"""
        if not values:
            return []
        
        groups = defaultdict(list)
        for value in values:
            # Find the group this value belongs to
            group_key = None
            for key in groups:
                if abs(value - key) <= tolerance:
                    group_key = key
                    break
            
            if group_key is None:
                group_key = value
            
            groups[group_key].append(value)
        
        # Return groups with more than one occurrence
        common = []
        for key, group in groups.items():
            if len(group) >= 2:
                avg_value = sum(group) // len(group)
                common.append(avg_value)
        
        return sorted(common)
    
    def _generate_precision_format(self, text_elements: List[TextElement], layout_analysis: Dict) -> str:
        """Generate precision-formatted text that preserves exact layout"""
        if not text_elements:
            return ""
        
        lines = self._group_elements_by_lines(text_elements)
        result = []
        
        prev_y = 0
        base_x = min([elem.x for elem in text_elements]) if text_elements else 0
        
        for line_elements in lines:
            if not line_elements:
                continue
            
            # Calculate vertical spacing
            current_y = line_elements[0].y
            if prev_y > 0:
                line_gap = current_y - prev_y
                spacing = layout_analysis.get('spacing', {})
                avg_gap = spacing.get('avg_line_gap', 5)
                
                # Add extra newlines for larger gaps
                if line_gap > avg_gap * 2:
                    result.append("")  # Extra blank line
            
            # Build the line with horizontal spacing
            line_text = ""
            prev_x = base_x
            
            for elem in line_elements:
                # Calculate horizontal spacing
                x_gap = elem.x - prev_x
                if x_gap > 20:  # Significant gap
                    spaces_needed = max(1, x_gap // 8)  # Approximate character width
                    line_text += " " * spaces_needed
                elif line_text and not line_text.endswith(" "):
                    line_text += " "
                
                line_text += elem.text
                prev_x = elem.x + elem.width
            
            result.append(line_text.rstrip())
            prev_y = current_y + max([elem.height for elem in line_elements])
        
        return "\n".join(result)
    
    def _generate_html_layout(self, text_elements: List[TextElement], layout_analysis: Dict, image_shape: Tuple) -> str:
        """Generate HTML representation with absolute positioning"""
        height, width = image_shape[:2]
        
        html = f"""
        <div style="position: relative; width: {width}px; height: {height}px; border: 1px solid #ccc; margin: 20px 0;">
        """
        
        for elem in text_elements:
            style = f"""
                position: absolute;
                left: {elem.x}px;
                top: {elem.y}px;
                font-size: {elem.font_size}px;
                font-weight: {'bold' if elem.is_bold else 'normal'};
                font-style: {'italic' if elem.is_italic else 'normal'};
                white-space: nowrap;
                color: #333;
            """
            
            html += f'<span style="{style}">{elem.text}</span>\n'
        
        html += "</div>"
        return html
    
    def _generate_markdown_layout(self, text_elements: List[TextElement], layout_analysis: Dict) -> str:
        """Generate markdown with preserved structure"""
        if not text_elements:
            return ""
        
        lines = self._group_elements_by_lines(text_elements)
        result = []
        
        structure = layout_analysis.get('structure', {})
        title_lines = {item['line'] for item in structure.get('titles', [])}
        bullet_lines = {item['line'] for item in structure.get('bullet_points', [])}
        numbered_lines = {item['line'] for item in structure.get('numbered_lists', [])}
        
        for i, line_elements in enumerate(lines):
            if not line_elements:
                continue
            
            line_text = ' '.join([elem.text for elem in line_elements])
            
            # Apply markdown formatting based on structure
            if i in title_lines:
                # Determine heading level based on font size and position
                avg_font_size = sum([elem.font_size for elem in line_elements]) / len(line_elements)
                if avg_font_size > 20:
                    result.append(f"# {line_text}")
                elif avg_font_size > 16:
                    result.append(f"## {line_text}")
                else:
                    result.append(f"### {line_text}")
                result.append("")  # Blank line after heading
            elif i in bullet_lines:
                # Clean up bullet point
                clean_text = re.sub(r'^\s*[•·▪▫◦‣⁃\-*+○●]\s*', '- ', line_text)
                result.append(clean_text)
            elif i in numbered_lines:
                # Keep numbered list as is
                result.append(line_text)
            else:
                result.append(line_text)
        
        return "\n".join(result)
    
    def _element_to_dict(self, element: TextElement) -> Dict[str, Any]:
        """Convert TextElement to dictionary for JSON serialization"""
        return {
            'text': element.text,
            'x': element.x,
            'y': element.y,
            'width': element.width,
            'height': element.height,
            'confidence': element.confidence,
            'font_size': element.font_size,
            'is_bold': element.is_bold,
            'is_italic': element.is_italic,
            'alignment': element.alignment
        }

def extract_text_with_precision_layout(image: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
    """
    Main function to extract text with precision layout preservation
    
    Args:
        image: Input image
        
    Returns:
        Dict with precision layout results
    """
    precision_ocr = PrecisionLayoutOCR()
    return precision_ocr.extract_with_precision_layout(image)

# Example usage
if __name__ == "__main__":
    import time
    
    # Test with sample image
    test_image_path = "sample_document.jpg"
    if os.path.exists(test_image_path):
        image = cv2.imread(test_image_path)
        
        start_time = time.time()
        result = extract_text_with_precision_layout(image)
        end_time = time.time()
        
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        print(f"Status: {result.get('status', 'unknown')}")
        print("\n--- Precision Formatted Text ---")
        print(result.get('precision_formatted', ''))
        print("\n--- Layout Analysis ---")
        print(json.dumps(result.get('layout_analysis', {}), indent=2))
    else:
        print(f"Test image not found: {test_image_path}")
