#!/usr/bin/env python
"""
Layout-Preserving OCR System
Maintains original document formatting and spatial relationships
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class TextRegion:
    """Represents a text region with position and content"""
    text: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    line_number: int
    column_number: int
    font_size: int = 12
    is_title: bool = False
    is_bullet: bool = False
    indentation: int = 0

class LayoutPreservingOCR:
    """OCR system that preserves original document layout and formatting"""
    
    def __init__(self):
        self.text_regions = []
        self.page_width = 0
        self.page_height = 0
        
    def detect_text_regions(self, image: np.ndarray, ocr_results: List) -> List[TextRegion]:
        """Detect and analyze text regions with spatial information"""
        regions = []
        
        # Sort results by vertical position (top to bottom)
        sorted_results = sorted(ocr_results, key=lambda x: x[0][0][1])  # Sort by top-left y coordinate
        
        current_line = 0
        last_y = -1
        line_threshold = 20  # Pixels to consider same line
        
        for i, result in enumerate(sorted_results):
            bbox, text, confidence = result
            
            # Extract coordinates
            points = np.array(bbox, dtype=np.int32)
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            
            # Determine line number
            if last_y == -1 or abs(y1 - last_y) > line_threshold:
                current_line += 1
            last_y = y1
            
            # Analyze text characteristics
            is_title = self._is_title_text(text, y2 - y1)
            is_bullet = self._is_bullet_point(text)
            indentation = self._calculate_indentation(x1, image.shape[1])
            font_size = self._estimate_font_size(y2 - y1)
            
            # Determine column (simple left-to-right ordering)
            column = self._determine_column(x1, image.shape[1])
            
            region = TextRegion(
                text=text.strip(),
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                line_number=current_line,
                column_number=column,
                font_size=font_size,
                is_title=is_title,
                is_bullet=is_bullet,
                indentation=indentation
            )
            
            regions.append(region)
        
        return regions
    
    def _is_title_text(self, text: str, height: int) -> bool:
        """Determine if text is likely a title"""
        # Check for title characteristics
        is_short = len(text.split()) <= 6
        is_caps = text.isupper()
        is_large = height > 25
        is_centered_style = any(word in text.upper() for word in ['TITLE', 'HEADER', 'SECTION'])
        
        return (is_large and is_short) or is_caps or is_centered_style
    
    def _is_bullet_point(self, text: str) -> bool:
        """Determine if text is a bullet point"""
        bullet_patterns = [r'^[•·▪▫◦‣⁃]\s', r'^\d+\.\s', r'^[a-zA-Z]\.\s', r'^[-*+]\s']
        return any(re.match(pattern, text) for pattern in bullet_patterns)
    
    def _calculate_indentation(self, x_pos: int, page_width: int) -> int:
        """Calculate indentation level based on x position"""
        # Normalize to percentage of page width
        relative_pos = x_pos / page_width
        
        if relative_pos < 0.1:
            return 0  # Left margin
        elif relative_pos < 0.2:
            return 1  # First level indent
        elif relative_pos < 0.3:
            return 2  # Second level indent
        else:
            return 3  # Deep indent
    
    def _estimate_font_size(self, height: int) -> int:
        """Estimate font size based on text height"""
        if height > 30:
            return 18  # Large title
        elif height > 20:
            return 14  # Subtitle
        else:
            return 12  # Body text
    
    def _determine_column(self, x_pos: int, page_width: int) -> int:
        """Determine column number for multi-column layouts"""
        relative_pos = x_pos / page_width
        
        if relative_pos < 0.5:
            return 1  # Left column
        else:
            return 2  # Right column
    
    def format_preserved_text(self, regions: List[TextRegion]) -> str:
        """Format text while preserving original layout"""
        if not regions:
            return ""
        
        # Group regions by line
        lines = {}
        for region in regions:
            line_num = region.line_number
            if line_num not in lines:
                lines[line_num] = []
            lines[line_num].append(region)
        
        # Sort each line by column/x position
        for line_num in lines:
            lines[line_num].sort(key=lambda r: (r.column_number, r.bbox[0]))
        
        formatted_text = []
        last_indentation = 0
        
        for line_num in sorted(lines.keys()):
            line_regions = lines[line_num]
            line_text = ""
            
            for i, region in enumerate(line_regions):
                # Add indentation
                if i == 0:  # Only for first region in line
                    if region.is_bullet:
                        line_text += "  " * region.indentation + region.text
                    elif region.indentation > last_indentation:
                        line_text += "    " * region.indentation + region.text
                    else:
                        line_text += region.text
                    last_indentation = region.indentation
                else:
                    # Add spacing between columns
                    line_text += "    " + region.text
            
            # Add title formatting
            if line_regions and line_regions[0].is_title:
                if line_regions[0].font_size > 16:
                    line_text = line_text.upper()
                    formatted_text.append("")  # Add space before title
                    formatted_text.append(line_text)
                    formatted_text.append("=" * len(line_text))  # Underline
                else:
                    formatted_text.append("")
                    formatted_text.append(line_text)
                    formatted_text.append("-" * len(line_text))
            else:
                formatted_text.append(line_text)
        
        return "\n".join(formatted_text)
    
    def create_visual_layout_map(self, image: np.ndarray, regions: List[TextRegion]) -> np.ndarray:
        """Create a visual representation of the detected layout"""
        # Create a copy of the original image
        layout_image = image.copy()
        
        # Colors for different types of text
        colors = {
            'title': (255, 0, 0),     # Red for titles
            'bullet': (0, 255, 0),   # Green for bullets
            'body': (0, 0, 255),     # Blue for body text
            'column2': (255, 165, 0) # Orange for second column
        }
        
        for region in regions:
            x1, y1, x2, y2 = region.bbox
            
            # Choose color based on text type
            if region.is_title:
                color = colors['title']
            elif region.is_bullet:
                color = colors['bullet']
            elif region.column_number == 2:
                color = colors['column2']
            else:
                color = colors['body']
            
            # Draw bounding box
            cv2.rectangle(layout_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"L{region.line_number}C{region.column_number}"
            cv2.putText(layout_image, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return layout_image
    
    def extract_with_layout_preservation(self, image: np.ndarray, ocr_engine_results: Dict) -> Dict[str, Any]:
        """Extract text while preserving layout from multiple OCR engines"""
        self.page_height, self.page_width = image.shape[:2]
        
        # Use the best OCR result for layout analysis
        best_engine = None
        best_results = None
        highest_confidence = 0
        
        for engine_name, result in ocr_engine_results.items():
            if result['status'] == 'success' and result['confidence'] > highest_confidence:
                highest_confidence = result['confidence']
                best_engine = engine_name
                
                # Convert result to standard format for layout analysis
                if engine_name == 'easyocr':
                    # EasyOCR format: [(bbox, text, confidence), ...]
                    best_results = result.get('raw_results', [])
                elif engine_name == 'paddleocr':
                    # PaddleOCR format: [[[bbox], [text, confidence]], ...]
                    best_results = []
                    for item in result.get('raw_results', []):
                        if item:
                            bbox, (text, conf) = item
                            best_results.append((bbox, text, conf))
        
        if not best_results:
            return {
                'formatted_text': 'No text detected',
                'layout_preserved_text': 'No text detected',
                'text_regions': [],
                'layout_analysis': {},
                'visual_layout': None
            }
        
        # Detect text regions with layout information
        text_regions = self.detect_text_regions(image, best_results)
        
        # Create formatted text preserving layout
        layout_preserved_text = self.format_preserved_text(text_regions)
        
        # Create visual layout map
        visual_layout = self.create_visual_layout_map(image, text_regions)
        
        # Analyze layout statistics
        layout_analysis = {
            'total_lines': max([r.line_number for r in text_regions]) if text_regions else 0,
            'columns_detected': max([r.column_number for r in text_regions]) if text_regions else 0,
            'titles_detected': sum(1 for r in text_regions if r.is_title),
            'bullet_points': sum(1 for r in text_regions if r.is_bullet),
            'indentation_levels': max([r.indentation for r in text_regions]) if text_regions else 0,
            'average_confidence': np.mean([r.confidence for r in text_regions]) if text_regions else 0,
            'best_engine_used': best_engine
        }
        
        return {
            'formatted_text': layout_preserved_text,
            'layout_preserved_text': layout_preserved_text,
            'text_regions': text_regions,
            'layout_analysis': layout_analysis,
            'visual_layout': visual_layout,
            'original_text': ocr_engine_results.get(best_engine, {}).get('text', ''),
            'best_engine': best_engine
        }


def integrate_with_enhanced_ocr(image, enhanced_ocr_result):
    """Integrate layout preservation with enhanced multi-engine OCR"""
    try:
        # Initialize layout preserving OCR
        layout_ocr = LayoutPreservingOCR()
        
        # Extract layout-preserved result
        layout_result = layout_ocr.extract_with_layout_preservation(
            image, enhanced_ocr_result.get('results', {})
        )
        
        # Combine with original result
        enhanced_result = enhanced_ocr_result.copy()
        enhanced_result.update({
            'layout_preserved_text': layout_result['layout_preserved_text'],
            'layout_analysis': layout_result['layout_analysis'],
            'text_regions': layout_result['text_regions'],
            'visual_layout': layout_result['visual_layout'],
            'formatting_applied': True
        })
        
        return enhanced_result
        
    except Exception as e:
        logger.error(f"Layout preservation failed: {e}")
        # Return original result with error info
        enhanced_ocr_result['layout_preserved_text'] = enhanced_ocr_result.get('text', '')
        enhanced_ocr_result['formatting_applied'] = False
        enhanced_ocr_result['layout_error'] = str(e)
        return enhanced_ocr_result


# Example usage and testing
if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            # Load image
            image = cv2.imread(image_path)
            
            # Simulate OCR results for testing
            mock_ocr_results = {
                'easyocr': {
                    'status': 'success',
                    'confidence': 0.85,
                    'text': 'Sample text for testing layout preservation',
                    'raw_results': [
                        ([[[100, 50], [300, 50], [300, 80], [100, 80]]], 'DOCUMENT TITLE', 0.95),
                        ([[[100, 100], [400, 100], [400, 130], [100, 130]]], 'This is the first paragraph of text.', 0.88),
                        ([[[120, 150], [420, 150], [420, 180], [120, 180]]], '• First bullet point', 0.90),
                        ([[[120, 190], [420, 190], [420, 220], [120, 220]]], '• Second bullet point', 0.87)
                    ]
                }
            }
            
            # Test layout preservation
            layout_ocr = LayoutPreservingOCR()
            result = layout_ocr.extract_with_layout_preservation(image, mock_ocr_results)
            
            print("Layout-Preserved Text:")
            print("=" * 40)
            print(result['layout_preserved_text'])
            print("\nLayout Analysis:")
            print(result['layout_analysis'])
            
        else:
            print(f"Image file not found: {image_path}")
    else:
        print("Usage: python layout_preserving_ocr.py <image_path>")
