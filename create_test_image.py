#!/usr/bin/env python3
"""
Test Image Creator for OCR Testing

This script creates a simple test image with various text elements
to verify OCR functionality across different engines.
"""

import os
import sys
import platform
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_test_image(output_path="simple_test_image.jpg"):
    """Create a simple test image with text for OCR testing"""
    try:
        width, height = 800, 400
        background_color = (255, 255, 255)
        text_color = (0, 0, 0)
        
        # Create a blank image
        image = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(image)
        
        # Try to get a font
        try:
            # Try to use a common font
            if platform.system() == "Windows":
                font_path = "C:\\Windows\\Fonts\\Arial.ttf"
            elif platform.system() == "Darwin":  # macOS
                font_path = "/System/Library/Fonts/Helvetica.ttc"
            else:  # Linux and others
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 32)
                small_font = ImageFont.truetype(font_path, 20)
            else:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Add text to the image
        text1 = "OCR Test Image"
        text2 = "This is a sample image for testing OCR functionality."
        text3 = "It contains different text sizes and formatting."
        text4 = "1234567890 !@#$%^&*()"
        
        # Header text
        draw.text((width/2-150, 50), text1, fill=text_color, font=font)
        
        # Main paragraphs
        draw.text((50, 150), text2, fill=text_color, font=font)
        draw.text((50, 200), text3, fill=text_color, font=font)
        draw.text((50, 250), text4, fill=text_color, font=font)
        
        # Draw some lines and a rectangle
        draw.line([(50, 300), (750, 300)], fill=(0, 0, 0), width=3)
        draw.rectangle([(50, 320), (750, 380)], outline=(0, 0, 0), width=2)
        draw.text((400-100, 340), "Text in a box", fill=text_color, font=font)
        
        # Draw different font sizes for testing
        draw.text((600, 150), "Small", fill=text_color, font=small_font)
        
        # Save the image
        image.save(output_path, quality=95)
        print(f"Created test image: {output_path}")
        return True
    except Exception as e:
        print(f"Error creating test image: {str(e)}")
        return False

def create_complex_test_image(output_path="complex_test_image.jpg"):
    """Create a more complex test image with tables and multiple columns"""
    try:
        width, height = 1000, 800
        background_color = (255, 255, 255)
        text_color = (0, 0, 0)
        
        # Create a blank image
        image = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(image)
        
        # Try to get a font
        try:
            if platform.system() == "Windows":
                font_path = "C:\\Windows\\Fonts\\Arial.ttf"
            elif platform.system() == "Darwin":  # macOS
                font_path = "/System/Library/Fonts/Helvetica.ttc"
            else:  # Linux and others
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                
            if os.path.exists(font_path):
                title_font = ImageFont.truetype(font_path, 36)
                normal_font = ImageFont.truetype(font_path, 24)
                small_font = ImageFont.truetype(font_path, 18)
            else:
                title_font = ImageFont.load_default()
                normal_font = ImageFont.load_default()
                small_font = ImageFont.load_default()
        except:
            title_font = ImageFont.load_default()
            normal_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            
        # Add title
        draw.text((width/2-150, 30), "Complex OCR Test Document", fill=text_color, font=title_font)
        
        # Add document section
        draw.text((50, 100), "Document Section", fill=text_color, font=normal_font)
        draw.line([(50, 130), (950, 130)], fill=text_color, width=2)
        
        # Left column text
        column_text = "This left column contains a paragraph of text that should be recognized as a single block. " + \
                      "OCR systems should maintain the layout and detect that this text is separate from the right column."
        
        # Wrap text to fit column width (simple implementation)
        words = column_text.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) * 10 < 400:  # rough estimate of text width
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        # Draw left column
        for i, line in enumerate(lines):
            draw.text((50, 150 + i*30), line, fill=text_color, font=normal_font)
        
        # Draw divider between columns
        draw.line([(500, 150), (500, 300)], fill=text_color, width=1)
        
        # Right column
        draw.text((550, 150), "Right Column Header", fill=text_color, font=normal_font)
        draw.text((550, 190), "• First bullet point", fill=text_color, font=small_font)
        draw.text((550, 220), "• Second bullet point", fill=text_color, font=small_font)
        draw.text((550, 250), "• Third with numbers 123", fill=text_color, font=small_font)
        
        # Draw table
        table_top = 350
        table_left = 50
        cell_width = 200
        cell_height = 50
        
        # Table header
        draw.text((width/2-100, table_top-40), "Sample Table", fill=text_color, font=normal_font)
        
        # Draw table grid
        for i in range(5):  # 5 columns
            # Draw vertical line
            x = table_left + i * cell_width
            draw.line([(x, table_top), (x, table_top + 4 * cell_height)], fill=text_color, width=2)
            
        # Last vertical line
        draw.line([(table_left + 4 * cell_width, table_top), 
                  (table_left + 4 * cell_width, table_top + 4 * cell_height)], 
                  fill=text_color, width=2)
        
        for i in range(5):  # 5 rows
            # Draw horizontal line
            y = table_top + i * cell_height
            draw.line([(table_left, y), (table_left + 4 * cell_width, y)], fill=text_color, width=2)
        
        # Table headers
        headers = ["Name", "Age", "City", "Score"]
        for i, header in enumerate(headers):
            draw.text((table_left + i * cell_width + 10, table_top + 10), 
                    header, fill=text_color, font=normal_font)
        
        # Table data
        data = [
            ["John Smith", "28", "New York", "95"],
            ["Alice Johnson", "34", "London", "87"],
            ["Robert Lee", "45", "Tokyo", "78"]
        ]
        
        for row_idx, row in enumerate(data):
            for col_idx, cell in enumerate(row):
                draw.text((table_left + col_idx * cell_width + 10, 
                         table_top + (row_idx+1) * cell_height + 10), 
                        cell, fill=text_color, font=normal_font)
        
        # Draw a barcode-like element
        barcode_left = 600
        barcode_top = 600
        barcode_height = 100
        barcode_width = 300
        
        # Draw barcode text
        draw.text((barcode_left, barcode_top - 30), "Sample Barcode", fill=text_color, font=small_font)
        
        # Draw barcode lines with varying widths
        x = barcode_left
        while x < barcode_left + barcode_width:
            line_width = np.random.randint(1, 6)
            if np.random.random() < 0.5:  # 50% chance for a black line
                draw.rectangle([(x, barcode_top), (x + line_width, barcode_top + barcode_height)], 
                            fill=text_color)
            x += line_width + np.random.randint(1, 5)
        
        # Save the image
        image.save(output_path, quality=95)
        print(f"Created complex test image: {output_path}")
        return True
    except Exception as e:
        print(f"Error creating complex test image: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Creating test images for OCR testing...")
    create_test_image()
    create_complex_test_image()
    print("Done!")
