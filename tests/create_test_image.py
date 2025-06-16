from PIL import Image, ImageDraw, ImageFont
import os

def create_test_image(text, output_path, size=(800, 400), bg_color='white', text_color='black'):
    """Create a test image with specified text"""
    # Create a new image with white background
    image = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except IOError:
        font = ImageFont.load_default()
    
    # Calculate text position to center it
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw the text
    draw.text((x, y), text, font=font, fill=text_color)
    
    # Save the image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    return output_path

def create_test_images():
    """Create various test images for OCR testing"""
    test_dir = "tests/test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create simple text image
    create_test_image(
        "Hello, World!\nThis is a test image.",
        os.path.join(test_dir, "simple_text.png")
    )
    
    # Create image with multiple lines
    create_test_image(
        "Line 1\nLine 2\nLine 3\nLine 4",
        os.path.join(test_dir, "multiple_lines.png")
    )
    
    # Create image with special characters
    create_test_image(
        "Special chars: !@#$%^&*()\nNumbers: 1234567890",
        os.path.join(test_dir, "special_chars.png")
    )
    
    # Create image with different font sizes
    create_test_image(
        "Large Text\nSmall Text",
        os.path.join(test_dir, "different_sizes.png"),
        size=(800, 600)
    )

if __name__ == "__main__":
    create_test_images() 