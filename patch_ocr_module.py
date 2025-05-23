#!/usr/bin/env python3
"""
OCR Module Patch - Fixes issues with OCR engine initialization and improves error handling
"""

import importlib
import os
import sys
from pathlib import Path

def get_module_path():
    """Get the absolute path to the OCR module"""
    ocr_module_path = os.path.join(os.getcwd(), "ocr_module.py")
    if os.path.exists(ocr_module_path):
        return ocr_module_path
    else:
        # Try src directory
        src_path = os.path.join(os.getcwd(), "src", "ocr_module.py") 
        if os.path.exists(src_path):
            return src_path
    return None

def fix_paddle_ocr_function():
    """Fix the paddle_ocr function to handle initialization errors better"""
    ocr_module_path = get_module_path()
    
    if not ocr_module_path:
        print("Error: Could not locate ocr_module.py")
        return False
    
    print(f"Patching OCR module at: {ocr_module_path}")
    
    with open(ocr_module_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the paddle_ocr function
    paddle_ocr_def = "def paddle_ocr(image, preserve_layout=True):"
    paddle_ocr_pos = content.find(paddle_ocr_def)
    
    if paddle_ocr_pos == -1:
        print("Error: Could not find paddle_ocr function")
        return False
    
    # Find the next function definition to isolate the paddle_ocr function
    next_func_pos = content.find("def ", paddle_ocr_pos + 1)
    if next_func_pos == -1:
        print("Error: Could not find the end of paddle_ocr function")
        return False
    
    paddle_ocr_function = content[paddle_ocr_pos:next_func_pos]
    
    # Create the new function with better error handling
    new_paddle_ocr_function = """def paddle_ocr(image, preserve_layout=True):
    \"\"\"Perform OCR using PaddleOCR\"\"\"
    start_time = time.time()
    
    try:
        # Check if PaddleOCR is available and import directly
        try:
            from paddleocr import PaddleOCR
            has_paddle = True
        except ImportError:
            print("PaddleOCR not available")
            has_paddle = False
            return ""
        
        paddle = get_paddle_ocr()
        if paddle is None:
            # Try to initialize directly as a fallback
            try:
                print("Initializing PaddleOCR directly...")
                paddle = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            except Exception as e:
                print(f"Error initializing PaddleOCR directly: {e}")
                return ""
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            # Convert to RGB if it's not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_np = np.array(image)
        else:
            img_np = image
"""

    # Replace the function
    updated_content = content.replace(paddle_ocr_function, new_paddle_ocr_function)
    
    # Save the updated file
    with open(ocr_module_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("Successfully patched paddle_ocr function")
    return True

def fix_easyocr_function():
    """Fix the easyocr_ocr function to handle initialization errors better"""
    ocr_module_path = get_module_path()
    
    if not ocr_module_path:
        print("Error: Could not locate ocr_module.py")
        return False
    
    with open(ocr_module_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the easyocr_ocr function
    easyocr_func_def = "def easyocr_ocr(image, preserve_layout=True):"
    easyocr_pos = content.find(easyocr_func_def)
    
    if easyocr_pos == -1:
        print("Error: Could not find easyocr_ocr function")
        return False
    
    # Find the next function definition to isolate the easyocr_ocr function
    next_func_pos = content.find("def ", easyocr_pos + 1)
    if next_func_pos == -1:
        print("Error: Could not find the end of easyocr_ocr function")
        return False
    
    easyocr_function = content[easyocr_pos:next_func_pos]
    
    # Create the new function with better error handling
    new_easyocr_function = """def easyocr_ocr(image, preserve_layout=True):
    \"\"\"Perform OCR using EasyOCR\"\"\"
    start_time = time.time()
    
    try:
        # Check if EasyOCR is available and import directly
        try:
            import easyocr
            has_easyocr = True
        except ImportError:
            print("EasyOCR not available")
            has_easyocr = False
            return ""
        
        reader = get_easy_ocr()
        if reader is None:
            # Try to initialize directly as a fallback
            try:
                print("Initializing EasyOCR directly...")
                reader = easyocr.Reader(['en'], gpu=False)
            except Exception as e:
                print(f"Error initializing EasyOCR directly: {e}")
                return ""
        
        # Convert to numpy array if it's a PIL Image
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
"""

    # Replace the function
    updated_content = content.replace(easyocr_function, new_easyocr_function)
    
    # Save the updated file
    with open(ocr_module_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("Successfully patched easyocr_ocr function")
    return True

def backup_file(file_path):
    """Create a backup of the file"""
    backup_path = file_path + ".bak"
    import shutil
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at: {backup_path}")

def main():
    """Main entry point"""
    print("OCR Module Patcher")
    print("------------------")
    
    ocr_module_path = get_module_path()
    if not ocr_module_path:
        print("Error: Could not locate ocr_module.py")
        return 1
    
    print(f"Found OCR module at: {ocr_module_path}")
    
    # Backup the original file
    backup_file(ocr_module_path)
    
    # Fix the paddle_ocr function
    if fix_paddle_ocr_function():
        print("✓ Fixed paddle_ocr function")
    else:
        print("✗ Failed to fix paddle_ocr function")
    
    # Fix the easyocr_ocr function
    if fix_easyocr_function():
        print("✓ Fixed easyocr_ocr function")
    else:
        print("✗ Failed to fix easyocr_ocr function")
    
    print("Patch completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
