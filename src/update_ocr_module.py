import re
import sys

def update_ocr_module():
    try:
        # Path to the OCR module
        ocr_module_path = r"c:\Users\rayyan.a\Downloads\Repo\OCR-Image-to-text\ocr_module.py"
        
        # Read the original file content
        with open(ocr_module_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Pattern to identify the perform_ocr function's OCR engine selection part
        pattern = r"""(\s+# Apply OCR based on selected engine
\s+if engine == "paddle":
\s+result = paddle_ocr\(processed_image, preserve_layout\)
\s+if not result:
\s+# Fallback to EasyOCR if PaddleOCR fails
\s+print\("PaddleOCR failed, falling back to EasyOCR..."\)
\s+result = easyocr_ocr\(processed_image, preserve_layout\)
\s+return result or "OCR processing failed - no text detected"
\s+elif engine == "easy":
\s+result = easyocr_ocr\(processed_image, preserve_layout\)
\s+if not result:
\s+# Fallback to PaddleOCR if EasyOCR fails
\s+print\("EasyOCR failed, falling back to PaddleOCR..."\)
\s+result = paddle_ocr\(processed_image, preserve_layout\)
\s+return result or "OCR processing failed - no text detected"
\s+else:  # Combined approach with parallel processing
\s+result = combined_ocr\(processed_image, preserve_layout\)
\s+return result or "OCR processing failed - no text detected")"""
        
        # Updated implementation with pytesseract fallback
        replacement = """        # Import the tesseract fallback
        try:
            import pytesseract
            has_tesseract_fallback = True
        except ImportError:
            has_tesseract_fallback = False

        # Apply OCR based on selected engine
        if engine == "paddle":
            result = paddle_ocr(processed_image, preserve_layout)
            if not result:
                # Fallback to EasyOCR if PaddleOCR fails
                print("PaddleOCR failed, falling back to EasyOCR...")
                result = easyocr_ocr(processed_image, preserve_layout)
                
                # If EasyOCR also fails, try pytesseract
                if not result and has_tesseract_fallback:
                    print("EasyOCR failed, falling back to pytesseract OCR...")
                    result = pytesseract_ocr(processed_image, preserve_layout)
            return result or "OCR processing failed - no text detected"
        elif engine == "easy":
            result = easyocr_ocr(processed_image, preserve_layout)
            if not result:
                # Fallback to PaddleOCR if EasyOCR fails
                print("EasyOCR failed, falling back to PaddleOCR...")
                result = paddle_ocr(processed_image, preserve_layout)
                
                # If PaddleOCR also fails, try pytesseract
                if not result and has_tesseract_fallback:
                    print("PaddleOCR failed, falling back to pytesseract OCR...")
                    result = pytesseract_ocr(processed_image, preserve_layout)
            return result or "OCR processing failed - no text detected"
        else:  # Combined approach with parallel processing
            result = combined_ocr(processed_image, preserve_layout)
            
            # If combined approach fails, try pytesseract as last resort
            if not result and has_tesseract_fallback:
                print("All OCR engines failed, falling back to pytesseract OCR...")
                result = pytesseract_ocr(processed_image, preserve_layout)
            
            return result or "OCR processing failed - no text detected\""""
        
        # Replace function implementation
        updated_content = re.sub(pattern, replacement, content)
        
        # Check if a change was actually made
        if updated_content == content:
            print("Pattern not found in the file.")
            return False
        
        # Write the updated content back to the file
        with open(ocr_module_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)
        
        print("OCR module updated successfully.")
        return True
    
    except Exception as e:
        print(f"Error updating OCR module: {e}")
        return False

if __name__ == "__main__":
    update_ocr_module()
