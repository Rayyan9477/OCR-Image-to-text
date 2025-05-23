# ocr_module_lite.py - A lightweight version of the OCR module
import os
import re
import sys
import time
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import importlib.util

# Try to import OpenCV if available
try:
    import cv2
    has_cv2 = True
except ImportError:
    has_cv2 = False

# Try to import Tesseract OCR if available
try:
    import pytesseract
    has_tesseract = True
except ImportError:
    has_tesseract = False

class LiteOCR:
    """Lightweight OCR class that doesn't rely on heavy dependencies"""
    
    def __init__(self):
        self.available_engines = self._check_engines()
        
    def _check_engines(self):
        """Check which OCR engines are available"""
        engines = {}
        
        # Check Tesseract
        if has_tesseract:
            try:
                pytesseract.get_tesseract_version()
                engines["tesseract"] = True
            except:
                engines["tesseract"] = False
        else:
            engines["tesseract"] = False
        
        # Check EasyOCR
        try:
            importlib.util.find_spec('easyocr')
            engines["easyocr"] = True
        except ImportError:
            engines["easyocr"] = False
        
        # Check PaddleOCR
        try:
            importlib.util.find_spec('paddleocr')
            engines["paddleocr"] = True
        except ImportError:
            engines["paddleocr"] = False
            
        return engines
    
    def preprocess_image(self, image):
        """Basic image preprocessing"""
        if has_cv2:
            # OpenCV-based preprocessing
            img_np = np.array(image)
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return Image.fromarray(thresh)
        else:
            # PIL-based preprocessing
            img = image.convert('L')  # Convert to grayscale
            img = img.filter(ImageFilter.GaussianBlur(radius=1))
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            return img
    
    def perform_ocr(self, image, engine="auto", preserve_layout=True):
        """Perform OCR on the image using the best available engine"""
        
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        
        # Auto-select the best available engine if not specified
        if engine == "auto":
            if self.available_engines.get("paddleocr", False):
                engine = "paddleocr"
            elif self.available_engines.get("easyocr", False):
                engine = "easyocr"
            elif self.available_engines.get("tesseract", False):
                engine = "tesseract"
            else:
                return "No OCR engines available. Please install pytesseract, EasyOCR, or PaddleOCR."
        
        # Perform OCR with the selected engine
        if engine == "tesseract" and self.available_engines.get("tesseract", False):
            return self._tesseract_ocr(processed_image, preserve_layout)
        elif engine == "easyocr" and self.available_engines.get("easyocr", False):
            return self._easyocr_ocr(processed_image, preserve_layout)
        elif engine == "paddleocr" and self.available_engines.get("paddleocr", False):
            return self._paddleocr_ocr(processed_image, preserve_layout)
        elif engine == "combined":
            # Try all available engines and return the best result
            results = []
            
            if self.available_engines.get("paddleocr", False):
                paddle_result = self._paddleocr_ocr(processed_image, preserve_layout)
                results.append(("paddleocr", paddle_result))
                
            if self.available_engines.get("easyocr", False):
                easy_result = self._easyocr_ocr(processed_image, preserve_layout)
                results.append(("easyocr", easy_result))
                
            if self.available_engines.get("tesseract", False):
                tesseract_result = self._tesseract_ocr(processed_image, preserve_layout)
                results.append(("tesseract", tesseract_result))
                
            if not results:
                return "No OCR engines available."
                
            # Choose the result with the highest score
            best_result = None
            best_score = -1
            
            for engine_name, text in results:
                score = self._score_result(text)
                if score > best_score:
                    best_score = score
                    best_result = text
                    
            return best_result or "No text detected."
        else:
            return f"OCR engine '{engine}' is not available."
    
    def _tesseract_ocr(self, image, preserve_layout=True):
        """Perform OCR using pytesseract"""
        try:
            config = ""
            if preserve_layout:
                # PSM 6: Assume a single uniform block of text (preserves layout better)
                config = "--psm 6"
            else:
                # PSM 3: Fully automatic page segmentation (better for mixed content)
                config = "--psm 3"
            
            text = pytesseract.image_to_string(image, config=config)
            return text
        except Exception as e:
            print(f"Tesseract OCR error: {e}")
            return ""
    
    def _easyocr_ocr(self, image, preserve_layout=True):
        """Perform OCR using EasyOCR"""
        try:
            import easyocr
            reader = easyocr.Reader(['en'])
            result = reader.readtext(np.array(image))
            
            if not preserve_layout:
                # Simple concatenation of detected text
                return " ".join([r[1] for r in result])
            else:
                # Sort by y-coordinate for layout preservation
                sorted_results = sorted(result, key=lambda x: (x[0][0][1] + x[0][2][1]) / 2)
                
                lines = []
                current_line = []
                last_y = None
                y_tolerance = image.height * 0.01
                
                for box, text, _ in sorted_results:
                    current_y = (box[0][1] + box[2][1]) / 2
                    
                    if last_y is None or abs(current_y - last_y) <= y_tolerance:
                        current_line.append((box, text))
                    else:
                        # Sort by x-coordinate
                        current_line.sort(key=lambda x: x[0][0][0])
                        lines.append(" ".join([word[1] for word in current_line]))
                        current_line = [(box, text)]
                    
                    last_y = current_y
                
                if current_line:
                    current_line.sort(key=lambda x: x[0][0][0])
                    lines.append(" ".join([word[1] for word in current_line]))
                
                return "\n".join(lines)
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return ""
    
    def _paddleocr_ocr(self, image, preserve_layout=True):
        """Perform OCR using PaddleOCR"""
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            result = ocr.ocr(np.array(image), cls=True)
            
            if not preserve_layout:
                # Simple concatenation
                return " ".join([line[1][0] for line in result[0] if line[1][0]])
            else:
                # Sort by y-coordinate
                if result[0]:
                    sorted_results = sorted(result[0], key=lambda x: (x[0][0][1] + x[0][3][1]) / 2)
                    
                    lines = []
                    current_line = []
                    last_y = None
                    y_tolerance = image.height * 0.01
                    
                    for box in sorted_results:
                        current_y = (box[0][0][1] + box[0][3][1]) / 2
                        
                        if last_y is None or abs(current_y - last_y) <= y_tolerance:
                            current_line.append(box)
                        else:
                            # Sort by x-coordinate
                            current_line.sort(key=lambda x: x[0][0][0])
                            lines.append(" ".join([word[1][0] for word in current_line]))
                            current_line = [box]
                        
                        last_y = current_y
                    
                    if current_line:
                        current_line.sort(key=lambda x: x[0][0][0])
                        lines.append(" ".join([word[1][0] for word in current_line]))
                    
                    return "\n".join(lines)
                return ""
        except Exception as e:
            print(f"PaddleOCR error: {e}")
            return ""
    
    def _score_result(self, text):
        """Score OCR result based on quality heuristics"""
        if not text:
            return 0
        
        score = 0
        
        # Length of text (more is usually better)
        score += min(len(text) / 100, 10)
        
        # Word count (more distinct words is better)
        words = set(re.findall(r'\b\w+\b', text.lower()))
        score += min(len(words) / 10, 10)
        
        # Ratio of alphanumeric characters (higher is better)
        alnum_count = sum(c.isalnum() for c in text)
        if len(text) > 0:
            alnum_ratio = alnum_count / len(text)
            score += alnum_ratio * 10
        
        # Detect if text has good formatting
        if '\n' in text:
            score += 5
        
        # Penalize very short output
        if len(text) < 20:
            score -= 5
        
        return score

# For direct testing
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            image = Image.open(image_path)
            ocr = LiteOCR()
            print("Available OCR engines:", ocr.available_engines)
            result = ocr.perform_ocr(image, "auto")
            print("\nOCR Result:")
            print("-" * 40)
            print(result)
        else:
            print(f"Image file not found: {image_path}")
    else:
        print("Please provide an image path: python ocr_module_lite.py <image_path>")