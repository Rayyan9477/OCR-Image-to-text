# ocr_module.py
import re
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance
import warnings
import model_manager
from model_manager import get_paddle_ocr, get_easy_ocr, get_ocr_config
import concurrent.futures
from threading import Lock
import time
from dolphin_ocr import dolphin_ocr as dolphin_model_ocr

warnings.filterwarnings('ignore')

# Thread-safe performance tracking
_perf_lock = Lock()
_performance_stats = {
    "paddle": {"calls": 0, "total_time": 0},
    "easy": {"calls": 0, "total_time": 0},
    "preprocess": {"calls": 0, "total_time": 0}
}

def update_perf_stats(engine, elapsed_time):
    """Update performance statistics for the OCR engines"""
    with _perf_lock:
        if engine in _performance_stats:
            _performance_stats[engine]["calls"] += 1
            _performance_stats[engine]["total_time"] += elapsed_time

def get_performance_stats():
    """Get performance statistics for the OCR engines"""
    with _perf_lock:
        stats = {}
        for engine, data in _performance_stats.items():
            if data["calls"] > 0:
                avg_time = data["total_time"] / data["calls"]
                stats[engine] = {
                    "calls": data["calls"],
                    "total_time": data["total_time"],
                    "avg_time": avg_time
                }
    return stats

def preprocess_image(image, enhance=True, denoise=True, adaptive_threshold=True):
    """
    Apply various preprocessing techniques to improve OCR quality
    with options to control which enhancements are applied
    """
    start_time = time.time()
    
    try:
        # Convert PIL Image to OpenCV format
        if isinstance(image, Image.Image):
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            image_cv = image
            
        # Apply grayscale
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply image enhancement if requested
        if enhance:
            # Convert back to PIL Image for enhancement
            pil_img = Image.fromarray(gray)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(1.5)
            
            # Convert back to OpenCV
            gray = np.array(pil_img)
        
        # Apply adaptive thresholding if requested
        if adaptive_threshold:
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            # Simple thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply denoising if requested
        if denoise:
            processed = cv2.medianBlur(thresh, 3)
        else:
            processed = thresh
        
        # Convert back to PIL Image for compatibility
        result = Image.fromarray(processed)
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        # Return original image if preprocessing fails
        if isinstance(image, Image.Image):
            result = image
        else:
            result = Image.fromarray(image_cv)
    
    elapsed_time = time.time() - start_time
    update_perf_stats("preprocess", elapsed_time)
    
    return result

def correct_orientation(image):
    """Detect and correct image orientation"""
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
            
        rotation = 0
        
        # Try using Tesseract's OSD
        try:
            osd = pytesseract.image_to_osd(img_np)
            rotation = int(re.search(r'(?<=Rotate: )\d+', osd).group(0))
        except:
            # If Tesseract OSD fails, try PaddleOCR's angle classifier
            try:
                paddle = get_paddle_ocr()
                result = paddle.ocr(img_np, cls=True)
                if result[0] and result[0][0]:  # Check if angle detection returned results
                    angle = result[0][0]
                    if angle != 0:
                        rotation = angle
            except:
                # Both methods failed, try to detect rotation using contours
                try:
                    # Convert to grayscale
                    if len(img_np.shape) == 3:
                        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = img_np
                    
                    # Find edges
                    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                    
                    # Find lines using Hough transform
                    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
                    
                    if lines is not None and len(lines) > 0:
                        # Calculate orientations of lines
                        angles = []
                        for line in lines:
                            rho, theta = line[0]
                            angle = theta * 180 / np.pi
                            angles.append(angle % 180)
                        
                        # Find most common angle
                        angles = np.array(angles)
                        hist, _ = np.histogram(angles, bins=180)
                        max_bin = np.argmax(hist)
                        
                        # Check if we need to rotate the image
                        if max_bin < 45:
                            rotation = 90
                        elif max_bin > 135:
                            rotation = 270
                        elif max_bin > 45 and max_bin < 135:
                            rotation = 180
                except:
                    pass  # Failed to detect orientation using contours
        
        if rotation != 0:
            return image.rotate(360 - rotation, expand=True)
        
        return image
    except Exception as e:
        print(f"Could not determine orientation: {e}")
        return image

def perform_ocr(image, engine="paddle", preserve_layout=True, enhance_image=True):
    """
    Perform OCR on the image using the selected engine
    
    Args:
        image: PIL Image or numpy array
        engine: OCR engine to use ('paddle', 'easy', or 'combined')
        preserve_layout: Whether to preserve the text layout
        enhance_image: Whether to apply image enhancements
        
    Returns:
        Extracted text (formatted when preserve_layout is True)
    """
    try:
        # Check if image is valid
        if image is None or (isinstance(image, np.ndarray) and image.size == 0):
            return "Error: Invalid image"
            
        # Check if any OCR engines are available
        paddle_available = model_manager.OPTIONAL_MODULES.get('paddleocr_available', False)
        easy_available = model_manager.OPTIONAL_MODULES.get('easyocr_available', False)
        tesseract_available = False
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            tesseract_available = True
        except:
            pass
        
        if not (paddle_available or easy_available or tesseract_available):
            return "Error: No OCR engines available. Please install at least one of: PaddleOCR, EasyOCR, or pytesseract."
            
        # Correct orientation
        image = correct_orientation(image)
        
        # Preprocess image with enhancement options from config
        config = get_ocr_config()
        preprocessing_options = config.get("preprocessing", {})
        
        processed_image = preprocess_image(
            image, 
            enhance=enhance_image and preprocessing_options.get("default_contrast", 1.5) > 1,
            denoise=preprocessing_options.get("denoise", True),
            adaptive_threshold=preprocessing_options.get("adaptive_threshold", True)
        )
        
        # Import the tesseract fallback
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
        elif engine == "dolphin":
            print("Performing OCR using Dolphin model...")
            try:
                result = dolphin_model_ocr(processed_image)
                return result or "OCR processing failed - no text detected"
            except Exception as e:
                return f"Error in Dolphin OCR: {str(e)}"
        else:  # Combined approach with parallel processing
            result = combined_ocr(processed_image, preserve_layout)
            
            # If combined approach fails, try pytesseract as last resort
            if not result and has_tesseract_fallback:
                print("All OCR engines failed, falling back to pytesseract OCR...")
                result = pytesseract_ocr(processed_image, preserve_layout)
            
            return result or "OCR processing failed - no text detected"
            
    except Exception as e:
        return f"Error during OCR: {str(e)}"

def paddle_ocr(image, preserve_layout=True):
    """Perform OCR using PaddleOCR"""
    start_time = time.time()
    
    try:
        # Check if PaddleOCR is available
        try:
            from paddleocr import PaddleOCR
            has_paddle = True
        except ImportError:
            print("PaddleOCR not available")
            return ""
            
        # Get PaddleOCR instance
        paddle = get_paddle_ocr()
        
        # If the model manager failed to provide a model, try direct initialization
        if paddle is None:
            try:
                print("Initializing PaddleOCR directly...")
                paddle = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            except Exception as e:
                print(f"Error initializing PaddleOCR directly: {e}")
                return ""
        
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
        
        result = paddle.ocr(img_np, cls=True)
        
        if not preserve_layout:
            # Simple concatenation of detected text
            text = " ".join([line[1][0] for line in result[0] if line[1][0]])
            return text
        else:
            # Sort results by y-coordinate (top to bottom)
            if result[0]:
                sorted_results = sorted(result[0], key=lambda x: (x[0][0][1] + x[0][3][1]) / 2)
                
                # Group lines by their y-position (with adaptive tolerance)
                # Use a percentage of the image height for better adaptability
                img_height = image.height if isinstance(image, Image.Image) else img_np.shape[0]
                y_tolerance = max(10, img_height * 0.01)  # At least 10px or 1% of image height
                
                lines = []
                current_line = []
                last_y = None
                
                for box in sorted_results:
                    current_y = (box[0][0][1] + box[0][3][1]) / 2
                    
                    if last_y is None or abs(current_y - last_y) <= y_tolerance:
                        current_line.append(box)
                    else:
                        # Sort words in the line by x-coordinate
                        current_line.sort(key=lambda x: x[0][0][0])
                        lines.append(current_line)
                        current_line = [box]
                    
                    last_y = current_y
                
                # Add the last line
                if current_line:
                    current_line.sort(key=lambda x: x[0][0][0])
                    lines.append(current_line)
                
                # Join words in each line with spaces, and lines with newlines
                formatted_text = "\n".join([" ".join([word[1][0] for word in line]) for line in lines])
                
                # Update performance stats
                elapsed_time = time.time() - start_time
                update_perf_stats("paddle", elapsed_time)
                
                return formatted_text
            else:
                return ""
    except Exception as e:
        print(f"Error in PaddleOCR: {str(e)}")
        return ""

def easyocr_ocr(image, preserve_layout=True):
    """Perform OCR using EasyOCR"""
    start_time = time.time()
    
    try:
        # Check if EasyOCR is available
        try:
            import easyocr
            has_easy_ocr = True
        except ImportError:
            print("EasyOCR not available")
            return ""
            
        # Get EasyOCR reader
        reader = get_easy_ocr()
        
        # If the model manager failed to provide a reader, try direct initialization
        if reader is None:
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
        
        result = reader.readtext(img_np)
        
        if not preserve_layout:
            # Simple concatenation of detected text
            text = " ".join([res[1] for res in result])
            return text
        else:
            # Sort by y-coordinate (top to bottom)
            sorted_results = sorted(result, key=lambda x: (x[0][0][1] + x[0][2][1]) / 2)
            
            # Group lines by their y-position (with adaptive tolerance)
            # Use a percentage of the image height for better adaptability
            img_height = image.height if isinstance(image, Image.Image) else img_np.shape[0]
            y_tolerance = max(10, img_height * 0.01)  # At least 10px or 1% of image height
            
            lines = []
            current_line = []
            last_y = None
            
            for box, text, conf in sorted_results:
                current_y = (box[0][1] + box[2][1]) / 2
                
                if last_y is None or abs(current_y - last_y) <= y_tolerance:
                    current_line.append((box, text, conf))
                else:
                    # Sort words in the line by x-coordinate
                    current_line.sort(key=lambda x: x[0][0][0])
                    lines.append(current_line)
                    current_line = [(box, text, conf)]
                
                last_y = current_y
            
            # Add the last line
            if current_line:
                current_line.sort(key=lambda x: x[0][0][0])
                lines.append(current_line)
            
            # Join words in each line with spaces, and lines with newlines
            formatted_text = "\n".join([" ".join([word[1] for word in line]) for line in lines])
            
            # Update performance stats
            elapsed_time = time.time() - start_time
            update_perf_stats("easy", elapsed_time)
            
            return formatted_text
    except Exception as e:
        print(f"Error in EasyOCR: {str(e)}")
        return ""

def pytesseract_ocr(image, preserve_layout=True):
    """Fallback OCR using pytesseract when other engines fail"""
    try:
        # Convert to numpy array if it's a PIL Image
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image

        # Use pytesseract to extract text
        if preserve_layout:
            # Use page segmentation mode 6 (assumes a single uniform block of text)
            text = pytesseract.image_to_string(img_np, config='--psm 6')
        else:
            # Use page segmentation mode 3 (fully automatic page segmentation)
            text = pytesseract.image_to_string(img_np, config='--psm 3')
            
        return text
    except Exception as e:
        print(f"Error in pytesseract OCR: {str(e)}")
        return ""

def combined_ocr(image, preserve_layout=True):
    """Combine results from multiple OCR engines for better accuracy using parallel processing"""
    try:
        # Run both OCR engines in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            paddle_future = executor.submit(paddle_ocr, image, preserve_layout)
            easy_future = executor.submit(easyocr_ocr, image, preserve_layout)
            
            paddle_text = paddle_future.result()
            easy_text = easy_future.result()
        
        # If one engine fails, use the other
        if not paddle_text and not easy_text:
            # Both failed, try pytesseract as last resort
            print("Both OCR engines failed, trying pytesseract fallback...")
            return pytesseract_ocr(image, preserve_layout)
        elif not paddle_text:
            return easy_text
        elif not easy_text:
            return paddle_text
            
        # Score the results from each engine
        paddle_score = score_ocr_result(paddle_text)
        easy_score = score_ocr_result(easy_text)
        
        # Use the result with the higher score
        if paddle_score >= easy_score:
            return paddle_text
        else:
            return easy_text
            
    except Exception as e:
        print(f"Error in combined OCR: {str(e)}")
        # Fallback to sequential processing
        paddle_text = paddle_ocr(image, preserve_layout)
        if paddle_text:
            return paddle_text
        
        easy_text = easyocr_ocr(image, preserve_layout)
        if easy_text:
            return easy_text
            
        # Last resort: pytesseract
        return pytesseract_ocr(image, preserve_layout)

def score_ocr_result(text):
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

def detect_tables(image, min_lines=3):
    """Detect if the image contains tables and extract their structure"""
    try:
        # Convert PIL Image to OpenCV format if needed
        if isinstance(image, Image.Image):
            img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_np = image
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine horizontal and vertical lines
        table_mask = cv2.add(horizontal_lines, vertical_lines)
        
        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find intersections (grid points)
        if contours and len(contours) > 0:
            # Count horizontal and vertical lines
            horizontal_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            vertical_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check if we have enough lines to form a table
            if len(horizontal_contours) >= min_lines and len(vertical_contours) >= min_lines:
                return True
        
        # Alternative approach: check for evenly spaced text lines
        # This helps detect tables that don't have explicit grid lines
        try:
            if isinstance(image, Image.Image):
                # Get text with bounding boxes
                reader = get_easy_ocr()
                result = reader.readtext(np.array(image))
                
                # Check if we have multiple text items
                if len(result) > 10:
                    # Extract y-coordinates of text items
                    y_coords = [(box[0][1] + box[2][1]) / 2 for box, _, _ in result]
                    y_coords.sort()
                    
                    # Calculate distances between adjacent y-coordinates
                    diffs = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
                    
                    # Check if we have consistent spacing (indicating table rows)
                    if len(diffs) > 5:
                        mean_diff = sum(diffs) / len(diffs)
                        std_diff = np.std(diffs)
                        
                        # If standard deviation is low compared to mean, we likely have a table
                        if std_diff < mean_diff * 0.5 and std_diff > 0:
                            return True
        except Exception as inner_e:
            print(f"Error in alternative table detection: {str(inner_e)}")
                
        return False
    except Exception as e:
        print(f"Error in table detection: {str(e)}")
        return False

def detect_image_quality(image):
    """Detect image quality issues that might affect OCR performance"""
    try:
        # Convert PIL Image to OpenCV format if needed
        if isinstance(image, Image.Image):
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_cv = image
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        issues = []
        
        # Check image resolution (DPI)
        height, width = gray.shape
        if width < 1000 or height < 1000:
            issues.append("low_resolution")
        
        # Check for blur using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        if laplacian_var < 100:
            issues.append("blurry")
            
        # Check for low contrast
        min_val, max_val, _, _ = cv2.minMaxLoc(gray)
        if max_val - min_val < 50:
            issues.append("low_contrast")
            
        # Check for noise using standard deviation in a high-pass filter
        high_pass = cv2.subtract(gray, cv2.GaussianBlur(gray, (5, 5), 0))
        if high_pass.std() > 20:
            issues.append("noise")
            
        return issues
    except Exception as e:
        print(f"Error in image quality detection: {str(e)}")
        return []