import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from src.ocr.nanonets_ocr import NanonetsOCR
from src.ocr.preprocessor import ImagePreprocessor
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
MAX_IMAGE_DIMENSION = 4096  # Maximum image dimension

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize OCR engine
ocr_engine = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(image):
    """Validate image dimensions and format"""
    try:
        if not isinstance(image, Image.Image):
            raise ValueError("Invalid image format")
        
        width, height = image.size
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            raise ValueError(f"Image dimensions exceed maximum allowed size of {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}")
        
        if width < 32 or height < 32:
            raise ValueError("Image dimensions too small")
        
        return True
    except Exception as e:
        logger.error(f"Image validation failed: {str(e)}")
        raise

def initialize_ocr():
    """Initialize OCR engine"""
    global ocr_engine
    if ocr_engine is None:
        try:
            ocr_engine = NanonetsOCR()
            logger.info("OCR engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR engine: {str(e)}")
            raise

@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler"""
    logger.error(f"Unhandled error: {str(error)}")
    return jsonify({
        "error": str(error),
        "message": "An unexpected error occurred"
    }), 500

@app.route('/health', methods=['GET'])
@limiter.exempt
def health_check():
    """Health check endpoint"""
    try:
        if ocr_engine is None:
            initialize_ocr()
        return jsonify({
            "status": "healthy",
            "ocr_engine": "initialized"
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/ocr', methods=['POST'])
@limiter.limit("10 per minute")
def process_image():
    """Process image and extract text"""
    temp_dir = None
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        # Check if file is valid
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                "error": "File type not allowed",
                "allowed_types": list(ALLOWED_EXTENSIONS)
            }), 400
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        filepath = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(filepath)
        
        try:
            # Load and validate image
            image = Image.open(filepath)
            validate_image(image)
            
            # Get preprocessing options from request
            enhance = request.form.get('enhance', 'true').lower() == 'true'
            denoise = request.form.get('denoise', 'true').lower() == 'true'
            correct_skew = request.form.get('correct_skew', 'true').lower() == 'true'
            
            # Preprocess image
            preprocessor = ImagePreprocessor()
            processed_image = preprocessor.preprocess(
                image,
                enhance=enhance,
                denoise=denoise,
                correct_skew=correct_skew
            )
            
            # Initialize OCR if not already done
            if ocr_engine is None:
                initialize_ocr()
            
            # Extract text
            result = ocr_engine.extract_text(processed_image)
            
            return jsonify(result)
            
        finally:
            # Clean up temporary files
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return jsonify({"error": str(e)}), 500

@app.route('/batch', methods=['POST'])
@limiter.limit("5 per minute")
def process_batch():
    """Process multiple images in batch"""
    temp_dir = None
    try:
        # Check if files are present in request
        if 'files[]' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files[]')
        
        if not files:
            return jsonify({"error": "No files selected"}), 400
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Process each file
        results = []
        for file in files:
            if file.filename == '':
                continue
                
            if not allowed_file(file.filename):
                continue
            
            # Save file temporarily
            filepath = os.path.join(temp_dir, secure_filename(file.filename))
            file.save(filepath)
            
            try:
                # Load and validate image
                image = Image.open(filepath)
                validate_image(image)
                
                # Get preprocessing options from request
                enhance = request.form.get('enhance', 'true').lower() == 'true'
                denoise = request.form.get('denoise', 'true').lower() == 'true'
                correct_skew = request.form.get('correct_skew', 'true').lower() == 'true'
                
                # Preprocess image
                preprocessor = ImagePreprocessor()
                processed_image = preprocessor.preprocess(
                    image,
                    enhance=enhance,
                    denoise=denoise,
                    correct_skew=correct_skew
                )
                
                # Initialize OCR if not already done
                if ocr_engine is None:
                    initialize_ocr()
                
                # Extract text
                result = ocr_engine.extract_text(processed_image)
                result['filename'] = file.filename
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return jsonify({"results": results})
        
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize OCR engine on startup
    try:
        initialize_ocr()
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise