import cv2
import numpy as np
from PIL import Image, ImageEnhance
import logging
from typing import Union, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    MAX_IMAGE_SIZE = 4096  # Maximum dimension
    MIN_IMAGE_SIZE = 32    # Minimum dimension

    @staticmethod
    def validate_image_size(image: Union[Image.Image, np.ndarray]) -> Tuple[int, int]:
        """
        Validate and adjust image size if necessary
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            tuple: (width, height) of the validated image
        """
        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
        else:
            width, height = image.size

        # Check minimum size
        if width < ImagePreprocessor.MIN_IMAGE_SIZE or height < ImagePreprocessor.MIN_IMAGE_SIZE:
            raise ValueError(f"Image too small: {width}x{height}. Minimum size is {ImagePreprocessor.MIN_IMAGE_SIZE}x{ImagePreprocessor.MIN_IMAGE_SIZE}")

        # Check maximum size
        if width > ImagePreprocessor.MAX_IMAGE_SIZE or height > ImagePreprocessor.MAX_IMAGE_SIZE:
            ratio = ImagePreprocessor.MAX_IMAGE_SIZE / max(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            logger.warning(f"Image resized from {width}x{height} to {new_width}x{new_height}")
            return new_width, new_height

        return width, height

    @staticmethod
    def enhance_image(image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Enhance image quality for better OCR results
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            PIL Image: Enhanced image
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Validate size
            width, height = ImagePreprocessor.validate_image_size(image)
            if (width, height) != image.size:
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
            
            return image
            
        except Exception as e:
            logger.error(f"Error in image enhancement: {str(e)}")
            raise

    @staticmethod
    def denoise_image(image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Remove noise from image while preserving color
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            PIL Image: Denoised image
        """
        try:
            # Convert to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Validate size
            width, height = ImagePreprocessor.validate_image_size(image)
            if (width, height) != image.shape[:2][::-1]:
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
            
            # Apply denoising to each color channel
            if len(image.shape) == 3:
                denoised = np.zeros_like(image)
                for i in range(3):
                    denoised[:,:,i] = cv2.fastNlMeansDenoisingColored(image)[:,:,i]
            else:
                denoised = cv2.fastNlMeansDenoising(image)
            
            return Image.fromarray(denoised)
            
        except Exception as e:
            logger.error(f"Error in image denoising: {str(e)}")
            raise

    @staticmethod
    def correct_skew(image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Correct image skew
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            PIL Image: Corrected image
        """
        try:
            # Convert to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Validate size
            width, height = ImagePreprocessor.validate_image_size(image)
            if (width, height) != image.shape[:2][::-1]:
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
            
            # Convert to grayscale for edge detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Find edges
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find lines
            lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
            
            if lines is not None:
                # Calculate angles
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = theta * 180 / np.pi
                    angles.append(angle % 180)
                
                # Find most common angle
                angles = np.array(angles)
                hist, _ = np.histogram(angles, bins=180)
                max_bin = np.argmax(hist)
                
                # Calculate skew angle
                skew_angle = max_bin - 90
                
                # Rotate image if skew is significant
                if abs(skew_angle) > 0.5:
                    image = Image.fromarray(image)
                    image = image.rotate(skew_angle, expand=True, resample=Image.Resampling.LANCZOS)
                    return image
            
            return Image.fromarray(image)
            
        except Exception as e:
            logger.error(f"Error in skew correction: {str(e)}")
            raise

    @staticmethod
    def preprocess(image: Union[Image.Image, np.ndarray], enhance: bool = True, 
                  denoise: bool = True, correct_skew: bool = True) -> Image.Image:
        """
        Apply all preprocessing steps
        
        Args:
            image: PIL Image or numpy array
            enhance: Whether to enhance image
            denoise: Whether to denoise image
            correct_skew: Whether to correct skew
            
        Returns:
            PIL Image: Preprocessed image
        """
        try:
            # Validate input
            if image is None:
                raise ValueError("Input image is None")
            
            # Validate size first
            ImagePreprocessor.validate_image_size(image)
            
            if enhance:
                image = ImagePreprocessor.enhance_image(image)
            
            if denoise:
                image = ImagePreprocessor.denoise_image(image)
            
            if correct_skew:
                image = ImagePreprocessor.correct_skew(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise 