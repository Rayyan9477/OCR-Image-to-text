# ocr_module.py
import re
from turtle import st
import easyocr
import numpy as np
import pytesseract

def perform_ocr(image):
    try:
        reader = easyocr.Reader(['en'])
        image = correct_orientation(image)
        result = reader.readtext(np.array(image))
        extracted_text = " ".join([res[1] for res in result])
        return extracted_text
    except Exception as e:
        return f"Error during OCR: {str(e)}"
    
def correct_orientation(image):
    try:
        osd = pytesseract.image_to_osd(image)
        rotation = int(re.search('(?<=Rotate: )\d+', osd).group(0))
        if rotation != 0:
            image = image.rotate(360 - rotation, expand=True)
        return image
    except Exception as e:
        st.warning(f"Could not determine orientation: {e}")
        return image