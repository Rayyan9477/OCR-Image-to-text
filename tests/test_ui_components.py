import pytest
import streamlit as st
from src.ui.components import display_ocr_settings, display_extracted_text, display_qa_interface
from src.core.model_manager import ModelManager
from unittest.mock import patch, MagicMock

@pytest.fixture
def model_manager():
    with patch('src.core.model_manager._check_available_modules') as mock_check:
        mock_check.return_value = {
            'paddleocr_available': False,
            'easyocr_available': False,
            'tesseract_available': False
        }
        return ModelManager()

@pytest.mark.skipif(not ModelManager().OPTIONAL_MODULES.get('paddleocr_available', False) and 
                   not ModelManager().OPTIONAL_MODULES.get('easyocr_available', False),
                   reason="No OCR engines available")
def test_display_ocr_settings(model_manager):
    """Test OCR settings display"""
    with patch('streamlit.selectbox') as mock_selectbox, \
         patch('streamlit.checkbox') as mock_checkbox:
        # Mock the Streamlit widgets
        mock_selectbox.side_effect = ['paddle', 'English']
        mock_checkbox.side_effect = [True, True]
        
        settings = display_ocr_settings(model_manager)
        assert isinstance(settings, dict)
        assert 'engine' in settings
        assert 'preserve_layout' in settings
        assert 'preprocessing' in settings
        assert 'language' in settings
        
        # Check valid engine values
        assert settings['engine'] in ['paddle', 'easy', 'combined']
        
        # Check boolean settings
        assert isinstance(settings['preserve_layout'], bool)
        assert isinstance(settings['preprocessing'], bool)
        
        # Check language setting
        assert settings['language'] in ["English", "Multi-language (Slower)"]

def test_display_extracted_text():
    """Test extracted text display"""
    with patch('streamlit.text_area') as mock_text_area:
        test_text = "This is a test text"
        display_extracted_text(test_text)
        mock_text_area.assert_called_once()

def test_display_qa_interface():
    """Test Q&A interface display"""
    with patch('streamlit.text_input') as mock_text_input, \
         patch('streamlit.button') as mock_button:
        test_text = "This is a test text"
        display_qa_interface(test_text)
        mock_text_input.assert_called_once()
        mock_button.assert_called_once()

def test_display_qa_interface_empty():
    """Test Q&A interface with empty text"""
    with patch('streamlit.text_input') as mock_text_input, \
         patch('streamlit.button') as mock_button:
        display_qa_interface("")
        mock_text_input.assert_called_once()
        mock_button.assert_called_once() 