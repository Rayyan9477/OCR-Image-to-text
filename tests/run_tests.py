import pytest
import os
import sys
from create_test_image import create_test_images

def run_tests():
    """Run all tests in the tests directory"""
    # Create test images first
    create_test_images()
    
    # Add src directory to Python path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Run pytest with verbosity
    pytest.main([
        '--verbose',
        '--cov=src',
        '--cov-report=term-missing',
        '--cov-report=html',
        'tests/'
    ])

if __name__ == "__main__":
    run_tests() 