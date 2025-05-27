#!/usr/bin/env python3
"""
CLI Entry Point for OCR Application
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from ocr_app.ui.cli import OCRCLI
except ImportError:
    # Try relative import if absolute fails
    try:
        from .ocr_app.ui.cli import OCRCLI
    except ImportError:
        print("Error: Could not import OCRCLI. Please check your installation.")
        sys.exit(1)

def main():
    """Main entry point for CLI application"""
    cli = OCRCLI()
    
    try:
        result = cli.run()
        return result
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 130
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
