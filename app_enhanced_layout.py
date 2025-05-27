#!/usr/bin/env python
"""
Final Enhanced OCR Application with Precision Layout Preservation
Combines multi-engine OCR with ultra-accurate layout formatting
"""

import os
import sys
import logging
import streamlit as st
import time
from PIL import Image
import base64
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
import json
import cv2
import numpy as np

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set environment variables for consistent behavior
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"

# Import enhanced multi-engine system
try:
    from enhanced_multi_engine_ocr import extract_text_enhanced_multi_engine
    ENHANCED_OCR_AVAILABLE = True
except ImportError as e:
    ENHANCED_OCR_AVAILABLE = False
    logging.warning(f"Enhanced multi-engine OCR system not available: {e}")

# Import precision layout system
try:
    from precision_layout_ocr import extract_text_with_precision_layout
    PRECISION_LAYOUT_AVAILABLE = True
except ImportError as e:
    PRECISION_LAYOUT_AVAILABLE = False
    logging.warning(f"Precision layout OCR not available: {e}")

# Import layout-aware system
try:
    from layout_aware_ocr import extract_text_with_layout_preservation
    LAYOUT_AWARE_AVAILABLE = True
except ImportError as e:
    LAYOUT_AWARE_AVAILABLE = False
    logging.warning(f"Layout-aware OCR not available: {e}")

def load_css():
    """Load custom CSS for better styling"""
    css = """
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .engine-status {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    
    .engine-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        min-width: 150px;
    }
    
    .engine-success {
        border-left: 4px solid #28a745;
    }
    
    .engine-error {
        border-left: 4px solid #dc3545;
    }
    
    .layout-preview {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background: #fafafa;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .text-comparison {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .comparison-panel {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        background: white;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .download-section {
        background: #e8f5e8;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .precision-controls {
        background: #f0f8ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #007bff;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def create_download_link(content: str, filename: str, mime_type: str = "text/plain") -> str:
    """Create a download link for content"""
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:{mime_type};base64,{b64}" download="{filename}" class="download-btn">📥 Download {filename}</a>'

def display_layout_analysis(analysis: Dict[str, Any]):
    """Display layout analysis results"""
    if not analysis:
        st.warning("No layout analysis available")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Text Elements", analysis.get('total_elements', 0))
        st.metric("Line Groups", analysis.get('line_groups', 0))
    
    with col2:
        columns_info = analysis.get('columns', {})
        st.metric("Columns Detected", columns_info.get('count', 1))
        structure = analysis.get('structure', {})
        st.metric("Titles Found", len(structure.get('titles', [])))
    
    with col3:
        st.metric("Bullet Points", len(structure.get('bullet_points', [])))
        st.metric("Numbered Lists", len(structure.get('numbered_lists', [])))
    
    # Display spacing information
    spacing = analysis.get('spacing', {})
    if spacing:
        st.subheader("📏 Spacing Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Average Line Height:** {spacing.get('avg_line_height', 0):.1f}px")
            st.write(f"**Average Line Gap:** {spacing.get('avg_line_gap', 0):.1f}px")
        with col2:
            indents = spacing.get('common_indentations', [])
            st.write(f"**Common Indentations:** {', '.join(map(str, indents)) if indents else 'None'}")

def display_engine_status(result: Dict[str, Any]):
    """Display status of each OCR engine"""
    st.subheader("🔧 Engine Status")
    
    if 'individual_results' not in result:
        st.warning("Engine status information not available")
        return
    
    engine_html = '<div class="engine-status">'
    
    for engine_result in result['individual_results']:
        engine_name = engine_result['engine']
        status = engine_result['status']
        confidence = engine_result.get('confidence', 0)
        
        status_class = 'engine-success' if status == 'success' else 'engine-error'
        status_icon = '✅' if status == 'success' else '❌'
        
        engine_html += f'''
        <div class="engine-card {status_class}">
            <h4>{status_icon} {engine_name}</h4>
            <p><strong>Status:</strong> {status}</p>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
        </div>
        '''
    
    engine_html += '</div>'
    st.markdown(engine_html, unsafe_allow_html=True)

def display_precision_results(precision_result: Dict[str, Any]):
    """Display precision layout results"""
    if not precision_result or 'error' in precision_result:
        st.error(f"Precision layout error: {precision_result.get('error', 'Unknown error')}")
        return
    
    st.subheader("🎯 Precision Layout Results")
    
    # Display layout analysis
    if 'layout_analysis' in precision_result:
        display_layout_analysis(precision_result['layout_analysis'])
    
    # Display different output formats
    tab1, tab2, tab3, tab4 = st.tabs(["Precision Formatted", "HTML Layout", "Markdown", "Text Elements"])
    
    with tab1:
        st.text_area(
            "Precision Formatted Text",
            value=precision_result.get('precision_formatted', ''),
            height=300,
            key="precision_text"
        )
    
    with tab2:
        html_content = precision_result.get('html_layout', '')
        if html_content:
            st.markdown("**Visual Layout Preview:**")
            st.markdown(html_content, unsafe_allow_html=True)
        else:
            st.warning("HTML layout not available")
    
    with tab3:
        st.text_area(
            "Markdown Format",
            value=precision_result.get('markdown_layout', ''),
            height=300,
            key="markdown_text"
        )
    
    with tab4:
        elements = precision_result.get('text_elements', [])
        if elements:
            st.write(f"**Total Text Elements:** {len(elements)}")
            
            # Display first few elements as example
            st.write("**Sample Elements:**")
            for i, elem in enumerate(elements[:5]):
                with st.expander(f"Element {i+1}: '{elem['text'][:50]}...'"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Position:** ({elem['x']}, {elem['y']})")
                        st.write(f"**Size:** {elem['width']} × {elem['height']}")
                    with col2:
                        st.write(f"**Confidence:** {elem['confidence']:.2%}")
                        st.write(f"**Font Size:** {elem['font_size']}px")
        else:
            st.warning("No text elements found")

def main():
    st.set_page_config(
        page_title="Enhanced OCR with Precision Layout",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Enhanced OCR with Precision Layout</h1>
        <p>Multi-engine OCR system with ultra-accurate layout preservation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OCR Engine Options
        st.subheader("OCR Engines")
        use_multi_engine = st.checkbox("Use Multi-Engine OCR", value=True, help="Use multiple OCR engines for best results")
        
        # Layout Options
        st.subheader("Layout Preservation")
        enable_precision_layout = st.checkbox("Enable Precision Layout", value=True, help="Ultra-accurate layout preservation")
        enable_basic_layout = st.checkbox("Enable Basic Layout", value=False, help="Standard layout preservation")
        
        # Output Options
        st.subheader("Output Formats")
        show_html_preview = st.checkbox("Show HTML Preview", value=True)
        show_markdown = st.checkbox("Show Markdown", value=True)
        show_comparison = st.checkbox("Show Format Comparison", value=True)
        
        # Processing Options
        st.subheader("Processing")
        show_engine_details = st.checkbox("Show Engine Details", value=True)
        show_confidence_scores = st.checkbox("Show Confidence Scores", value=True)
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📁 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image containing text to extract"
        )
        
        # System Status
        st.header("🔋 System Status")
        status_items = [
            ("Enhanced OCR", ENHANCED_OCR_AVAILABLE),
            ("Precision Layout", PRECISION_LAYOUT_AVAILABLE),
            ("Layout Aware", LAYOUT_AWARE_AVAILABLE)
        ]
        
        for name, available in status_items:
            icon = "✅" if available else "❌"
            status = "Available" if available else "Not Available"
            st.write(f"{icon} **{name}:** {status}")
    
    with col2:
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert to OpenCV format
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Processing button
            if st.button("🚀 Process Image", type="primary"):
                with st.spinner("Processing image with enhanced OCR system..."):
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = {}
                    
                    # Step 1: Multi-engine OCR
                    if use_multi_engine and ENHANCED_OCR_AVAILABLE:
                        status_text.text("Running multi-engine OCR...")
                        progress_bar.progress(20)
                        
                        start_time = time.time()
                        ocr_result = extract_text_enhanced_multi_engine(image)
                        ocr_time = time.time() - start_time
                        
                        results['multi_engine'] = ocr_result
                        results['multi_engine']['processing_time'] = ocr_time
                    
                    progress_bar.progress(40)
                    
                    # Step 2: Precision Layout
                    if enable_precision_layout and PRECISION_LAYOUT_AVAILABLE:
                        status_text.text("Applying precision layout preservation...")
                        progress_bar.progress(60)
                        
                        start_time = time.time()
                        precision_result = extract_text_with_precision_layout(image_np)
                        precision_time = time.time() - start_time
                        
                        results['precision_layout'] = precision_result
                        results['precision_layout']['processing_time'] = precision_time
                    
                    # Step 3: Basic Layout (if enabled)
                    if enable_basic_layout and LAYOUT_AWARE_AVAILABLE:
                        status_text.text("Applying basic layout preservation...")
                        progress_bar.progress(80)
                        
                        start_time = time.time()
                        layout_result = extract_text_with_layout_preservation(image_np, preserve_layout=True)
                        layout_time = time.time() - start_time
                        
                        results['basic_layout'] = layout_result
                        results['basic_layout']['processing_time'] = layout_time
                    
                    progress_bar.progress(100)
                    status_text.text("Processing complete!")
                    
                    # Display results
                    st.success("Processing completed successfully!")
                    
                    # Performance metrics
                    st.subheader("⚡ Performance Metrics")
                    perf_cols = st.columns(len(results))
                    
                    for i, (method, result) in enumerate(results.items()):
                        with perf_cols[i]:
                            processing_time = result.get('processing_time', 0)
                            st.metric(
                                f"{method.replace('_', ' ').title()} Time",
                                f"{processing_time:.2f}s"
                            )
                    
                    # Engine Status
                    if show_engine_details and 'multi_engine' in results:
                        display_engine_status(results['multi_engine'])
                    
                    # Results Tabs
                    result_tabs = []
                    if 'precision_layout' in results:
                        result_tabs.append("Precision Layout")
                    if 'multi_engine' in results:
                        result_tabs.append("Multi-Engine OCR")
                    if 'basic_layout' in results:
                        result_tabs.append("Basic Layout")
                    
                    if result_tabs:
                        tabs = st.tabs(result_tabs)
                        
                        tab_idx = 0
                        
                        # Precision Layout Tab
                        if 'precision_layout' in results:
                            with tabs[tab_idx]:
                                display_precision_results(results['precision_layout'])
                            tab_idx += 1
                        
                        # Multi-Engine OCR Tab
                        if 'multi_engine' in results:
                            with tabs[tab_idx]:
                                st.subheader("🔧 Multi-Engine OCR Results")
                                
                                ocr_result = results['multi_engine']
                                
                                # Main text result
                                st.text_area(
                                    "Extracted Text",
                                    value=ocr_result.get('text', ''),
                                    height=200,
                                    key="multi_engine_text"
                                )
                                
                                # Individual engine results
                                if show_confidence_scores and 'individual_results' in ocr_result:
                                    st.subheader("Individual Engine Results")
                                    
                                    for engine_result in ocr_result['individual_results']:
                                        if engine_result['status'] == 'success':
                                            with st.expander(f"{engine_result['engine']} (Confidence: {engine_result['confidence']:.1%})"):
                                                st.text_area(
                                                    f"{engine_result['engine']} Text",
                                                    value=engine_result['text'],
                                                    height=150,
                                                    key=f"engine_{engine_result['engine']}"
                                                )
                            tab_idx += 1
                        
                        # Basic Layout Tab
                        if 'basic_layout' in results:
                            with tabs[tab_idx]:
                                st.subheader("📄 Basic Layout Results")
                                
                                layout_result = results['basic_layout']
                                
                                if 'error' in layout_result:
                                    st.error(f"Layout preservation error: {layout_result['error']}")
                                else:
                                    # Layout preserved text
                                    st.text_area(
                                        "Layout Preserved Text",
                                        value=layout_result.get('layout_preserved_text', ''),
                                        height=200,
                                        key="basic_layout_text"
                                    )
                                    
                                    # Additional layout info
                                    if layout_result.get('formatting_applied', False):
                                        st.success("✅ Layout formatting successfully applied")
                                    else:
                                        st.warning("⚠️ Basic layout formatting used")
                    
                    # Download Section
                    st.subheader("📥 Download Results")
                    
                    download_col1, download_col2, download_col3 = st.columns(3)
                    
                    with download_col1:
                        if 'precision_layout' in results and results['precision_layout'].get('precision_formatted'):
                            precision_text = results['precision_layout']['precision_formatted']
                            st.markdown(
                                create_download_link(precision_text, "precision_layout.txt"),
                                unsafe_allow_html=True
                            )
                    
                    with download_col2:
                        if 'precision_layout' in results and results['precision_layout'].get('markdown_layout'):
                            markdown_text = results['precision_layout']['markdown_layout']
                            st.markdown(
                                create_download_link(markdown_text, "formatted_text.md", "text/markdown"),
                                unsafe_allow_html=True
                            )
                    
                    with download_col3:
                        if 'multi_engine' in results and results['multi_engine'].get('text'):
                            basic_text = results['multi_engine']['text']
                            st.markdown(
                                create_download_link(basic_text, "extracted_text.txt"),
                                unsafe_allow_html=True
                            )
                    
                    # JSON Export
                    if st.button("📄 Export Full Results (JSON)"):
                        json_data = json.dumps(results, indent=2, default=str)
                        st.markdown(
                            create_download_link(json_data, "ocr_results.json", "application/json"),
                            unsafe_allow_html=True
                        )
        else:
            st.info("👆 Please upload an image to begin OCR processing")
            
            # Feature showcase
            st.markdown("""
            <div class="feature-card">
                <h3>🌟 Features</h3>
                <ul>
                    <li><strong>Multi-Engine OCR:</strong> Combines EasyOCR, PaddleOCR, and Tesseract</li>
                    <li><strong>Precision Layout:</strong> Ultra-accurate layout preservation</li>
                    <li><strong>Format Options:</strong> Plain text, HTML, and Markdown output</li>
                    <li><strong>Visual Preview:</strong> See formatted layout in real-time</li>
                    <li><strong>Export Options:</strong> Download results in multiple formats</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
