#!/usr/bin/env python3
"""
OCR Performance Benchmark Tool

This tool helps evaluate the performance of different OCR engines
on a set of test images with known content.
"""

import os
import time
import argparse
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from difflib import SequenceMatcher
from ocr_module import perform_ocr, detect_tables, detect_image_quality
from model_manager import initialize_models, get_performance_stats

def calculate_similarity(a, b):
    """Calculate text similarity between two strings"""
    if not a or not b:
        return 0
    return SequenceMatcher(None, a, b).ratio()

def run_benchmark(image_dir, ground_truth_file=None, engines=None):
    """
    Run OCR benchmark on a directory of images
    
    Args:
        image_dir: Directory containing test images
        ground_truth_file: CSV file with ground truth text for each image
        engines: List of OCR engines to test
    """
    if engines is None:
        engines = ["paddle", "easy", "combined"]
    
    # Initialize models
    print("Initializing OCR models...")
    initialize_models()
    
    # Load ground truth if provided
    ground_truth = {}
    if ground_truth_file and os.path.exists(ground_truth_file):
        try:
            df = pd.read_csv(ground_truth_file)
            ground_truth = dict(zip(df['filename'], df['text']))
        except Exception as e:
            print(f"Error loading ground truth file: {str(e)}")
    
    # Collect image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
        image_files.extend(list(Path(image_dir).glob(f'*{ext}')))
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    # Prepare results storage
    results = []
    
    # Process each image with each engine
    for img_path in image_files:
        img_filename = os.path.basename(img_path)
        print(f"\nProcessing {img_filename}...")
        
        try:
            # Load image
            image = Image.open(img_path)
            
            # Check image quality
            quality_issues = detect_image_quality(image)
            if quality_issues:
                print(f"Image quality issues: {', '.join(quality_issues)}")
            
            # Check for tables
            has_tables = detect_tables(image)
            print(f"Tables detected: {'Yes' if has_tables else 'No'}")
            
            # Ground truth for this image
            gt_text = ground_truth.get(img_filename, "")
            
            # Test each engine
            for engine in engines:
                print(f"Testing {engine} engine...")
                
                # With layout preservation
                start_time = time.time()
                text_formatted = perform_ocr(image, engine=engine, preserve_layout=True)
                format_time = time.time() - start_time
                
                # Without layout preservation
                start_time = time.time()
                text_plain = perform_ocr(image, engine=engine, preserve_layout=False)
                plain_time = time.time() - start_time
                
                # Calculate accuracy if ground truth is available
                accuracy_formatted = calculate_similarity(text_formatted, gt_text) if gt_text else None
                accuracy_plain = calculate_similarity(text_plain, gt_text) if gt_text else None
                
                # Store results
                results.append({
                    'filename': img_filename,
                    'engine': engine,
                    'layout_preserved': True,
                    'processing_time': format_time,
                    'text_length': len(text_formatted),
                    'accuracy': accuracy_formatted,
                    'quality_issues': quality_issues,
                    'has_tables': has_tables
                })
                
                results.append({
                    'filename': img_filename,
                    'engine': engine,
                    'layout_preserved': False,
                    'processing_time': plain_time,
                    'text_length': len(text_plain),
                    'accuracy': accuracy_plain,
                    'quality_issues': quality_issues,
                    'has_tables': has_tables
                })
                
                print(f"  Time (formatted): {format_time:.2f}s, Chars: {len(text_formatted)}")
                print(f"  Time (plain): {plain_time:.2f}s, Chars: {len(text_plain)}")
                if gt_text:
                    print(f"  Accuracy (formatted): {accuracy_formatted:.2%}")
                    print(f"  Accuracy (plain): {accuracy_plain:.2%}")
        
        except Exception as e:
            print(f"Error processing {img_filename}: {str(e)}")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Display performance statistics
    perf_stats = get_performance_stats()
    print("\n=== Performance Statistics ===")
    for engine, stats in perf_stats.items():
        print(f"{engine}: {stats['calls']} calls, avg {stats['avg_time']:.3f}s per call")
    
    return df

def plot_results(results):
    """Plot benchmark results"""
    if results is None or len(results) == 0:
        print("No results to plot")
        return
    
    # Ensure plotting directory exists
    os.makedirs('benchmark_results', exist_ok=True)
    
    # 1. Processing time by engine
    plt.figure(figsize=(10, 6))
    time_by_engine = results.groupby(['engine', 'layout_preserved'])['processing_time'].mean().unstack()
    time_by_engine.plot(kind='bar')
    plt.title('Average Processing Time by Engine')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Engine')
    plt.savefig('benchmark_results/processing_time.png')
    
    # 2. Accuracy by engine (if available)
    if 'accuracy' in results and results['accuracy'].notna().any():
        plt.figure(figsize=(10, 6))
        acc_by_engine = results.groupby(['engine', 'layout_preserved'])['accuracy'].mean().unstack()
        acc_by_engine.plot(kind='bar')
        plt.title('Average Accuracy by Engine')
        plt.ylabel('Accuracy')
        plt.xlabel('Engine')
        plt.savefig('benchmark_results/accuracy.png')
    
    # 3. Text length by engine
    plt.figure(figsize=(10, 6))
    len_by_engine = results.groupby(['engine', 'layout_preserved'])['text_length'].mean().unstack()
    len_by_engine.plot(kind='bar')
    plt.title('Average Text Length by Engine')
    plt.ylabel('Character Count')
    plt.xlabel('Engine')
    plt.savefig('benchmark_results/text_length.png')
    
    # 4. Performance with tables vs without
    if 'has_tables' in results and results['has_tables'].any():
        plt.figure(figsize=(12, 8))
        table_perf = results.groupby(['engine', 'has_tables', 'layout_preserved'])['processing_time'].mean().unstack()
        table_perf.plot(kind='bar')
        plt.title('Processing Time: Tables vs No Tables')
        plt.ylabel('Time (seconds)')
        plt.savefig('benchmark_results/table_performance.png')
    
    # Save detailed results
    results.to_csv('benchmark_results/benchmark_results.csv', index=False)
    
    print("\nResults plots saved to 'benchmark_results' directory")

def main():
    parser = argparse.ArgumentParser(description='OCR Performance Benchmark')
    parser.add_argument('--image-dir', '-i', required=True, help='Directory containing test images')
    parser.add_argument('--ground-truth', '-g', help='CSV file with ground truth text')
    parser.add_argument('--engines', '-e', nargs='+', 
                        choices=['paddle', 'easy', 'combined'],
                        default=['paddle', 'easy', 'combined'],
                        help='OCR engines to benchmark')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.image_dir):
        print(f"Image directory not found: {args.image_dir}")
        return 1
    
    # Run benchmark
    results = run_benchmark(args.image_dir, args.ground_truth, args.engines)
    
    # Plot results
    if results is not None:
        plot_results(results)
    
    return 0

if __name__ == "__main__":
    main()
