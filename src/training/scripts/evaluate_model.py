#!/usr/bin/env python3
"""
Model Evaluation Script

Script này thực hiện đánh giá chi tiết các model đã được training,
bao gồm metrics, visualization và so sánh performance.

Usage:
    python evaluate_model.py --model_path ../runs/car_detection/best_model/weights/best.pt --data_path ../../data/datasets/car-detection-dataset
    python evaluate_model.py --model_path ../runs/sign_detection/best_model/weights/best.pt --data_path ../../data/datasets/traffic-signs --detailed

Author: Highway Guardian Team
Date: 2024
"""

import argparse
import os
import sys
import yaml
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ultralytics import YOLO
import torch
from sklearn.metrics import classification_report, confusion_matrix

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, save_training_results


def load_model(model_path):
    """Load YOLO model from path."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = YOLO(model_path)
    return model


def evaluate_on_dataset(model, data_path, split='test'):
    """Evaluate model on specified dataset split."""
    logger = setup_logging('model_evaluation')
    
    # Run validation
    logger.info(f"Evaluating model on {split} set...")
    results = model.val(data=data_path, split=split, save_json=True, plots=True)
    
    return results


def calculate_detailed_metrics(model, data_path, split='test'):
    """Calculate detailed metrics for each class."""
    logger = setup_logging('model_evaluation')
    
    # Get predictions
    results = model.val(data=data_path, split=split)
    
    metrics = {
        'overall': {
            'mAP50': float(results.box.map50),
            'mAP50_95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr)
        },
        'per_class': {}
    }
    
    # Per-class metrics
    if hasattr(results.box, 'ap_class_index') and results.box.ap_class_index is not None:
        for i, class_idx in enumerate(results.box.ap_class_index):
            class_name = model.names[int(class_idx)]
            metrics['per_class'][class_name] = {
                'mAP50': float(results.box.ap50[i]) if i < len(results.box.ap50) else 0.0,
                'mAP50_95': float(results.box.ap[i]) if i < len(results.box.ap) else 0.0
            }
    
    return metrics


def create_performance_report(metrics, model_name, output_dir):
    """Create detailed performance report."""
    report = f"""# Model Performance Report: {model_name}

## Overall Metrics
- **mAP50**: {metrics['overall']['mAP50']:.4f}
- **mAP50-95**: {metrics['overall']['mAP50_95']:.4f}
- **Precision**: {metrics['overall']['precision']:.4f}
- **Recall**: {metrics['overall']['recall']:.4f}

## Per-Class Performance

| Class | mAP50 | mAP50-95 |
|-------|-------|----------|
"""
    
    for class_name, class_metrics in metrics['per_class'].items():
        report += f"| {class_name} | {class_metrics['mAP50']:.4f} | {class_metrics['mAP50_95']:.4f} |\n"
    
    report += f"""

## Analysis

### Best Performing Classes
"""
    
    # Sort classes by mAP50
    sorted_classes = sorted(metrics['per_class'].items(), 
                          key=lambda x: x[1]['mAP50'], reverse=True)
    
    for i, (class_name, class_metrics) in enumerate(sorted_classes[:5]):
        report += f"{i+1}. **{class_name}**: mAP50={class_metrics['mAP50']:.4f}\n"
    
    report += "\n### Classes Needing Improvement\n"
    
    for i, (class_name, class_metrics) in enumerate(sorted_classes[-5:]):
        report += f"{i+1}. **{class_name}**: mAP50={class_metrics['mAP50']:.4f}\n"
    
    # Save report
    report_path = os.path.join(output_dir, f'{model_name}_performance_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report_path


def create_visualization(metrics, model_name, output_dir):
    """Create performance visualization plots."""
    plt.style.use('seaborn-v0_8')
    
    # Per-class mAP50 bar plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    classes = list(metrics['per_class'].keys())
    map50_values = [metrics['per_class'][cls]['mAP50'] for cls in classes]
    map50_95_values = [metrics['per_class'][cls]['mAP50_95'] for cls in classes]
    
    # mAP50 plot
    bars1 = ax1.bar(range(len(classes)), map50_values, color='skyblue', alpha=0.8)
    ax1.set_title(f'{model_name} - mAP50 per Class', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('mAP50')
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, map50_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # mAP50-95 plot
    bars2 = ax2.bar(range(len(classes)), map50_95_values, color='lightcoral', alpha=0.8)
    ax2.set_title(f'{model_name} - mAP50-95 per Class', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('mAP50-95')
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, map50_95_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{model_name}_class_performance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path


def compare_models(model_paths, model_names, data_path, output_dir):
    """Compare multiple models performance."""
    logger = setup_logging('model_comparison')
    
    comparison_data = {}
    
    for model_path, model_name in zip(model_paths, model_names):
        logger.info(f"Evaluating {model_name}...")
        model = load_model(model_path)
        metrics = calculate_detailed_metrics(model, data_path)
        comparison_data[model_name] = metrics
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = list(comparison_data.keys())
    metrics_names = ['mAP50', 'mAP50_95', 'precision', 'recall']
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics_names):
        values = [comparison_data[model]['overall'][metric.replace('_', '-')] for model in models]
        ax.bar(x + i * width, values, width, label=metric, alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comparison data
    comparison_json_path = os.path.join(output_dir, 'model_comparison.json')
    with open(comparison_json_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    return comparison_plot_path, comparison_json_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model weights file')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset YAML file')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--model_name', type=str, default='model',
                       help='Model name for reports')
    parser.add_argument('--detailed', action='store_true',
                       help='Generate detailed analysis')
    parser.add_argument('--compare', nargs='+', type=str,
                       help='Additional model paths for comparison')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger = setup_logging('model_evaluation')
    logger.info(f"Starting model evaluation: {args.model_path}")
    
    try:
        # Load and evaluate main model
        model = load_model(args.model_path)
        
        # Basic evaluation
        logger.info("Running basic evaluation...")
        results = evaluate_on_dataset(model, args.data_path)
        
        if args.detailed:
            # Detailed metrics
            logger.info("Calculating detailed metrics...")
            metrics = calculate_detailed_metrics(model, args.data_path)
            
            # Create performance report
            report_path = create_performance_report(metrics, args.model_name, args.output_dir)
            logger.info(f"Performance report saved: {report_path}")
            
            # Create visualization
            plot_path = create_visualization(metrics, args.model_name, args.output_dir)
            logger.info(f"Performance plot saved: {plot_path}")
            
            # Save metrics as JSON
            metrics_path = os.path.join(args.output_dir, f'{args.model_name}_metrics.json')
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # Model comparison
        if args.compare:
            logger.info("Running model comparison...")
            all_models = [args.model_path] + args.compare
            all_names = [args.model_name] + [f'model_{i+1}' for i in range(len(args.compare))]
            
            comparison_plot, comparison_data = compare_models(
                all_models, all_names, args.data_path, args.output_dir
            )
            logger.info(f"Comparison results saved: {comparison_plot}")
        
        logger.info("Evaluation completed successfully!")
        print(f"\n=== Evaluation Summary ===")
        print(f"Model: {args.model_name}")
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()