#!/usr/bin/env python3
"""
Model Validation Script
Script để validate và đánh giá performance của trained models

Features:
- Validate model trên test dataset
- Generate detailed metrics report
- Create confusion matrix và classification report
- Compare multiple models
- Export validation results

Usage:
    python scripts/model_validator.py validate --model runs/detect/exp1/weights/best.pt --data configs/car_det.yaml
    python scripts/model_validator.py compare --models model1.pt,model2.pt --data configs/sign_det.yaml
    python scripts/model_validator.py benchmark --model best.pt --data test_data.yaml

Author: Highway Guardian Team
"""

import os
import sys
import argparse
import json
import yaml
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

try:
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")

class ModelValidator:
    def __init__(self):
        self.project_root = project_root
        self.results_dir = self.project_root / "validation_results"
        self.results_dir.mkdir(exist_ok=True)
        
    def validate_model(self, model_path: str, data_config: str, 
                      save_results: bool = True) -> Dict:
        """
        Validate a single model
        
        Args:
            model_path: Path to model weights
            data_config: Path to data configuration
            save_results: Whether to save results
        
        Returns:
            Validation results dictionary
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics is required for model validation")
        
        model_path = Path(model_path)
        data_config = Path(data_config)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not data_config.exists():
            raise FileNotFoundError(f"Data config not found: {data_config}")
        
        print(f"🔍 Validating model: {model_path.name}")
        print(f"📊 Data config: {data_config.name}")
        
        # Load model
        model = YOLO(str(model_path))
        
        # Run validation
        print("🚀 Running validation...")
        results = model.val(data=str(data_config), verbose=True)
        
        # Extract metrics
        validation_results = {
            'model_path': str(model_path),
            'data_config': str(data_config),
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'map50': float(results.box.map50),
                'map50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'map_per_class': results.box.maps.tolist() if results.box.maps is not None else [],
                'fitness': float(results.fitness) if hasattr(results, 'fitness') else 0.0
            },
            'class_names': model.names,
            'num_classes': len(model.names)
        }
        
        # Add detailed per-class metrics if available
        if hasattr(results.box, 'ap_class_index') and results.box.ap_class_index is not None:
            validation_results['per_class_metrics'] = self._extract_per_class_metrics(results)
        
        print(f"✅ Validation completed")
        print(f"📊 mAP50: {validation_results['metrics']['map50']:.3f}")
        print(f"📊 mAP50-95: {validation_results['metrics']['map50_95']:.3f}")
        print(f"📊 Precision: {validation_results['metrics']['precision']:.3f}")
        print(f"📊 Recall: {validation_results['metrics']['recall']:.3f}")
        
        if save_results:
            self._save_validation_results(validation_results)
        
        return validation_results
    
    def _extract_per_class_metrics(self, results) -> Dict:
        """
        Extract per-class metrics from validation results
        """
        per_class = {}
        
        if hasattr(results.box, 'ap_class_index') and results.box.ap_class_index is not None:
            for i, class_idx in enumerate(results.box.ap_class_index):
                class_name = results.names[int(class_idx)] if hasattr(results, 'names') else f"class_{class_idx}"
                per_class[class_name] = {
                    'ap50': float(results.box.ap50[i]) if i < len(results.box.ap50) else 0.0,
                    'ap50_95': float(results.box.ap[i]) if i < len(results.box.ap) else 0.0,
                }
        
        return per_class
    
    def compare_models(self, model_paths: List[str], data_config: str) -> Dict:
        """
        Compare multiple models
        
        Args:
            model_paths: List of model paths
            data_config: Path to data configuration
        
        Returns:
            Comparison results
        """
        print(f"🔄 Comparing {len(model_paths)} models...")
        
        results = []
        for model_path in model_paths:
            try:
                result = self.validate_model(model_path, data_config, save_results=False)
                results.append(result)
            except Exception as e:
                print(f"❌ Error validating {model_path}: {e}")
                continue
        
        if not results:
            print("❌ No valid models to compare")
            return {}
        
        # Create comparison
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'data_config': data_config,
            'models': results,
            'summary': self._create_comparison_summary(results)
        }
        
        # Display comparison
        self._display_comparison(comparison)
        
        # Save comparison
        self._save_comparison_results(comparison)
        
        return comparison
    
    def _create_comparison_summary(self, results: List[Dict]) -> Dict:
        """
        Create summary of model comparison
        """
        summary = {
            'best_map50': {'value': 0, 'model': ''},
            'best_map50_95': {'value': 0, 'model': ''},
            'best_precision': {'value': 0, 'model': ''},
            'best_recall': {'value': 0, 'model': ''},
            'metrics_table': []
        }
        
        for result in results:
            model_name = Path(result['model_path']).name
            metrics = result['metrics']
            
            # Update best metrics
            if metrics['map50'] > summary['best_map50']['value']:
                summary['best_map50'] = {'value': metrics['map50'], 'model': model_name}
            
            if metrics['map50_95'] > summary['best_map50_95']['value']:
                summary['best_map50_95'] = {'value': metrics['map50_95'], 'model': model_name}
            
            if metrics['precision'] > summary['best_precision']['value']:
                summary['best_precision'] = {'value': metrics['precision'], 'model': model_name}
            
            if metrics['recall'] > summary['best_recall']['value']:
                summary['best_recall'] = {'value': metrics['recall'], 'model': model_name}
            
            # Add to metrics table
            summary['metrics_table'].append({
                'model': model_name,
                'map50': metrics['map50'],
                'map50_95': metrics['map50_95'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'fitness': metrics.get('fitness', 0.0)
            })
        
        return summary
    
    def _display_comparison(self, comparison: Dict) -> None:
        """
        Display model comparison results
        """
        print("\n" + "="*80)
        print("📊 MODEL COMPARISON RESULTS")
        print("="*80)
        
        summary = comparison['summary']
        
        # Display table
        print(f"\n{'Model':<30} {'mAP50':<8} {'mAP50-95':<10} {'Precision':<10} {'Recall':<8} {'Fitness':<8}")
        print("-" * 80)
        
        for row in summary['metrics_table']:
            print(f"{row['model']:<30} {row['map50']:<8.3f} {row['map50_95']:<10.3f} "
                  f"{row['precision']:<10.3f} {row['recall']:<8.3f} {row['fitness']:<8.3f}")
        
        # Display best models
        print("\n🏆 BEST MODELS:")
        print(f"  📈 Best mAP50: {summary['best_map50']['model']} ({summary['best_map50']['value']:.3f})")
        print(f"  📈 Best mAP50-95: {summary['best_map50_95']['model']} ({summary['best_map50_95']['value']:.3f})")
        print(f"  🎯 Best Precision: {summary['best_precision']['model']} ({summary['best_precision']['value']:.3f})")
        print(f"  🔍 Best Recall: {summary['best_recall']['model']} ({summary['best_recall']['value']:.3f})")
    
    def benchmark_model(self, model_path: str, data_config: str, 
                       num_runs: int = 3) -> Dict:
        """
        Benchmark model performance with multiple runs
        
        Args:
            model_path: Path to model weights
            data_config: Path to data configuration
            num_runs: Number of benchmark runs
        
        Returns:
            Benchmark results
        """
        print(f"⏱️  Benchmarking model: {Path(model_path).name}")
        print(f"🔄 Running {num_runs} validation runs...")
        
        results = []
        for i in range(num_runs):
            print(f"\n📊 Run {i+1}/{num_runs}")
            result = self.validate_model(model_path, data_config, save_results=False)
            results.append(result['metrics'])
        
        # Calculate statistics
        metrics_stats = self._calculate_benchmark_stats(results)
        
        benchmark_results = {
            'model_path': model_path,
            'data_config': data_config,
            'num_runs': num_runs,
            'timestamp': datetime.now().isoformat(),
            'individual_runs': results,
            'statistics': metrics_stats
        }
        
        # Display results
        self._display_benchmark_results(benchmark_results)
        
        # Save results
        self._save_benchmark_results(benchmark_results)
        
        return benchmark_results
    
    def _calculate_benchmark_stats(self, results: List[Dict]) -> Dict:
        """
        Calculate statistics from multiple benchmark runs
        """
        metrics = ['map50', 'map50_95', 'precision', 'recall', 'fitness']
        stats = {}
        
        for metric in metrics:
            values = [result.get(metric, 0.0) for result in results]
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return stats
    
    def _display_benchmark_results(self, results: Dict) -> None:
        """
        Display benchmark results
        """
        print("\n" + "="*60)
        print("⏱️  BENCHMARK RESULTS")
        print("="*60)
        
        stats = results['statistics']
        
        print(f"\n{'Metric':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Median':<8}")
        print("-" * 60)
        
        for metric, values in stats.items():
            print(f"{metric:<12} {values['mean']:<8.3f} {values['std']:<8.3f} "
                  f"{values['min']:<8.3f} {values['max']:<8.3f} {values['median']:<8.3f}")
        
        print(f"\n📊 Model: {Path(results['model_path']).name}")
        print(f"🔄 Runs: {results['num_runs']}")
        print(f"📈 Stability (lower std is better):")
        print(f"  - mAP50: ±{stats['map50']['std']:.4f}")
        print(f"  - mAP50-95: ±{stats['map50_95']['std']:.4f}")
    
    def create_validation_report(self, model_path: str, data_config: str, 
                               output_dir: str = None) -> str:
        """
        Create comprehensive validation report
        
        Args:
            model_path: Path to model weights
            data_config: Path to data configuration
            output_dir: Output directory for report
        
        Returns:
            Path to generated report
        """
        if output_dir is None:
            output_dir = self.results_dir / "reports"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📝 Creating validation report...")
        
        # Run validation
        results = self.validate_model(model_path, data_config, save_results=False)
        
        # Generate report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = Path(model_path).stem
        report_path = output_dir / f"validation_report_{model_name}_{timestamp}.html"
        
        html_content = self._generate_html_report(results)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Report saved: {report_path}")
        return str(report_path)
    
    def _generate_html_report(self, results: Dict) -> str:
        """
        Generate HTML validation report
        """
        model_name = Path(results['model_path']).name
        timestamp = results['timestamp']
        metrics = results['metrics']
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Validation Report - {model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        .metric-label {{ font-size: 14px; color: #666; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .good {{ color: #4CAF50; }}
        .warning {{ color: #FF9800; }}
        .poor {{ color: #F44336; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Model Validation Report</h1>
        <p><strong>Model:</strong> {model_name}</p>
        <p><strong>Validation Date:</strong> {timestamp}</p>
        <p><strong>Data Config:</strong> {results['data_config']}</p>
    </div>
    
    <h2>📊 Overall Metrics</h2>
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value">{metrics['map50']:.3f}</div>
            <div class="metric-label">mAP50</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['map50_95']:.3f}</div>
            <div class="metric-label">mAP50-95</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['precision']:.3f}</div>
            <div class="metric-label">Precision</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['recall']:.3f}</div>
            <div class="metric-label">Recall</div>
        </div>
    </div>
    
    <h2>🎯 Performance Analysis</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Assessment</th>
            <th>Recommendation</th>
        </tr>
"""
        
        # Add performance analysis
        assessments = [
            ('mAP50', metrics['map50'], self._assess_metric(metrics['map50'], 'map50')),
            ('mAP50-95', metrics['map50_95'], self._assess_metric(metrics['map50_95'], 'map50_95')),
            ('Precision', metrics['precision'], self._assess_metric(metrics['precision'], 'precision')),
            ('Recall', metrics['recall'], self._assess_metric(metrics['recall'], 'recall'))
        ]
        
        for metric_name, value, assessment in assessments:
            color_class = 'good' if assessment['level'] == 'good' else 'warning' if assessment['level'] == 'warning' else 'poor'
            html += f"""
        <tr>
            <td>{metric_name}</td>
            <td>{value:.3f}</td>
            <td class="{color_class}">{assessment['status']}</td>
            <td>{assessment['recommendation']}</td>
        </tr>
"""
        
        html += """
    </table>
    
    <h2>📋 Model Information</h2>
    <table>
        <tr><th>Property</th><th>Value</th></tr>
        <tr><td>Number of Classes</td><td>{num_classes}</td></tr>
        <tr><td>Class Names</td><td>{class_names}</td></tr>
        <tr><td>Model Path</td><td>{model_path}</td></tr>
    </table>
    
    <h2>💡 Recommendations</h2>
    <ul>
""".format(
            num_classes=results['num_classes'],
            class_names=', '.join(results['class_names'].values()) if isinstance(results['class_names'], dict) else str(results['class_names']),
            model_path=results['model_path']
        )
        
        # Add recommendations
        recommendations = self._generate_recommendations(metrics)
        for rec in recommendations:
            html += f"<li>{rec}</li>\n"
        
        html += """
    </ul>
    
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px;">
        Generated by Highway Guardian Model Validator
    </footer>
</body>
</html>
"""
        
        return html
    
    def _assess_metric(self, value: float, metric_type: str) -> Dict:
        """
        Assess metric performance
        """
        thresholds = {
            'map50': {'good': 0.7, 'warning': 0.5},
            'map50_95': {'good': 0.5, 'warning': 0.3},
            'precision': {'good': 0.8, 'warning': 0.6},
            'recall': {'good': 0.8, 'warning': 0.6}
        }
        
        thresh = thresholds.get(metric_type, {'good': 0.8, 'warning': 0.6})
        
        if value >= thresh['good']:
            return {
                'level': 'good',
                'status': '✅ Excellent',
                'recommendation': 'Performance is good. Consider fine-tuning for specific use cases.'
            }
        elif value >= thresh['warning']:
            return {
                'level': 'warning',
                'status': '⚠️ Moderate',
                'recommendation': 'Consider more training data, data augmentation, or hyperparameter tuning.'
            }
        else:
            return {
                'level': 'poor',
                'status': '❌ Poor',
                'recommendation': 'Significant improvement needed. Review data quality, model architecture, and training process.'
            }
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """
        Generate improvement recommendations
        """
        recommendations = []
        
        if metrics['map50_95'] < 0.3:
            recommendations.append("Consider using a larger model (YOLOv8m or YOLOv8l) for better accuracy")
            recommendations.append("Increase training epochs and use early stopping")
            recommendations.append("Review and improve data quality and annotations")
        
        if metrics['precision'] < 0.6:
            recommendations.append("High false positive rate. Consider adjusting confidence threshold")
            recommendations.append("Review data for mislabeled examples")
        
        if metrics['recall'] < 0.6:
            recommendations.append("High false negative rate. Consider data augmentation")
            recommendations.append("Ensure balanced dataset across all classes")
        
        if abs(metrics['precision'] - metrics['recall']) > 0.2:
            recommendations.append("Imbalanced precision/recall. Consider weighted loss or focal loss")
        
        if not recommendations:
            recommendations.append("Model performance is good. Consider deployment optimization")
            recommendations.append("Test on additional validation datasets for robustness")
        
        return recommendations
    
    def _save_validation_results(self, results: Dict) -> None:
        """
        Save validation results to file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = Path(results['model_path']).stem
        filename = f"validation_{model_name}_{timestamp}.json"
        
        filepath = self.results_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved: {filepath}")
    
    def _save_comparison_results(self, comparison: Dict) -> None:
        """
        Save comparison results to file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"comparison_{timestamp}.json"
        
        filepath = self.results_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Comparison saved: {filepath}")
    
    def _save_benchmark_results(self, results: Dict) -> None:
        """
        Save benchmark results to file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = Path(results['model_path']).stem
        filename = f"benchmark_{model_name}_{timestamp}.json"
        
        filepath = self.results_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Benchmark saved: {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Model Validation Tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate single model')
    validate_parser.add_argument('--model', required=True, help='Model weights path')
    validate_parser.add_argument('--data', required=True, help='Data config path')
    validate_parser.add_argument('--no-save', action='store_true', help='Don\'t save results')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--models', required=True, help='Comma-separated model paths')
    compare_parser.add_argument('--data', required=True, help='Data config path')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark model performance')
    benchmark_parser.add_argument('--model', required=True, help='Model weights path')
    benchmark_parser.add_argument('--data', required=True, help='Data config path')
    benchmark_parser.add_argument('--runs', type=int, default=3, help='Number of runs')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate validation report')
    report_parser.add_argument('--model', required=True, help='Model weights path')
    report_parser.add_argument('--data', required=True, help='Data config path')
    report_parser.add_argument('--output', help='Output directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    validator = ModelValidator()
    
    try:
        if args.command == 'validate':
            validator.validate_model(
                args.model, 
                args.data, 
                save_results=not args.no_save
            )
        
        elif args.command == 'compare':
            models = [model.strip() for model in args.models.split(',')]
            validator.compare_models(models, args.data)
        
        elif args.command == 'benchmark':
            validator.benchmark_model(args.model, args.data, args.runs)
        
        elif args.command == 'report':
            report_path = validator.create_validation_report(
                args.model, 
                args.data, 
                args.output
            )
            print(f"\n📄 Open report: {report_path}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()