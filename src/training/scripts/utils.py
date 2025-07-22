#!/usr/bin/env python3
"""
Utility Functions for Training

File này chứa các utility functions được sử dụng trong quá trình training,
bao gồm logging, directory management, và các helper functions khác.

Author: Highway Guardian Team
Date: 2024
"""

import os
import yaml
import logging
import datetime
from pathlib import Path
import json
import shutil
from typing import Dict, Any, Optional


def setup_logging(name: str, log_level: str = 'INFO') -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create console handler if not exists
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    return logger


def create_experiment_dir(project_path: str, experiment_name: str) -> str:
    """
    Create experiment directory with timestamp.
    
    Args:
        project_path: Base project path
        experiment_name: Experiment name
    
    Returns:
        Path to created experiment directory
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(project_path, f"{experiment_name}_{timestamp}")
    
    # Create directory structure
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'plots'), exist_ok=True)
    
    return experiment_dir


def save_training_config(config: Dict[Any, Any], experiment_dir: str) -> None:
    """
    Save training configuration to experiment directory.
    
    Args:
        config: Training configuration dictionary
        experiment_dir: Experiment directory path
    """
    config_path = os.path.join(experiment_dir, 'training_config.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def load_training_results(experiment_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load training results from experiment directory.
    
    Args:
        experiment_dir: Experiment directory path
    
    Returns:
        Training results dictionary or None if not found
    """
    results_path = os.path.join(experiment_dir, 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_training_results(results: Dict[str, Any], experiment_dir: str) -> None:
    """
    Save training results to experiment directory.
    
    Args:
        results: Training results dictionary
        experiment_dir: Experiment directory path
    """
    results_path = os.path.join(experiment_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def get_latest_experiment(project_path: str, experiment_prefix: str) -> Optional[str]:
    """
    Get the latest experiment directory.
    
    Args:
        project_path: Base project path
        experiment_prefix: Experiment name prefix
    
    Returns:
        Path to latest experiment directory or None if not found
    """
    if not os.path.exists(project_path):
        return None
    
    experiments = []
    for item in os.listdir(project_path):
        if item.startswith(experiment_prefix) and os.path.isdir(os.path.join(project_path, item)):
            experiments.append(item)
    
    if not experiments:
        return None
    
    # Sort by timestamp (assuming format: name_YYYYMMDD_HHMMSS)
    experiments.sort(reverse=True)
    return os.path.join(project_path, experiments[0])


def copy_best_model(experiment_dir: str, destination_dir: str) -> None:
    """
    Copy best model weights to destination directory.
    
    Args:
        experiment_dir: Source experiment directory
        destination_dir: Destination directory
    """
    best_weights_path = os.path.join(experiment_dir, 'weights', 'best.pt')
    if os.path.exists(best_weights_path):
        os.makedirs(destination_dir, exist_ok=True)
        destination_path = os.path.join(destination_dir, 'best.pt')
        shutil.copy2(best_weights_path, destination_path)
        print(f"Best model copied to: {destination_path}")
    else:
        print(f"Best model not found in: {best_weights_path}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate training configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if valid, False otherwise
    """
    required_sections = ['data', 'model', 'training', 'output']
    
    for section in required_sections:
        if section not in config:
            print(f"Missing required section: {section}")
            return False
    
    # Validate data section
    data_required = ['path', 'train', 'val', 'nc', 'names']
    for key in data_required:
        if key not in config['data']:
            print(f"Missing required data key: {key}")
            return False
    
    # Validate model section
    model_required = ['weights']
    for key in model_required:
        if key not in config['model']:
            print(f"Missing required model key: {key}")
            return False
    
    # Validate training section
    training_required = ['epochs', 'batch_size', 'image_size']
    for key in training_required:
        if key not in config['training']:
            print(f"Missing required training key: {key}")
            return False
    
    return True


def calculate_dataset_stats(dataset_path: str) -> Dict[str, Any]:
    """
    Calculate dataset statistics.
    
    Args:
        dataset_path: Path to dataset directory
    
    Returns:
        Dictionary containing dataset statistics
    """
    stats = {
        'train_images': 0,
        'val_images': 0,
        'test_images': 0,
        'total_images': 0
    }
    
    # Count images in each split
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_path, split, 'images')
        if os.path.exists(split_path):
            image_count = len([f for f in os.listdir(split_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            stats[f'{split}_images'] = image_count
            stats['total_images'] += image_count
    
    return stats


def format_training_summary(results: Dict[str, Any]) -> str:
    """
    Format training results into a readable summary.
    
    Args:
        results: Training results dictionary
    
    Returns:
        Formatted summary string
    """
    summary = "\n=== Training Summary ===\n"
    
    if 'metrics/mAP50(B)' in results:
        summary += f"mAP50: {results['metrics/mAP50(B)']:.4f}\n"
    
    if 'metrics/mAP50-95(B)' in results:
        summary += f"mAP50-95: {results['metrics/mAP50-95(B)']:.4f}\n"
    
    if 'metrics/precision(B)' in results:
        summary += f"Precision: {results['metrics/precision(B)']:.4f}\n"
    
    if 'metrics/recall(B)' in results:
        summary += f"Recall: {results['metrics/recall(B)']:.4f}\n"
    
    summary += "========================\n"
    
    return summary


def cleanup_old_experiments(project_path: str, keep_latest: int = 5) -> None:
    """
    Clean up old experiment directories, keeping only the latest N.
    
    Args:
        project_path: Base project path
        keep_latest: Number of latest experiments to keep
    """
    if not os.path.exists(project_path):
        return
    
    experiments = []
    for item in os.listdir(project_path):
        item_path = os.path.join(project_path, item)
        if os.path.isdir(item_path) and '_' in item:
            experiments.append(item)
    
    if len(experiments) <= keep_latest:
        return
    
    # Sort by timestamp and remove old ones
    experiments.sort(reverse=True)
    to_remove = experiments[keep_latest:]
    
    for exp in to_remove:
        exp_path = os.path.join(project_path, exp)
        shutil.rmtree(exp_path)
        print(f"Removed old experiment: {exp}")