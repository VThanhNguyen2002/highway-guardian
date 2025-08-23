#!/usr/bin/env python3
"""
Dataset Management Script
Script hỗ trợ quản lý dataset cho Highway Guardian

Features:
- Download datasets from Kaggle
- Validate dataset structure
- Split datasets
- Generate statistics
- Convert annotations

Usage:
    python scripts/dataset_manager.py download --dataset roboflow/traffic-signs
    python scripts/dataset_manager.py validate --path data/traffic_signs
    python scripts/dataset_manager.py split --path data/raw --output data/processed
    python scripts/dataset_manager.py stats --path data/traffic_signs

Author: Highway Guardian Team
"""

import os
import sys
import argparse
import shutil
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import random
from collections import Counter, defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import kaggle
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("Warning: Kaggle API not available. Install with: pip install kaggle")

try:
    import cv2
    import numpy as np
    from PIL import Image
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")

class DatasetManager:
    def __init__(self):
        self.project_root = project_root
        self.data_dir = self.project_root / "data"
        self.data_dir.mkdir(exist_ok=True)
    
    def download_kaggle_dataset(self, dataset_name: str, output_path: str = None) -> bool:
        """
        Download dataset from Kaggle
        
        Args:
            dataset_name: Kaggle dataset name (e.g., 'username/dataset-name')
            output_path: Output directory path
        
        Returns:
            True if successful
        """
        if not KAGGLE_AVAILABLE:
            print("❌ Kaggle API not available")
            return False
        
        if output_path is None:
            output_path = self.data_dir / "raw" / dataset_name.split('/')[-1]
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"📥 Downloading dataset: {dataset_name}")
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=str(output_path), 
                unzip=True
            )
            print(f"✅ Dataset downloaded to: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to download dataset: {e}")
            return False
    
    def validate_yolo_dataset(self, dataset_path: str) -> Dict:
        """
        Validate YOLO dataset structure
        
        Args:
            dataset_path: Path to dataset
        
        Returns:
            Validation results dictionary
        """
        dataset_path = Path(dataset_path)
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        print(f"🔍 Validating dataset: {dataset_path}")
        
        # Check basic structure
        required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
        for dir_name in required_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                results['errors'].append(f"Missing directory: {dir_name}")
                results['valid'] = False
        
        if not results['valid']:
            return results
        
        # Check data.yaml
        data_yaml = dataset_path / 'data.yaml'
        if not data_yaml.exists():
            results['warnings'].append("Missing data.yaml file")
        else:
            try:
                with open(data_yaml, 'r') as f:
                    data_config = yaml.safe_load(f)
                results['stats']['classes'] = data_config.get('nc', 0)
                results['stats']['class_names'] = data_config.get('names', [])
            except Exception as e:
                results['errors'].append(f"Invalid data.yaml: {e}")
        
        # Count files
        for split in ['train', 'val']:
            images_dir = dataset_path / 'images' / split
            labels_dir = dataset_path / 'labels' / split
            
            if images_dir.exists() and labels_dir.exists():
                image_files = list(images_dir.glob('*'))
                label_files = list(labels_dir.glob('*.txt'))
                
                results['stats'][f'{split}_images'] = len(image_files)
                results['stats'][f'{split}_labels'] = len(label_files)
                
                # Check if images have corresponding labels
                image_stems = {f.stem for f in image_files}
                label_stems = {f.stem for f in label_files}
                
                missing_labels = image_stems - label_stems
                missing_images = label_stems - image_stems
                
                if missing_labels:
                    results['warnings'].append(
                        f"{split}: {len(missing_labels)} images without labels"
                    )
                
                if missing_images:
                    results['warnings'].append(
                        f"{split}: {len(missing_images)} labels without images"
                    )
        
        # Analyze class distribution
        if CV2_AVAILABLE:
            class_counts = self._analyze_class_distribution(dataset_path)
            results['stats']['class_distribution'] = class_counts
        
        return results
    
    def _analyze_class_distribution(self, dataset_path: Path) -> Dict:
        """
        Analyze class distribution in dataset
        """
        class_counts = Counter()
        
        for split in ['train', 'val']:
            labels_dir = dataset_path / 'labels' / split
            if labels_dir.exists():
                for label_file in labels_dir.glob('*.txt'):
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                if line.strip():
                                    class_id = int(line.split()[0])
                                    class_counts[class_id] += 1
                    except Exception:
                        continue
        
        return dict(class_counts)
    
    def split_dataset(self, input_path: str, output_path: str, 
                     train_ratio: float = 0.8, val_ratio: float = 0.15, 
                     test_ratio: float = 0.05) -> bool:
        """
        Split dataset into train/val/test sets
        
        Args:
            input_path: Input dataset path
            output_path: Output dataset path
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
        
        Returns:
            True if successful
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            print(f"❌ Input path does not exist: {input_path}")
            return False
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            print("❌ Ratios must sum to 1.0")
            return False
        
        print(f"📂 Splitting dataset: {input_path} -> {output_path}")
        print(f"   Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'labels']:
                (output_path / subdir / split).mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'**/*{ext}'))
            image_files.extend(input_path.glob(f'**/*{ext.upper()}'))
        
        if not image_files:
            print("❌ No image files found")
            return False
        
        # Shuffle files
        random.shuffle(image_files)
        
        # Calculate split indices
        total_files = len(image_files)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)
        
        splits = {
            'train': image_files[:train_end],
            'val': image_files[train_end:val_end],
            'test': image_files[val_end:]
        }
        
        # Copy files
        for split_name, files in splits.items():
            print(f"   Copying {len(files)} files to {split_name}...")
            
            for img_file in files:
                # Copy image
                dst_img = output_path / 'images' / split_name / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # Copy corresponding label if exists
                label_file = img_file.with_suffix('.txt')
                if label_file.exists():
                    dst_label = output_path / 'labels' / split_name / label_file.name
                    shutil.copy2(label_file, dst_label)
        
        # Create data.yaml
        self._create_data_yaml(output_path)
        
        print("✅ Dataset split completed")
        return True
    
    def _create_data_yaml(self, dataset_path: Path, class_names: List[str] = None):
        """
        Create data.yaml file for YOLO
        """
        if class_names is None:
            # Try to infer class names from existing labels
            class_names = self._infer_class_names(dataset_path)
        
        data_config = {
            'path': str(dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_names),
            'names': class_names
        }
        
        data_yaml_path = dataset_path / 'data.yaml'
        with open(data_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
    
    def _infer_class_names(self, dataset_path: Path) -> List[str]:
        """
        Infer class names from label files
        """
        class_ids = set()
        
        for split in ['train', 'val']:
            labels_dir = dataset_path / 'labels' / split
            if labels_dir.exists():
                for label_file in labels_dir.glob('*.txt'):
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                if line.strip():
                                    class_id = int(line.split()[0])
                                    class_ids.add(class_id)
                    except Exception:
                        continue
        
        # Generate generic class names
        max_class = max(class_ids) if class_ids else 0
        return [f'class_{i}' for i in range(max_class + 1)]
    
    def generate_stats(self, dataset_path: str) -> Dict:
        """
        Generate comprehensive dataset statistics
        
        Args:
            dataset_path: Path to dataset
        
        Returns:
            Statistics dictionary
        """
        dataset_path = Path(dataset_path)
        stats = {
            'dataset_path': str(dataset_path),
            'splits': {},
            'class_distribution': {},
            'image_stats': {},
            'annotation_stats': {}
        }
        
        print(f"📊 Generating statistics for: {dataset_path}")
        
        # Analyze each split
        for split in ['train', 'val', 'test']:
            images_dir = dataset_path / 'images' / split
            labels_dir = dataset_path / 'labels' / split
            
            if images_dir.exists():
                split_stats = self._analyze_split(images_dir, labels_dir)
                stats['splits'][split] = split_stats
        
        return stats
    
    def _analyze_split(self, images_dir: Path, labels_dir: Path) -> Dict:
        """
        Analyze a single split (train/val/test)
        """
        split_stats = {
            'image_count': 0,
            'label_count': 0,
            'class_distribution': Counter(),
            'image_sizes': [],
            'bbox_count': 0
        }
        
        if not images_dir.exists():
            return split_stats
        
        # Count images
        image_files = list(images_dir.glob('*'))
        split_stats['image_count'] = len(image_files)
        
        # Analyze images and labels
        if CV2_AVAILABLE:
            for img_file in image_files:
                try:
                    # Get image size
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        h, w = img.shape[:2]
                        split_stats['image_sizes'].append((w, h))
                    
                    # Analyze corresponding label
                    label_file = labels_dir / f"{img_file.stem}.txt"
                    if label_file.exists():
                        split_stats['label_count'] += 1
                        
                        with open(label_file, 'r') as f:
                            for line in f:
                                if line.strip():
                                    parts = line.split()
                                    if len(parts) >= 5:
                                        class_id = int(parts[0])
                                        split_stats['class_distribution'][class_id] += 1
                                        split_stats['bbox_count'] += 1
                except Exception:
                    continue
        
        return split_stats

def main():
    parser = argparse.ArgumentParser(description='Dataset Management Tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download dataset from Kaggle')
    download_parser.add_argument('--dataset', required=True, help='Kaggle dataset name')
    download_parser.add_argument('--output', help='Output directory')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate dataset structure')
    validate_parser.add_argument('--path', required=True, help='Dataset path')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Split dataset')
    split_parser.add_argument('--path', required=True, help='Input dataset path')
    split_parser.add_argument('--output', required=True, help='Output dataset path')
    split_parser.add_argument('--train', type=float, default=0.8, help='Train ratio')
    split_parser.add_argument('--val', type=float, default=0.15, help='Validation ratio')
    split_parser.add_argument('--test', type=float, default=0.05, help='Test ratio')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Generate dataset statistics')
    stats_parser.add_argument('--path', required=True, help='Dataset path')
    stats_parser.add_argument('--output', help='Output JSON file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = DatasetManager()
    
    if args.command == 'download':
        success = manager.download_kaggle_dataset(args.dataset, args.output)
        sys.exit(0 if success else 1)
    
    elif args.command == 'validate':
        results = manager.validate_yolo_dataset(args.path)
        
        print("\n📋 Validation Results:")
        print(f"Valid: {'✅' if results['valid'] else '❌'}")
        
        if results['errors']:
            print("\n❌ Errors:")
            for error in results['errors']:
                print(f"  - {error}")
        
        if results['warnings']:
            print("\n⚠️  Warnings:")
            for warning in results['warnings']:
                print(f"  - {warning}")
        
        if results['stats']:
            print("\n📊 Statistics:")
            for key, value in results['stats'].items():
                print(f"  {key}: {value}")
        
        sys.exit(0 if results['valid'] else 1)
    
    elif args.command == 'split':
        success = manager.split_dataset(
            args.path, args.output, 
            args.train, args.val, args.test
        )
        sys.exit(0 if success else 1)
    
    elif args.command == 'stats':
        stats = manager.generate_stats(args.path)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"📊 Statistics saved to: {args.output}")
        else:
            print("\n📊 Dataset Statistics:")
            print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()