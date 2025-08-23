#!/usr/bin/env python3
"""
Training Manager Script
Script quản lý và điều phối các training jobs

Features:
- Chạy multiple training experiments
- Monitor training progress
- Compare results
- Auto hyperparameter tuning
- Resume interrupted training

Usage:
    python scripts/training_manager.py run --config configs/car_det.yaml
    python scripts/training_manager.py monitor --experiment car_detection_20241201_120000
    python scripts/training_manager.py compare --experiments exp1,exp2,exp3
    python scripts/training_manager.py tune --config configs/sign_det_improved.yaml

Author: Highway Guardian Team
"""

import os
import sys
import argparse
import json
import yaml
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import threading
import queue

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Install with: pip install psutil")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class TrainingManager:
    def __init__(self):
        self.project_root = project_root
        self.runs_dir = self.project_root / "src" / "data" / "runs"
        self.configs_dir = self.project_root / "src" / "configs"
        self.scripts_dir = self.project_root / "src" / "training" / "scripts"
        
        # Create directories
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Training queue
        self.training_queue = queue.Queue()
        self.active_trainings = {}
        
    def run_training(self, config_path: str, script_name: str = None, 
                    experiment_name: str = None, **kwargs) -> bool:
        """
        Run a single training experiment
        
        Args:
            config_path: Path to config file
            script_name: Training script name
            experiment_name: Custom experiment name
            **kwargs: Additional arguments
        
        Returns:
            True if training started successfully
        """
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return False
        
        # Load config to determine script
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Determine script based on config or explicit parameter
        if script_name is None:
            if 'sign' in config_path.name.lower():
                script_name = 'train_sign_detection_improved.py'
            else:
                script_name = 'train_car_detection.py'
        
        script_path = self.scripts_dir / script_name
        if not script_path.exists():
            print(f"❌ Training script not found: {script_path}")
            return False
        
        # Generate experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = config_path.stem
            experiment_name = f"{base_name}_{timestamp}"
        
        # Build command
        cmd = [
            sys.executable, str(script_path),
            '--config', str(config_path),
            '--name', experiment_name
        ]
        
        # Add additional arguments
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f'--{key.replace("_", "-")}', str(value)])
        
        print(f"🚀 Starting training: {experiment_name}")
        print(f"📝 Config: {config_path}")
        print(f"🔧 Script: {script_path}")
        print(f"💻 Command: {' '.join(cmd)}")
        
        try:
            # Start training process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.project_root)
            )
            
            # Store process info
            self.active_trainings[experiment_name] = {
                'process': process,
                'start_time': datetime.now(),
                'config_path': str(config_path),
                'script_path': str(script_path),
                'command': cmd
            }
            
            print(f"✅ Training started with PID: {process.pid}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start training: {e}")
            return False
    
    def monitor_training(self, experiment_name: str = None) -> None:
        """
        Monitor training progress
        
        Args:
            experiment_name: Specific experiment to monitor (None for all)
        """
        if experiment_name:
            if experiment_name not in self.active_trainings:
                # Try to find experiment in runs directory
                exp_dir = self.runs_dir / "detect" / experiment_name
                if exp_dir.exists():
                    self._monitor_experiment_files(exp_dir)
                else:
                    print(f"❌ Experiment not found: {experiment_name}")
                return
            
            self._monitor_single_training(experiment_name)
        else:
            self._monitor_all_trainings()
    
    def _monitor_single_training(self, experiment_name: str) -> None:
        """
        Monitor a single training experiment
        """
        training_info = self.active_trainings[experiment_name]
        process = training_info['process']
        
        print(f"📊 Monitoring training: {experiment_name}")
        print(f"🕐 Started: {training_info['start_time']}")
        print(f"🆔 PID: {process.pid}")
        print("="*50)
        
        try:
            # Monitor process output
            while process.poll() is None:
                output = process.stdout.readline()
                if output:
                    print(output.strip())
                time.sleep(0.1)
            
            # Get final output
            stdout, stderr = process.communicate()
            if stdout:
                print(stdout)
            if stderr:
                print(f"Errors: {stderr}")
            
            # Check exit code
            if process.returncode == 0:
                print(f"✅ Training completed successfully: {experiment_name}")
            else:
                print(f"❌ Training failed with code {process.returncode}: {experiment_name}")
            
            # Remove from active trainings
            del self.active_trainings[experiment_name]
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Stopping monitoring for: {experiment_name}")
            print("Training continues in background...")
    
    def _monitor_all_trainings(self) -> None:
        """
        Monitor all active trainings
        """
        if not self.active_trainings:
            print("📭 No active trainings")
            return
        
        print(f"📊 Monitoring {len(self.active_trainings)} active trainings...")
        print("="*70)
        
        try:
            while self.active_trainings:
                for exp_name in list(self.active_trainings.keys()):
                    training_info = self.active_trainings[exp_name]
                    process = training_info['process']
                    
                    if process.poll() is not None:
                        # Training finished
                        if process.returncode == 0:
                            print(f"✅ {exp_name}: Completed successfully")
                        else:
                            print(f"❌ {exp_name}: Failed (code {process.returncode})")
                        
                        del self.active_trainings[exp_name]
                    else:
                        # Training still running
                        elapsed = datetime.now() - training_info['start_time']
                        print(f"🔄 {exp_name}: Running ({elapsed})")
                
                if self.active_trainings:
                    time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopping monitoring...")
            print(f"📝 {len(self.active_trainings)} trainings continue in background")
    
    def _monitor_experiment_files(self, exp_dir: Path) -> None:
        """
        Monitor experiment by reading log files
        """
        print(f"📁 Monitoring experiment directory: {exp_dir}")
        
        # Look for log files
        log_files = list(exp_dir.glob('*.log')) + list(exp_dir.glob('training.log'))
        results_file = exp_dir / 'results.csv'
        
        if log_files:
            print(f"📋 Found log file: {log_files[0]}")
            try:
                with open(log_files[0], 'r', encoding='utf-8') as f:
                    print(f.read())
            except Exception as e:
                print(f"❌ Error reading log: {e}")
        
        if results_file.exists():
            print(f"📊 Results file: {results_file}")
            try:
                import pandas as pd
                df = pd.read_csv(results_file)
                print(df.tail(10))  # Show last 10 rows
            except Exception as e:
                print(f"❌ Error reading results: {e}")
    
    def compare_experiments(self, experiment_names: List[str]) -> None:
        """
        Compare multiple experiments
        
        Args:
            experiment_names: List of experiment names to compare
        """
        print(f"📊 Comparing {len(experiment_names)} experiments...")
        print("="*70)
        
        results = []
        
        for exp_name in experiment_names:
            exp_dir = self.runs_dir / "detect" / exp_name
            if not exp_dir.exists():
                print(f"⚠️  Experiment not found: {exp_name}")
                continue
            
            # Load results
            exp_results = self._load_experiment_results(exp_dir)
            exp_results['name'] = exp_name
            results.append(exp_results)
        
        if not results:
            print("❌ No valid experiments found")
            return
        
        # Display comparison
        self._display_comparison(results)
    
    def _load_experiment_results(self, exp_dir: Path) -> Dict:
        """
        Load experiment results from directory
        """
        results = {
            'path': str(exp_dir),
            'status': 'unknown',
            'metrics': {}
        }
        
        # Check if training completed
        best_weights = exp_dir / 'weights' / 'best.pt'
        if best_weights.exists():
            results['status'] = 'completed'
        
        # Load metrics from results.csv
        results_csv = exp_dir / 'results.csv'
        if results_csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(results_csv)
                if not df.empty:
                    last_row = df.iloc[-1]
                    results['metrics'] = {
                        'epochs': len(df),
                        'final_map50': last_row.get('metrics/mAP50(B)', 0),
                        'final_map50_95': last_row.get('metrics/mAP50-95(B)', 0),
                        'best_map50': df['metrics/mAP50(B)'].max() if 'metrics/mAP50(B)' in df.columns else 0,
                        'best_map50_95': df['metrics/mAP50-95(B)'].max() if 'metrics/mAP50-95(B)' in df.columns else 0
                    }
            except Exception as e:
                print(f"⚠️  Error loading results for {exp_dir.name}: {e}")
        
        return results
    
    def _display_comparison(self, results: List[Dict]) -> None:
        """
        Display experiment comparison
        """
        print(f"{'Experiment':<30} {'Status':<12} {'mAP50':<8} {'mAP50-95':<10} {'Epochs':<8}")
        print("-" * 70)
        
        for result in results:
            name = result['name'][:29]
            status = result['status']
            metrics = result['metrics']
            
            map50 = metrics.get('best_map50', 0)
            map50_95 = metrics.get('best_map50_95', 0)
            epochs = metrics.get('epochs', 0)
            
            print(f"{name:<30} {status:<12} {map50:<8.3f} {map50_95:<10.3f} {epochs:<8}")
        
        # Find best experiment
        best_exp = max(results, key=lambda x: x['metrics'].get('best_map50_95', 0))
        print(f"\n🏆 Best experiment: {best_exp['name']} (mAP50-95: {best_exp['metrics'].get('best_map50_95', 0):.3f})")
    
    def hyperparameter_tuning(self, config_path: str, param_ranges: Dict) -> None:
        """
        Run hyperparameter tuning
        
        Args:
            config_path: Base config file path
            param_ranges: Dictionary of parameter ranges to tune
        """
        print(f"🔧 Starting hyperparameter tuning...")
        print(f"📝 Base config: {config_path}")
        print(f"🎛️  Parameters: {param_ranges}")
        
        # Generate parameter combinations
        combinations = self._generate_param_combinations(param_ranges)
        
        print(f"🔢 Generated {len(combinations)} combinations")
        
        # Run experiments
        for i, params in enumerate(combinations):
            exp_name = f"tune_{i+1:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"\n🚀 Running experiment {i+1}/{len(combinations)}: {exp_name}")
            print(f"📊 Parameters: {params}")
            
            # Create modified config
            modified_config = self._create_modified_config(config_path, params)
            
            # Run training
            success = self.run_training(
                modified_config,
                experiment_name=exp_name,
                **params
            )
            
            if not success:
                print(f"❌ Failed to start experiment {i+1}")
                continue
            
            # Wait for completion or run in parallel
            # For now, run sequentially
            self.monitor_training(exp_name)
    
    def _generate_param_combinations(self, param_ranges: Dict) -> List[Dict]:
        """
        Generate parameter combinations for tuning
        """
        import itertools
        
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _create_modified_config(self, base_config_path: str, params: Dict) -> str:
        """
        Create modified config file with new parameters
        """
        with open(base_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Update config with new parameters
        for key, value in params.items():
            if '.' in key:
                # Nested parameter (e.g., 'training.lr0')
                keys = key.split('.')
                current = config
                for k in keys[:-1]:
                    current = current[k]
                current[keys[-1]] = value
            else:
                # Top-level parameter
                config[key] = value
        
        # Save modified config
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        modified_path = self.configs_dir / f"modified_{timestamp}.yaml"
        
        with open(modified_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        return str(modified_path)
    
    def list_experiments(self) -> None:
        """
        List all experiments
        """
        detect_dir = self.runs_dir / "detect"
        if not detect_dir.exists():
            print("📭 No experiments found")
            return
        
        experiments = [d for d in detect_dir.iterdir() if d.is_dir()]
        experiments.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        print(f"📋 Found {len(experiments)} experiments:")
        print("="*70)
        print(f"{'Name':<40} {'Date':<20} {'Status':<10}")
        print("-" * 70)
        
        for exp_dir in experiments:
            name = exp_dir.name
            mtime = datetime.fromtimestamp(exp_dir.stat().st_mtime)
            date_str = mtime.strftime('%Y-%m-%d %H:%M:%S')
            
            # Check status
            if (exp_dir / 'weights' / 'best.pt').exists():
                status = "✅ Done"
            elif (exp_dir / 'weights' / 'last.pt').exists():
                status = "🔄 Running"
            else:
                status = "❌ Failed"
            
            print(f"{name:<40} {date_str:<20} {status:<10}")

def main():
    parser = argparse.ArgumentParser(description='Training Management Tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run training experiment')
    run_parser.add_argument('--config', required=True, help='Config file path')
    run_parser.add_argument('--script', help='Training script name')
    run_parser.add_argument('--name', help='Experiment name')
    run_parser.add_argument('--epochs', type=int, help='Number of epochs')
    run_parser.add_argument('--batch-size', type=int, help='Batch size')
    run_parser.add_argument('--device', help='Device (GPU ID)')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor training')
    monitor_parser.add_argument('--experiment', help='Experiment name')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare experiments')
    compare_parser.add_argument('--experiments', required=True, 
                               help='Comma-separated experiment names')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List experiments')
    
    # Tune command
    tune_parser = subparsers.add_parser('tune', help='Hyperparameter tuning')
    tune_parser.add_argument('--config', required=True, help='Base config file')
    tune_parser.add_argument('--params', required=True, 
                            help='JSON string of parameter ranges')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = TrainingManager()
    
    if args.command == 'run':
        kwargs = {}
        if args.epochs:
            kwargs['epochs'] = args.epochs
        if args.batch_size:
            kwargs['batch_size'] = args.batch_size
        if args.device:
            kwargs['device'] = args.device
        
        success = manager.run_training(
            args.config, 
            args.script, 
            args.name,
            **kwargs
        )
        sys.exit(0 if success else 1)
    
    elif args.command == 'monitor':
        manager.monitor_training(args.experiment)
    
    elif args.command == 'compare':
        experiments = [exp.strip() for exp in args.experiments.split(',')]
        manager.compare_experiments(experiments)
    
    elif args.command == 'list':
        manager.list_experiments()
    
    elif args.command == 'tune':
        try:
            param_ranges = json.loads(args.params)
            manager.hyperparameter_tuning(args.config, param_ranges)
        except json.JSONDecodeError:
            print("❌ Invalid JSON format for parameters")
            sys.exit(1)

if __name__ == '__main__':
    main()