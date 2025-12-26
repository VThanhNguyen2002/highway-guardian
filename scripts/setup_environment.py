#!/usr/bin/env python3
"""
One-Line Environment Setup Script
Tối ưu hóa cài đặt môi trường cho Highway Guardian Project

Usage:
    python setup_environment.py --mode [basic|full|gpu] [--force]
    
Modes:
    basic: Cài đặt cơ bản (CPU only)
    full: Cài đặt đầy đủ với GPU support
    gpu: Chỉ cài đặt GPU dependencies

Author: Highway Guardian Team
"""

import os
import sys
import subprocess
import argparse
import platform
from pathlib import Path
import json
import urllib.request

class EnvironmentSetup:
    def __init__(self):
        self.system = platform.system().lower()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.project_root = Path(__file__).parent
        self.requirements_file = self.project_root / "requirements.txt"
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ is required")
            return False
        print(f"✅ Python {self.python_version} detected")
        return True
    
    def check_cuda_availability(self):
        """Check CUDA availability"""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ NVIDIA GPU detected")
                return True
        except FileNotFoundError:
            pass
        print("⚠️  No NVIDIA GPU detected, using CPU mode")
        return False
    
    def install_pytorch(self, gpu_support=True):
        """Install PyTorch with appropriate CUDA support"""
        print("📦 Installing PyTorch...")
        
        if gpu_support and self.check_cuda_availability():
            # Install PyTorch with CUDA 11.8
            cmd = [
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ]
        else:
            # Install CPU-only PyTorch
            cmd = [
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ]
        
        return self.run_command(cmd)
    
    def install_requirements(self, mode="full"):
        """Install requirements based on mode"""
        print(f"📦 Installing requirements ({mode} mode)...")
        
        # Base requirements
        base_packages = [
            "ultralytics>=8.0.0",
            "opencv-python>=4.8.0",
            "Pillow>=9.0.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "pyyaml>=6.0",
            "tqdm>=4.64.0",
            "requests>=2.28.0",
            "scikit-learn>=1.1.0",
            "scipy>=1.9.0"
        ]
        
        # Additional packages for full mode
        full_packages = [
            "wandb>=0.15.0",
            "tensorboard>=2.10.0",
            "albumentations>=1.3.0",
            "plotly>=5.10.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.20.0",
            "streamlit>=1.25.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "kaggle>=1.5.0",
            "python-dotenv>=1.0.0",
            "click>=8.0.0"
        ]
        
        # Development packages
        dev_packages = [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "pre-commit>=2.20.0"
        ]
        
        # Ưu tiên cài từ requirements.txt nếu có
        requirements_path = self.project_root.parent / "src" / "requirements.txt"
        if requirements_path.exists():
            print(f"📄 Found requirements.txt at {requirements_path}")
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
            return self.run_command(cmd)
        else:
            print("⚠️ requirements.txt not found, falling back to manual list...")
            # Fallback to manual list (Code cũ)
            packages = base_packages
            if mode in ["full", "gpu"]:
                packages.extend(full_packages)
            if mode == "full":
                packages.extend(dev_packages)
            
            cmd = [sys.executable, "-m", "pip", "install"] + packages
            return self.run_command(cmd)

    def install_frontend_dependencies(self):
        """Install Frontend Dependencies (Node.js)"""
        print("📦 Installing Frontend dependencies (npm install)...")
        frontend_dir = self.project_root.parent / "frontend"
        
        if not frontend_dir.exists():
            print("❌ Frontend directory not found!")
            return False

        # Check if npm is installed
        if platform.system().lower() == "windows":
            npm_cmd = "npm.cmd"
        else:
            npm_cmd = "npm"

        try:
            subprocess.run([npm_cmd, "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Node.js/npm is not installed. Please install Node.js to run Frontend.")
            return False

        print(f"   Working directory: {frontend_dir}")
        return self.run_command([npm_cmd, "install"], cwd=str(frontend_dir))
    
    # Overriding run_command to support cwd
    def run_command(self, cmd, check=True, cwd=None):
        """Run shell command"""
        try:
            # Nếu cmd[0] là python, giữ nguyên, nếu không thì cứ chạy
            run_cmd = cmd
            
            print(f"   Running: {' '.join(run_cmd) if isinstance(run_cmd, list) else run_cmd}")
            result = subprocess.run(run_cmd, check=check, cwd=cwd, shell=(platform.system().lower()=="windows" and cwd is not None))
            # Note: shell=True might be needed for npm on windows if not using .cmd, but npm.cmd is safer
            
            if result.returncode == 0:
                print("   ✅ Success")
                return True
            else:
                print("   ❌ Failed")
                return False
        except Exception as e:
            print(f"❌ Error running command: {e}")
            return False
    
    def setup_directories(self):
        """Create necessary directories"""
        print("📁 Setting up directories...")
        
        directories = [
            "data/traffic_signs/images/train",
            "data/traffic_signs/images/val",
            "data/traffic_signs/images/test",
            "data/traffic_signs/labels/train",
            "data/traffic_signs/labels/val",
            "data/traffic_signs/labels/test",
            "data/vehicles/images/train",
            "data/vehicles/images/val",
            "data/vehicles/labels/train",
            "data/vehicles/labels/val",
            "models/yolo",
            "models/cnn",
            "outputs/predictions",
            "outputs/visualizations",
            "logs",
            "src/data/runs/detect",
            "src/data/runs/train",
            "src/data/runs/val"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print("✅ Directories created")
        return True
    
    def setup_git_hooks(self):
        """Setup git hooks for development"""
        print("🔧 Setting up git hooks...")
        
        try:
            # Install pre-commit hooks
            cmd = [sys.executable, "-m", "pre_commit", "install"]
            return self.run_command(cmd, check=False)
        except Exception:
            print("⚠️  Pre-commit hooks setup skipped")
            return True
    
    def create_env_file(self):
        """Create .env file template"""
        print("📝 Creating .env file...")
        
        env_content = """# Highway Guardian Environment Variables

# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=highway-guardian

# Kaggle API (for dataset download)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key

# Model paths
MODEL_DIR=models
DATA_DIR=data
OUTPUT_DIR=outputs

# Training settings
DEVICE=0
BATCH_SIZE=16
NUM_WORKERS=8

# API settings (for deployment)
API_HOST=0.0.0.0
API_PORT=8000
"""
        
        env_file = self.project_root / ".env"
        if not env_file.exists():
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(env_content)
            print("✅ .env file created")
        else:
            print("⚠️  .env file already exists")
        
        return True
    
    def download_sample_models(self):
        """Download sample YOLO models"""
        print("📥 Downloading sample models...")
        
        models = {
            "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
        }
        
        models_dir = self.project_root / "models" / "yolo"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, url in models.items():
            model_path = models_dir / model_name
            if not model_path.exists():
                try:
                    print(f"  Downloading {model_name}...")
                    urllib.request.urlretrieve(url, model_path)
                    print(f"  ✅ {model_name} downloaded")
                except Exception as e:
                    print(f"  ⚠️  Failed to download {model_name}: {e}")
        
        return True
    
    def run_command(self, cmd, check=True):
        """Run shell command"""
        try:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
            if result.returncode == 0:
                return True
            else:
                print(f"❌ Command failed: {' '.join(cmd)}")
                print(f"Error: {result.stderr}")
                return False
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {' '.join(cmd)}")
            print(f"Error: {e.stderr}")
            return False
        except FileNotFoundError:
            print(f"❌ Command not found: {cmd[0]}")
            return False
    
    def verify_installation(self):
        """Verify installation"""
        print("🔍 Verifying installation...")
        
        # Test imports
        test_imports = [
            "torch",
            "torchvision",
            "ultralytics",
            "cv2",
            "numpy",
            "pandas",
            "matplotlib",
            "yaml"
        ]
        
        failed_imports = []
        for module in test_imports:
            try:
                __import__(module)
                print(f"  ✅ {module}")
            except ImportError:
                print(f"  ❌ {module}")
                failed_imports.append(module)
        
        if failed_imports:
            print(f"❌ Failed to import: {', '.join(failed_imports)}")
            return False
        
        # Test CUDA if available
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  ✅ CUDA available: {torch.cuda.get_device_name()}")
            else:
                print(f"  ⚠️  CUDA not available")
        except Exception:
            pass
        
        print("✅ Installation verified")
        return True
    
    
    def setup(self, mode="full", force=False):
        """Main setup function"""
        print(f"🚀 Setting up Highway Guardian environment ({mode} mode)...")
        print(f"📍 Project root: {self.project_root}")
        
        # 1. Backend Setup
        print("\n--- BACKEND SETUP ---")
        if not self.check_python_version(): return False
        
        print("📦 Upgrading pip...")
        self.run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        if not self.install_requirements(mode):
            if not force: return False
            
        self.setup_directories()
        self.create_env_file()
        
        # 2. Frontend Setup
        print("\n--- FRONTEND SETUP ---")
        self.install_frontend_dependencies()

        print("\n" + "="*50)
        print("🎉 Setup Completed!")
        return True

def main():
    parser = argparse.ArgumentParser(description='Setup Highway Guardian environment')
    parser.add_argument('--mode', choices=['basic', 'full', 'gpu'], default='full',
                       help='Installation mode')
    parser.add_argument('--force', action='store_true',
                       help='Continue on errors')
    
    args = parser.parse_args()
    
    setup = EnvironmentSetup()
    success = setup.setup(args.mode, args.force)
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()