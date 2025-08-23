#!/bin/bash
# Quick Setup Script for Highway Guardian (Linux/Mac)
# One-line setup for Unix-based systems

set -e  # Exit on any error

echo "========================================"
echo "Highway Guardian - Quick Setup"
echo "========================================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH"
    print_error "Please install Python 3.8+ from your package manager"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "  macOS: brew install python3"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
print_status "Python detected: $PYTHON_VERSION"

# Check if version is 3.8+
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_status "Python version is compatible"
else
    print_error "Python 3.8+ is required, found $PYTHON_VERSION"
    exit 1
fi

echo

# Ask for installation mode
echo "Select installation mode:"
echo "1. Basic (CPU only)"
echo "2. Full (GPU support + all features)"
echo "3. GPU only (just GPU dependencies)"
echo
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        mode="basic"
        ;;
    2)
        mode="full"
        ;;
    3)
        mode="gpu"
        ;;
    *)
        print_warning "Invalid choice, using full mode"
        mode="full"
        ;;
esac

echo
print_status "Installing in $mode mode..."
echo

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_status "Virtual environment detected: $VIRTUAL_ENV"
else
    print_warning "No virtual environment detected"
    read -p "Do you want to create a virtual environment? (y/n): " create_venv
    
    if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
        print_status "Creating virtual environment..."
        python3 -m venv highway_guardian_env
        source highway_guardian_env/bin/activate
        print_status "Virtual environment activated"
    fi
fi

# Make the Python script executable
chmod +x setup_environment.py

# Run the Python setup script
if python3 setup_environment.py --mode $mode; then
    echo
    print_status "========================================"
    print_status "Setup completed successfully!"
    print_status "========================================"
    echo
    echo "You can now:"
    echo "1. Train car detection: python3 src/training/scripts/train_car_detection.py --config src/configs/car_detection_config.yaml"
    echo "2. Train improved sign detection: python3 src/training/scripts/train_sign_detection_improved.py --config src/configs/sign_det_improved.yaml"
    echo "3. Setup Docker: docker-compose up --build"
    echo
    
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo "Note: Remember to activate your virtual environment before running:"
        echo "  source highway_guardian_env/bin/activate"
        echo
    fi
else
    echo
    print_error "Setup failed! Check the error messages above."
    exit 1
fi