# 🐍 Hướng Dẫn Thiết Lập Environment cho Highway Guardian

## 🎯 Tại Sao Cần Environment Management?

Dựa trên phân tích dự án của bạn, việc quản lý thư viện trên máy local gặp khó khăn vì:
- Conflict giữa các phiên bản thư viện
- Docker chiếm nhiều dung lượng ổ đĩa
- Khó reproduce environment giữa các máy khác nhau
- Dependency hell khi cài đặt PyTorch, CUDA, và các ML libraries

## 🏆 Giải Pháp Được Đề Xuất

### 1. **Anaconda/Miniconda (HIGHLY RECOMMENDED)**

#### Tại sao chọn Anaconda?
- ✅ **Quản lý dependencies tốt nhất** cho ML/AI projects
- ✅ **Tự động resolve conflicts** giữa các packages
- ✅ **Built-in CUDA support** - không cần cài CUDA toolkit riêng
- ✅ **Lightweight** hơn Docker đáng kể
- ✅ **Cross-platform** - work trên Windows, Linux, macOS
- ✅ **Conda-forge channel** có hầu hết ML packages

#### Installation Steps:

```bash
# 1. Download Miniconda (lightweight version)
# Từ: https://docs.conda.io/en/latest/miniconda.html
# Chọn Python 3.9+ cho Windows

# 2. Tạo environment cho project
conda create -n highway-guardian python=3.9
conda activate highway-guardian

# 3. Install PyTorch với CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. Install computer vision và ML packages
conda install -c conda-forge opencv pandas matplotlib seaborn pillow

# 5. Install YOLO và monitoring tools
pip install ultralytics wandb tensorboard kaggle

# 6. Install Jupyter cho development
conda install jupyter notebook ipykernel

# 7. Register kernel cho Jupyter
python -m ipykernel install --user --name highway-guardian --display-name "Highway Guardian"
```

#### Environment Management:
```bash
# Activate environment
conda activate highway-guardian

# Deactivate
conda deactivate

# List environments
conda env list

# Export environment (for sharing)
conda env export > environment.yml

# Create from exported file
conda env create -f environment.yml

# Remove environment
conda env remove -n highway-guardian
```

### 2. **Poetry (Modern Alternative)**

#### Ưu điểm:
- ✅ **Modern dependency management**
- ✅ **Lock file** đảm bảo reproducible builds
- ✅ **Virtual environment tự động**
- ✅ **Easy publishing** nếu muốn package project

#### Setup:
```bash
# 1. Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# 2. Initialize project
cd highway-guardian
poetry init

# 3. Add dependencies
poetry add torch torchvision torchaudio --source pytorch
poetry add ultralytics opencv-python pandas matplotlib seaborn
poetry add wandb tensorboard kaggle
poetry add jupyter notebook --group dev

# 4. Install dependencies
poetry install

# 5. Activate shell
poetry shell
```

### 3. **Pipenv (Simple Alternative)**

```bash
# 1. Install pipenv
pip install pipenv

# 2. Create Pipfile
cd highway-guardian
pipenv install torch torchvision torchaudio ultralytics
pipenv install opencv-python pandas matplotlib seaborn
pipenv install wandb tensorboard kaggle

# 3. Activate environment
pipenv shell
```

### 4. **Docker Lightweight Setup**

Nếu vẫn muốn dùng Docker nhưng tiết kiệm dung lượng:

```dockerfile
# Dockerfile.slim
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

CMD ["python", "train.py"]
```

```bash
# Build với multi-stage để giảm size
docker build -f Dockerfile.slim -t highway-guardian:slim .

# Run với volume mount để không mất data
docker run -v $(pwd)/data:/app/data -v $(pwd)/runs:/app/runs highway-guardian:slim
```

## 🎯 Recommendation Matrix

| Criteria | Anaconda | Poetry | Pipenv | Docker |
|----------|----------|--------|--------|---------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **ML/AI Support** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Disk Usage** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Reproducibility** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **CUDA Support** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Windows Support** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## 🚀 Quick Start cho Highway Guardian

### Option A: Anaconda (Recommended)
```bash
# 1. Download và install Miniconda
# 2. Open Anaconda Prompt
conda create -n highway-guardian python=3.9
conda activate highway-guardian

# 3. Install core packages
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install ultralytics wandb tensorboard kaggle
conda install -c conda-forge opencv pandas matplotlib seaborn jupyter

# 4. Test installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('YOLO imported successfully')"

# 5. Start Jupyter
jupyter notebook
```

### Option B: Poetry (For Advanced Users)
```bash
# 1. Install Poetry
# 2. In project directory
poetry init
poetry add torch torchvision torchaudio ultralytics
poetry add opencv-python pandas matplotlib seaborn jupyter
poetry install
poetry shell
```

## 🔧 Troubleshooting Common Issues

### CUDA Issues:
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
# For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# CPU only (fallback):
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### Package Conflicts:
```bash
# Update conda
conda update conda

# Clean cache
conda clean --all

# Force reinstall
conda install --force-reinstall package_name
```

### Jupyter Kernel Issues:
```bash
# Install ipykernel
conda install ipykernel

# Register kernel
python -m ipykernel install --user --name highway-guardian

# List kernels
jupyter kernelspec list

# Remove kernel
jupyter kernelspec uninstall highway-guardian
```

## 📊 Disk Space Comparison

| Solution | Typical Size | Pros | Cons |
|----------|-------------|------|------|
| **Anaconda Full** | ~3-5GB | Complete ML stack | Large download |
| **Miniconda** | ~500MB-2GB | Minimal, add as needed | Manual package selection |
| **Poetry** | ~1-3GB | Modern, fast | Learning curve |
| **Docker** | ~5-10GB | Isolated, reproducible | Large images |
| **Pipenv** | ~1-2GB | Simple, familiar | Basic features |

## 🎯 Final Recommendation

**Cho dự án Highway Guardian của bạn, tôi strongly recommend Miniconda vì:**

1. **Perfect fit cho ML/AI**: Được thiết kế specifically cho data science
2. **CUDA handling**: Tự động manage CUDA dependencies
3. **Disk efficient**: Chỉ install những gì cần
4. **Windows friendly**: Native support, không cần WSL
5. **Easy sharing**: Export/import environment dễ dàng
6. **Community support**: Huge community trong ML/AI

### Next Steps:
1. **Install Miniconda** từ official website
2. **Follow quick start guide** ở trên
3. **Test với existing notebook** để đảm bảo everything works
4. **Export environment** để backup: `conda env export > environment.yml`

Với setup này, bạn sẽ có một environment stable, reproducible và efficient cho việc develop Highway Guardian project! 🚀