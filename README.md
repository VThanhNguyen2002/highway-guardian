# 🛣️ Highway Guardian

## 📌 Project Overview
**Highway Guardian** is an advanced AI-powered traffic sign detection and classification system. Designed for high performance and accuracy, it provides real-time insights into road signs to enhance traffic safety and vehicle navigation systems.

## 🏗️ Architecture
The system employs a robust **Two-Stage** approach for inference:
1. **Detection (YOLOv8)**: Rapidly localizes traffic signs within input images or video streams, extracting the regions of interest (ROIs).
2. **Classification (MobileNetV2)**: A lightweight PyTorch-based Convolutional Neural Network (CNN) that processes the cropped ROIs and accurately classifies them into specific traffic sign categories (supporting 103 distinct classes).

## 🛠️ Tech Stack
- **Backend API**: [FastAPI](https://fastapi.tiangolo.com/)
- **Database**: [SQLAlchemy](https://www.sqlalchemy.org/)
- **Deep Learning**: [PyTorch](https://pytorch.org/) (for MobileNetV2) & Ultralytics (for YOLOv8)
- **Inference UI**: [Streamlit](https://streamlit.io/)
- **Analytics Dashboard**: [Vue.js](https://vuejs.org/)

## 📂 Directory Structure
Below is a map of the key directories in the project:
```text
highway-guardian/
├── backend/          # FastAPI application, endpoints, services, and database schemas
├── frontend/         # Vue.js analytics dashboard and user interface
├── models/           # Pre-trained core weights (YOLOv8 .pt and MobileNetV2 .pth/.h5)
├── streamlit_app/    # Streamlit application for real-time inference and visualization
├── data/             # Datasets, assets, and database storage (.db)
├── scripts/          # Utility scripts and helpers
└── docs/             # Additional project documentation and architecture notes
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js & npm (for Vue.js frontend)
- Docker & Docker Compose (for containerized deployment)

### 1. Native Environment Setup

**Backend (FastAPI):**
```bash
# Navigate to the backend directory
cd backend  # or cd src based on your exact entrypoint

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server using Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
*API running at: http://localhost:8000*

**Streamlit UI:**
```bash
# Navigate to the streamlit_app directory
cd streamlit_app

# Install Streamlit dependencies if needed
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py
```
*Streamlit running at: http://localhost:8501*

**Frontend Dashboard (Vue.js):**
```bash
# Navigate to the frontend directory
cd frontend

# Install Node modules
npm install

# Start the Vue development server
npm run dev
```
*Frontend running at: http://localhost:5173*

### 2. Docker Environment Setup (Recommended)
You can launch the entire ecosystem simultaneously using Docker Compose.

```bash
# Build and start all services in detached mode
docker-compose up --build -d
```
All services will be brought up automatically. To stop the containers, run:
```bash
docker-compose down
```

## 📚 API Documentation
When the FastAPI backend is running, you can explore the endpoints and test the API directly using the built-in interactive documentation:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---
*Designed & Developed by VThanhNguyen2002*
