# Highway Guardian 🛡️

> AI-powered traffic sign detection and classification system using YOLOv8, CNN, FastAPI and Vue.js

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Vue.js](https://img.shields.io/badge/Vue.js-3.5+-brightgreen.svg)](https://vuejs.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Giới thiệu

Highway Guardian là hệ thống nhận diện biển báo giao thông thông minh với **2 chế độ detection**:

### 🚀 YOLO Only (Fast)
- Phát hiện và phân loại trực tiếp
- Real-time performance (~30-60 FPS)
- Phù hợp cho demo và testing

### 🎯 YOLO + CNN (Accurate)
- **Stage 1**: YOLO phát hiện và crop biển báo
- **Stage 2**: CNN phân loại chi tiết
- Độ chính xác cao hơn (~10-20 FPS)
- Phù hợp cho production

## ✨ Tính năng

- ✅ **Upload Image Detection**: Tải ảnh lên và nhận diện
- ✅ **Real-time Camera Detection**: Nhận diện trực tiếp từ webcam
- ✅ **70+ Vietnamese Signs**: Hỗ trợ 70+ biển báo tiếng Việt
- ✅ **Model Caching**: Load model nhanh hơn 95%
- ✅ **Beautiful UI**: Giao diện hiện đại với gradient colors
- ✅ **FPS Counter**: Theo dõi performance real-time
- ✅ **Firebase Auth**: Đăng nhập an toàn với popup notifications

## 🏗️ Kiến trúc

```
Highway Guardian
├── Backend (FastAPI)
│   ├── YOLO Detection (YOLOv8)
│   ├── CNN Classification (MobileNetV2)
│   ├── 2-Stage Pipeline
│   └── Model Caching (LRU)
│
└── Frontend (Vue.js 3)
    ├── Login Page (Firebase Auth)
    ├── Detect Page (Upload)
    ├── Camera Page (Real-time)
    ├── History Page
    └── Map Page
```

## 📁 Cấu trúc Project

```
highway-guardian/
├── src/                          # Backend
│   ├── main.py                   # FastAPI app
│   ├── config/                   # Configuration
│   ├── services/                 # Business logic
│   └── utils/                    # Utilities
│
├── frontend/                     # Frontend
│   ├── src/
│   │   ├── components/          # Vue components
│   │   ├── views/               # Pages
│   │   ├── stores/              # Pinia stores
│   │   └── router/              # Vue Router
│   └── public/
│
├── models/                       # ML Models
│   ├── yolo/                    # YOLO models
│   └── cnn/                     # CNN models
│
└── docs/                        # Documentation
```

## 📋 Yêu cầu

- **Python 3.8+**: Backend API
- **Node.js 18+**: Frontend development
- **Docker** (optional): Production deployment
- **GPU** (optional): Faster inference

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/your-username/highway-guardian.git
cd highway-guardian
```

### 2. Start Backend
```bash
cd src
pip install -r requirements.txt
python main.py
```

Backend sẽ chạy tại: **http://localhost:8000**

### 3. Start Frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend sẽ chạy tại: **http://localhost:5173**

### 4. Access Application
- Open http://localhost:5173
- Login với Firebase account
- Chọn trang Detect hoặc Camera
- Enjoy! 🎉

## 📚 Documentation

### Setup & Installation:
- 📖 [SETUP_GUIDE.md](SETUP_GUIDE.md) - Hướng dẫn cài đặt chi tiết
- 🐛 Troubleshooting tips
- 🔧 Environment setup

### User Guides:
- 🎨 [TWO_STAGE_DETECTION_GUIDE.md](frontend/TWO_STAGE_DETECTION_GUIDE.md) - Hướng dẫn sử dụng
- 📊 API documentation
- 🎯 Use cases

### Developer Guides:
- 🔧 [CNN_CLASS_MAPPING_GUIDE.md](CNN_CLASS_MAPPING_GUIDE.md) - Cập nhật CNN mappings
- 📝 [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - Chi tiết refactoring
- 🏗️ [src/README.md](src/README.md) - Backend architecture

### Project Summary:
- 🎉 [FINAL_SUMMARY.md](FINAL_SUMMARY.md) - Tổng kết project
- ✅ Achievements
- 🔮 Future plans

## 🎨 Screenshots

### Login Page
![Login](docs/screenshots/login.png)
*Beautiful login with gradient background and toast notifications*

### Detect Page
![Detect](docs/screenshots/detect.png)
*Upload images and see detection results with bounding boxes*

### Camera Page
![Camera](docs/screenshots/camera.png)
*Real-time detection with FPS counter and stats*

## 🔧 Configuration

### Backend (`src/config/settings.py`)
```python
API_HOST = "0.0.0.0"
API_PORT = 8000
YOLO_MODELS_DIR = "models/yolo"
CNN_MODELS_DIR = "models/cnn"
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
```

### Frontend (`frontend/.env`)
```env
VITE_FIREBASE_API_KEY="your-api-key"
VITE_FIREBASE_AUTH_DOMAIN="your-domain"
VITE_FIREBASE_PROJECT_ID="your-project-id"
```

## 📊 Performance

| Mode | Speed | Accuracy | Use Case |
|------|-------|----------|----------|
| YOLO Only | ⚡⚡⚡ 30-60 FPS | ⭐⭐⭐ Good | Real-time, Demo |
| YOLO + CNN | ⚡⚡ 10-20 FPS | ⭐⭐⭐⭐ Excellent | Production |

## 🛠️ Tech Stack

### Backend:
- **FastAPI**: Modern Python web framework
- **Ultralytics YOLOv8**: Object detection
- **TensorFlow/Keras**: CNN classification
- **Pillow**: Image processing
- **Uvicorn**: ASGI server

### Frontend:
- **Vue.js 3**: Progressive JavaScript framework
- **Vite**: Fast build tool
- **Pinia**: State management
- **Vue Router**: Routing
- **Firebase**: Authentication
- **TypeScript**: Type safety

### ML/AI:
- **YOLOv8**: Traffic sign detection
- **MobileNetV2**: Traffic sign classification
- **OpenCV**: Image processing
- **NumPy**: Numerical computing

## 🔮 Roadmap

### Phase 1: Core Features ✅
- [x] YOLO detection
- [x] CNN classification
- [x] 2-stage pipeline
- [x] Real-time camera
- [x] Beautiful UI

### Phase 2: Enhancements 🚧
- [ ] Model versioning
- [ ] A/B testing
- [ ] Batch processing
- [ ] Video processing
- [ ] Export results

### Phase 3: Production 📋
- [ ] Docker optimization
- [ ] CI/CD pipeline
- [ ] Monitoring dashboard
- [ ] Auto-scaling
- [ ] Load balancing

### Phase 4: Advanced 🔮
- [ ] Mobile app
- [ ] Edge deployment
- [ ] Model quantization
- [ ] TensorRT optimization
- [ ] Multi-language support

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [GitHub](https://github.com/your-username)

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Vue.js team
- FastAPI team
- Firebase team
- All contributors

## 📞 Support

- 📧 Email: your-email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/highway-guardian/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-username/highway-guardian/discussions)

---

<div align="center">

**Made with ❤️ by Highway Guardian Team**

⭐ Star us on GitHub — it motivates us a lot!

</div>
